"""
Simulation of Single-Bubble Sonoluminescence (SBSL) for Fusion Discovery.

IMPROVED SEED (Research-Based):
1. Log-Radius Transformation x = ln(R/R0) for numerical stability.
2. Deuterated Acetone parameters (Taleyarkhan et al. regime).
3. Radau solver for stiff ODEs.
4. Robust Van der Waals hard-core repulsion.

Variables to Evolve:
- Drive parameters (Amp, Freq)
- Liquid properties (Viscosity, Sigma, Speed of Sound)
- Gas EOS parameters (Excluded volume, Adiabatic index)
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

# === Physical Constants (Deuterated Acetone approx) ===
P0 = 101325.0  # Ambient pressure [Pa]
RHO_L = 790.0  # Liquid density (Acetone) [kg/m^3]
SIGMA = 0.0237  # Surface tension [N/m]
VISCOSITY = 0.00032  # Dynamic viscosity [Pa*s]
C_L = 1174.0  # Speed of sound in liquid [m/s]
P_VAPOR = 30000.0  # Vapor pressure [Pa] (Significant for acetone)

# Gas Properties (Deuterium/Argon mix)
POLY_GAMMA = 1.4  # Adiabatic index
VDW_H_RATIO = 8.5  # Hard core radius divisor (R0/h = 8.5)


def run_simulation() -> dict[str, Any]:
    """Main entry point for OpenEvolve."""

    # === Parameters to Evolve ===
    params = {
        "R0": 6.0e-6,  # Ambient radius [m] (Typical ~5-10 um)
        "Drive_Freq": 19.3e3,  # Frequency [Hz] (Taleyarkhan used ~19.3 kHz)
        "Drive_Amp": 15.0e5,  # Amplitude [Pa] (15 bar, extreme drive)
        "Liquid_Temp": 273.0,  # Temperature [K] (0 C)
        "Accom_Coeff": 0.4,  # Thermal accommodation coefficient
    }

    results = simulate_bubble(params)

    return {
        "parameters": params,
        "results": results,
        "description": "Log-Radius KM Model (Acetone)",
    }


def keller_miksis_log_ode(t, y, params):
    """
    Keller-Miksis equation in log-variables.
    x = ln(R/R0)
    u = R_dot

    Why? R ranges from 10^-4 to 10^-8. Linear R ODE is unstable.
    """
    x, u = y

    # Recover R
    R0 = params["R0"]
    R = R0 * np.exp(x)

    # Physics Parameters
    Pa = params["Drive_Amp"]
    w = 2 * np.pi * params["Drive_Freq"]
    h = R0 / VDW_H_RATIO

    # Acoustic Drive
    P_drive = -Pa * np.sin(w * t)

    # Gas Pressure (Van der Waals)
    # P_g = (P0 + 2s/R0 - Pv) * ((R0^3 - h^3)/(R^3 - h^3))^gamma
    # Note: vapor pressure adds to gas pressure logic

    # Hard core handling
    # If R approaches h, pressure skyrockets.
    # Use soft limit: vol_ratio = (R0^3 - h^3) / max(R^3 - h^3, epsilon)

    R3 = R**3
    h3 = h**3

    # Stability: prevent R < h
    if h >= R:
        # Inside hard core - elastic bounce (stiff spring)
        # Model as ultra-high pressure
        P_gas = 1e15
    else:
        P_stat = P0 + (2 * SIGMA / R0) - P_VAPOR
        vol_term = (R0**3 - h3) / (R3 - h3)
        P_gas = P_stat * (vol_term**POLY_GAMMA) + P_VAPOR

    # Liquid Pressure at wall
    # P_liq = P_gas - 2s/R - 4mu*u/R
    P_liq = P_gas - (2 * SIGMA / R) - (4 * VISCOSITY * u / R)

    # Keller-Miksis Terms
    # (1 - u/c)R * u_dot + (1.5 - u/2c)u^2 = (1 + u/c)/rho * (P_liq - P_inf) + R/rho/c * dP_liq/dt
    # Neglect dP_liq/dt for stability in seed (Evolve to add it back!)

    P_inf = P0 + P_drive

    # Numerator
    # (1 + u/c)/rho * (P_liq - P_inf) - (1.5 - u/2c)u^2
    term1 = (1 + u / C_L) * (P_liq - P_inf) / RHO_L
    term2 = (1.5 - u / (2 * C_L)) * (u**2)
    numerator = term1 - term2

    # Denominator
    # (1 - u/c)R
    denominator = (1 - u / C_L) * R

    u_dot = numerator / denominator

    return [u / R, u_dot]  # dx/dt = u/R, du/dt = u_dot


def simulate_bubble(params: dict[str, Any]) -> dict[str, Any]:
    """Run simulation with stiff solver."""

    R0 = params["R0"]
    freq = params["Drive_Freq"]
    period = 1.0 / freq

    # Initial conditions: Equilibrium
    # x = ln(1) = 0
    # u = 0
    y0 = [0.0, 0.0]

    # Integrate 3 cycles to settle transients
    t_span = (0, 3 * period)

    try:
        # Radau is excellent for stiff problems with algebraic constraints
        sol = solve_ivp(
            lambda t, y: keller_miksis_log_ode(t, y, params),
            t_span,
            y0,
            method="Radau",
            rtol=1e-5,
            atol=1e-8,
            max_step=1e-9,  # Force fine steps for collapse
        )

        # Reconstruct state regardless of success (to get partial traces)
        x_trace = sol.y[0] if hasattr(sol, "y") else []
        u_trace = sol.y[1] if hasattr(sol, "y") else []
        t_trace = sol.t.tolist() if hasattr(sol, "t") else []

        R_trace = (R0 * np.exp(x_trace)).tolist() if len(x_trace) > 0 else []

        if not sol.success:
            return {"success": False, "error": sol.message, "trace_t": t_trace, "trace_R": R_trace}

        # Analyze collapse
        R_min = np.min(R_trace)
        R_max = np.max(R_trace)

        # Temperature Estimate
        # Isentropic compression of gas core
        h = R0 / VDW_H_RATIO

        if R_min <= h:
            # Hit hard core
            comp_ratio_eff = (R0**3 - h**3) / (1e-20)  # Limit
        else:
            comp_ratio_eff = (R0**3 - h**3) / (R_min**3 - h**3)

        T_max = params["Liquid_Temp"] * (comp_ratio_eff ** (POLY_GAMMA - 1))

        # Mach number
        mach_max = np.max(np.abs(u_trace)) / C_L

        return {
            "success": True,
            "R_min": float(R_min),
            "R_max": float(R_max),
            "compression_ratio": float(R_max / R_min) if R_min > 0 else 0,
            "T_max": float(T_max),
            "mach_max": float(mach_max),
            "time_steps": len(sol.t),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    out = run_simulation()
    if out["results"].get("success"):
        print(f"Success! T_max: {out['results']['T_max']:.2e} K")
        print(f"Compression: {out['results']['compression_ratio']:.2f}")
    else:
        print(f"Failed: {out['results'].get('error')}")
