"""
Surrogate Model for Sonofusion.
Learned via PhysicsML from unstable Keller-Miksis traces.

This model replaces the unstable ODE integration with a numerically stable
analytical ansatz that captures the key phenomenology:
1. Acoustic driving (Sine term)
2. Non-linear collapse (Periodic Gaussian dip)
3. Isentropic heating (derived from R_min)

This stable "backbone" allows OpenEvolve to focus on optimizing the
parameters (phase, depth, sharpness) and adding physics terms (plasma, fusion)
without fighting the ODE solver.
"""

from typing import Any

import numpy as np

# === Learned Surrogate Parameters ===
SURROGATE_A = 9.4130  # Oscillation amplitude
SURROGATE_B = -1.7721  # Phase shift
SURROGATE_C = -8.9118  # Collapse depth
SURROGATE_DECAY = 0.0000  # Collapse sharpness

# Physical Constants
P0 = 101325.0
POLY_GAMMA = 1.4


def run_simulation() -> dict[str, Any]:
    """Main entry point."""

    # === Parameters to Evolve ===
    # Now we evolve the SHAPE of the collapse, not just the drive
    params = {
        "R0": 6.0e-6,
        "Drive_Freq": 20.0e3,
        "Liquid_Temp": 273.0,
        "Collapse_Sharpness": SURROGATE_DECAY,  # Evolve this!
        "Collapse_Depth": SURROGATE_C,  # Evolve this!
        "Shock_Efficiency": 0.1,  # New parameter: how well does the shock focus?
    }

    results = simulate_surrogate(params)

    return {
        "parameters": params,
        "results": results,
        "description": "Stable PhysicsML Surrogate with Shock",
    }


def simulate_surrogate(params: dict[str, Any]) -> dict[str, Any]:
    """
    Run the stable surrogate simulation.
    R(t) is computed analytically, ensuring stability.
    """
    R0 = params["R0"]
    freq = params["Drive_Freq"]
    period = 1.0 / freq

    # Time grid (high resolution for fusion check)
    t_steps = 10000
    t = np.linspace(0, 3 * period, t_steps)
    t_norm = t * freq  # Cycles

    # === The Learned Physics Model ===
    # R_norm = 1 + osc - collapse
    a = SURROGATE_A
    b = SURROGATE_B
    c = params["Collapse_Depth"]  # Allow evolution to deepen the collapse
    decay = params["Collapse_Sharpness"]  # Allow evolution to sharpen the collapse

    # Analytical Trajectory
    osc = a * np.sin(2 * np.pi * t_norm + b)
    collapse = c * np.exp(-decay * (np.sin(np.pi * (t_norm + 0.2)) ** 2))

    R_norm = 1.0 + osc - collapse

    # Physics Constraints
    # Van der Waals Hard Core Limit
    # Atoms have finite volume. R cannot go below ~ R0 / 8.5
    VDW_H_RATIO = 8.5
    R_min_limit = 1.0 / VDW_H_RATIO

    # Radius cannot be negative (Surrogate might predict it if c is too large)
    R_norm = np.maximum(R_norm, R_min_limit)

    R = R0 * R_norm
    R_dot = np.gradient(R, t)

    # === Derived Physics ===
    R_min = np.min(R)
    R_max = np.max(R)

    # Mach number
    C_L = 1481.0
    mach_max = np.max(np.abs(R_dot)) / C_L

    # Temperature (Isentropic compression)
    # T = T0 * (V0/V)^(gamma-1)
    compression = (R0 / R_min) ** 3
    T_adiabatic = params["Liquid_Temp"] * (compression ** (POLY_GAMMA - 1))

    # Shock Heating (Rankine-Hugoniot approx for strong shock)
    # Only active if wall speed exceeds speed of sound (Mach > 1)
    T_shock = 0.0
    gamma = POLY_GAMMA
    if mach_max > 1.0:
        # Strong shock limit: T2/T1 ~ 2*gamma*(gamma-1)/(gamma+1)^2 * M^2
        # We apply this amplification to the already compressed core
        coeff = (2 * gamma * (gamma - 1)) / ((gamma + 1) ** 2)
        T_shock = T_adiabatic * coeff * (mach_max**2)

    # Total Temperature
    efficiency = params.get("Shock_Efficiency", 0.0)
    T_max = T_adiabatic + efficiency * T_shock

    # Fusion Rate Estimate (Simple)
    # Rate ~ n^2 * <sigma v>
    # <sigma v> is highly non-linear with T
    fusion_yield = 0.0
    if T_max > 1e6:
        # Toy fusion model for optimization signal
        fusion_yield = 1e-20 * (T_max**2) * compression

    return {
        "success": True,
        "R_min": float(R_min),
        "R_max": float(R_max),
        "compression_ratio": float(R_max / R_min),
        "T_max": float(T_max),
        "mach_max": float(mach_max),
        "fusion_yield": float(fusion_yield),
        "trace_t": t.tolist(),
        "trace_R": R.tolist(),
        "shock_amplification": float(T_shock / T_adiabatic) if T_adiabatic > 0 else 0.0,
    }
