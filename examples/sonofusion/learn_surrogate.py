"""
PhysicsML Surrogate Learner for Sonofusion.

Goal: Learn a numerically stable surrogate model from unstable Keller-Miksis traces.
1. Generates training data by running the unstable simulation with random parameters.
2. Captures partial trajectories (before crash/overflow).
3. Fits a stable analytical ansatz (Surrogate Model) to the collapse phase.
4. Generates a new `surrogate_seed.py` for OpenEvolve.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Import the unstable simulation
sys.path.append(os.path.dirname(__file__))
from initial_program import simulate_bubble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhysicsML")


def generate_data(n_samples=50):
    """Generate traces from the unstable physics model."""
    logger.info(f"Generating {n_samples} training traces...")

    traces = []

    for i in range(n_samples):
        # Randomize parameters slightly to get robust coverage
        params = {
            "R0": np.random.uniform(2.0e-6, 10.0e-6),
            "Drive_Freq": np.random.uniform(15e3, 30e3),
            "Drive_Amp": np.random.uniform(1.0e5, 1.5e5),  # Lower amp to get some non-crashing runs
            "Liquid_Temp": 293.0,
            "Accom_Coeff": 0.4,
        }

        result = simulate_bubble(params)

        # We want traces that show collapse, even if they eventually crash
        if "trace_t" in result and "trace_R" in result:
            t = np.array(result["trace_t"])
            R = np.array(result["trace_R"])

            # Normalize
            t_norm = t * params["Drive_Freq"]  # Cycles
            R_norm = R / params["R0"]

            traces.append((t_norm, R_norm, params))

    logger.info(f"Collected {len(traces)} valid/partial traces.")
    return traces


def surrogate_ansatz(t, a, b, c, decay):
    """
    Proposed stable surrogate ansatz for bubble oscillation.
    Combines driving frequency response with non-linear collapse spikes.

    R(t) = 1 + a*sin(2*pi*t + b) - c * exp(-decay * (sin(pi*t)^2))

    This creates smooth oscillations with sharp "dips" (collapses) at integer t,
    without requiring numerical integration.
    """
    # Base oscillation
    osc = a * np.sin(2 * np.pi * t + b)

    # Sharp collapse feature (periodic)
    # exp(-k * sin^2(x)) creates periodic spikes
    collapse = c * np.exp(-decay * (np.sin(np.pi * (t + 0.2)) ** 2))

    return 1.0 + osc - collapse


def learn_model(traces):
    """Fit the surrogate ansatz to the noisy/partial data."""
    logger.info("Fitting surrogate model via PhysicsML (Curve Fitting)...")

    all_t = []
    all_R = []

    for t, R, _ in traces:
        # Only use the first few cycles to learn the stable periodic behavior
        mask = t < 3.0
        all_t.extend(t[mask])
        all_R.extend(R[mask])

    if not all_t:
        logger.error("No data to fit!")
        return None

    all_t = np.array(all_t)
    all_R = np.array(all_R)

    # Initial guess: a=0.2 (oscillation), b=0 (phase), c=0.5 (collapse depth), decay=10 (sharpness)
    p0 = [0.2, 0.0, 0.5, 10.0]

    try:
        popt, _ = curve_fit(surrogate_ansatz, all_t, all_R, p0=p0, maxfev=10000)
        logger.info(
            f"Learned Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, decay={popt[3]:.3f}"
        )
        return popt
    except Exception as e:
        logger.error(f"Fitting failed: {e}")
        return p0  # Fallback


def generate_surrogate_code(params):
    """Generate the new seed program using the learned surrogate."""
    a, b, c, decay = params

    code = f'''"""
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

import numpy as np
from typing import Dict, Any

# === Learned Surrogate Parameters ===
SURROGATE_A = {a:.4f}      # Oscillation amplitude
SURROGATE_B = {b:.4f}      # Phase shift
SURROGATE_C = {c:.4f}      # Collapse depth
SURROGATE_DECAY = {decay:.4f} # Collapse sharpness

# Physical Constants
P0 = 101325.0
POLY_GAMMA = 1.4

def run_simulation() -> Dict[str, Any]:
    """Main entry point."""

    # === Parameters to Evolve ===
    # Now we evolve the SHAPE of the collapse, not just the drive
    params = {{
        "R0": 6.0e-6,
        "Drive_Freq": 20.0e3,
        "Liquid_Temp": 273.0,
        "Collapse_Sharpness": SURROGATE_DECAY, # Evolve this!
        "Collapse_Depth": SURROGATE_C          # Evolve this!
    }}

    results = simulate_surrogate(params)

    return {{
        "parameters": params,
        "results": results,
        "description": "Stable PhysicsML Surrogate"
    }}

def simulate_surrogate(params: Dict[str, Any]) -> Dict[str, Any]:
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
    t_norm = t * freq # Cycles

    # === The Learned Physics Model ===
    # R_norm = 1 + osc - collapse
    a = SURROGATE_A
    b = SURROGATE_B
    c = params["Collapse_Depth"]        # Allow evolution to deepen the collapse
    decay = params["Collapse_Sharpness"] # Allow evolution to sharpen the collapse

    # Analytical Trajectory
    osc = a * np.sin(2 * np.pi * t_norm + b)
    collapse = c * np.exp(-decay * (np.sin(np.pi * (t_norm + 0.2))**2))

    R_norm = 1.0 + osc - collapse

    # Physics Constraints
    # Radius cannot be negative (Surrogate might predict it if c is too large)
    # Soft clamp at R_min_limit
    R_min_limit = 0.01 # 1% of R0
    R_norm = np.maximum(R_norm, R_min_limit)

    R = R0 * R_norm
    R_dot = np.gradient(R, t)

    # === Derived Physics ===
    R_min = np.min(R)
    R_max = np.max(R)

    # Temperature (Isentropic compression)
    # T = T0 * (V0/V)^(gamma-1)
    # At R_min:
    compression = (R0 / R_min)**3
    T_max = params["Liquid_Temp"] * (compression ** (POLY_GAMMA - 1))

    # Mach number
    C_L = 1481.0
    mach_max = np.max(np.abs(R_dot)) / C_L

    # Fusion Rate Estimate (Simple)
    # Rate ~ n^2 * <sigma v>
    # <sigma v> is highly non-linear with T
    fusion_yield = 0.0
    if T_max > 1e6:
        # Toy fusion model for optimization signal
        fusion_yield = 1e-20 * (T_max**2) * compression

    return {{
        "success": True,
        "R_min": float(R_min),
        "R_max": float(R_max),
        "compression_ratio": float(R_max/R_min),
        "T_max": float(T_max),
        "mach_max": float(mach_max),
        "fusion_yield": float(fusion_yield),
        "trace_t": t.tolist(),
        "trace_R": R.tolist()
    }}
'''
    return code


if __name__ == "__main__":
    traces = generate_data()
    if traces:
        params = learn_model(traces)
        if params is not None:
            code = generate_surrogate_code(params)
            with open("examples/sonofusion/surrogate_seed.py", "w") as f:
                f.write(code)
            logger.info("Generated examples/sonofusion/surrogate_seed.py")
