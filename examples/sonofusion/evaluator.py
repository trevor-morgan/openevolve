"""
Evaluator for Sonofusion Simulation Code.

Scores the simulation based on:
1. Physical Realism (Stability, conservation laws).
2. Proximity to Fusion Conditions (Lawson Criterion).
3. Numerical Robustness (Handling stiff collapse).
"""

import importlib.util
import sys
from typing import Any

import numpy as np


def evaluate(program_path: str) -> dict[str, Any]:
    """
    Evaluates the simulation code.
    """
    try:
        # Load module
        spec = importlib.util.spec_from_file_location("evolved_sim", program_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["evolved_sim"] = module
        spec.loader.exec_module(module)

        # Run simulation
        if not hasattr(module, "run_simulation"):
            return {"score": 0.0, "error": "Missing run_simulation function"}

        output = module.run_simulation()
        results = output.get("results", {})

        if not results.get("success", False):
            return {"score": 0.0, "error": f"Simulation failed: {results.get('error')}"}

        # Extract metrics
        T_max = results.get("T_max", 0.0)
        R_max = results.get("R_max", 1.0)
        R_min = results.get("R_min", 1.0)
        mach = results.get("mach_max", 0.0)

        # === SCORING ===
        score = 0.0

        # 1. Stability Check (0-0.3)
        # Expansion ratio should be significant but not infinite
        expansion_ratio = R_max / output["parameters"]["R0"]
        if 2.0 < expansion_ratio < 20.0:
            score += 0.2
        elif 1.1 < expansion_ratio <= 2.0:
            score += 0.1

        # Compression check
        if R_min < output["parameters"]["R0"] * 0.2:
            score += 0.1

        # 2. Temperature Metric (0-0.5)
        # Target: 10^7 K (Fusion)
        # Base Rayleigh-Plesset might hit 10^4 - 10^5 K
        # We use log scaling
        if T_max > 0:
            log_T = np.log10(T_max)
            # Map 3.0 (1000K) -> 0.0, 7.0 (10MK) -> 0.5
            t_score = (log_T - 3.0) / 8.0
            score += max(0.0, min(0.5, t_score))

        # 3. Density/Lawson Metric (0-0.2)
        # Need high density. For gas bubble, density ~ (R0/R_min)^3
        comp_ratio = results.get("compression_ratio", 1.0)
        if comp_ratio > 10.0:
            score += min(0.2, np.log10(comp_ratio) * 0.1)

        # Penalties for unphysical results
        if mach > 10.0:
            # Wall moving at Mach 10 is unlikely for simple SBSL without shock focusing
            score *= 0.5
        if T_max > 1e9:
            # Implausibly hot without radiation losses
            score *= 0.1

        metrics = {
            "T_max": T_max,
            "R_min": R_min,
            "compression_ratio": comp_ratio,
            "mach_max": mach,
        }

        return {
            "score": float(score),
            "metrics": metrics,
            "combined_score": float(score),  # Required key
        }

    except Exception as e:
        return {"score": 0.0, "error": str(e)}


if __name__ == "__main__":
    res = evaluate("initial_program.py")
    print(res)
