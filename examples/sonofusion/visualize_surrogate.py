"""
Visualize Sonofusion Surrogate Results.
"""

import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def visualize_surrogate(program_path, output_path="sonofusion_render.png"):
    # Load program
    spec = importlib.util.spec_from_file_location("sim", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim"] = module
    spec.loader.exec_module(module)

    # Run
    if hasattr(module, "run_simulation"):
        out = module.run_simulation()
        res = out["results"]
        params = out["parameters"]
    else:
        print("Invalid program format")
        return

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Radius vs Time
    t = np.array(res["trace_t"])
    R = np.array(res["trace_R"])
    R0 = params["R0"]

    # Normalize time to acoustic cycles
    freq = params["Drive_Freq"]
    t_cycles = t * freq

    ax1.plot(t_cycles, R / R0, "b-", label="Bubble Radius")
    ax1.set_ylabel("R / R0")
    ax1.set_xlabel("Acoustic Cycles")
    ax1.set_title(
        'Bubble Dynamics (Surrogate)\nR0={R0*1e6:.1f}um, f={freq/1e3:.1f}kHz, Pa={params.get("Drive_Amp",0)/1e5:.1f}bar'
    )
    ax1.grid(True)
    ax1.legend()

    # Temperature (Estimated)
    # T = T0 * (V0/V)^(gamma-1)
    # We can reconstruct T(t) trace from R(t)
    T0 = params["Liquid_Temp"]
    gamma = 1.4
    # Simple adiabatic estimate for trace
    T_trace = T0 * ((R0 / R) ** (3 * (gamma - 1)))

    ax2.semilogy(t_cycles, T_trace, "r-", label="Core Temperature")
    ax2.set_ylabel("Temperature (K)")
    ax2.set_xlabel("Acoustic Cycles")
    ax2.set_title('Core Temperature (Max: {res["T_max"]:.0f} K)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    visualize_surrogate("examples/sonofusion/openevolve_output/best/best_program.py")
