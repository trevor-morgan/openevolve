#!/usr/bin/env python3
"""
Visualization utilities for Magnetic Mirror configurations.

Creates publication-quality plots showing:
1. Magnetic field profiles (axial and 2D)
2. Coil geometry
3. Field line structure
4. Confinement metrics over evolution
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle


def plot_field_profile(config, ax=None, show_plasma=True):
    """Plot on-axis magnetic field profile."""
    from initial_program import calculate_field

    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))

    coil_set = config["coil_set"]
    plasma_length = config["plasma_length"]
    plasma_radius = config["plasma_radius"]

    # Calculate on-axis field
    z = np.linspace(-3, 3, 500)
    r = np.zeros_like(z)
    Bz, Br = calculate_field(coil_set, z, r)
    B = np.sqrt(Bz**2 + Br**2)

    ax.plot(z, B, "b-", linewidth=2, label="|B| on axis")
    ax.axhline(
        y=np.min(B), color="g", linestyle="--", alpha=0.5, label=f"B_min = {np.min(B):.3f} T"
    )
    ax.axhline(
        y=np.max(B), color="r", linestyle="--", alpha=0.5, label=f"B_max = {np.max(B):.3f} T"
    )

    if show_plasma:
        ax.axvspan(
            -plasma_length / 2, plasma_length / 2, alpha=0.2, color="orange", label="Plasma region"
        )

    # Mark coil positions
    for z_coil, r_coil, I in coil_set.coils:
        ax.axvline(x=z_coil, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Axial Position z (m)", fontsize=12)
    ax.set_ylabel("Magnetic Field |B| (T)", fontsize=12)
    ax.set_title("On-Axis Magnetic Field Profile", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_coil_geometry(config, ax=None):
    """Plot coil cross-section geometry."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 6))

    coil_set = config["coil_set"]
    plasma_length = config["plasma_length"]
    plasma_radius = config["plasma_radius"]

    # Plot coils as rectangles (cross-section view)
    coil_patches = []
    colors = []

    for z_coil, r_coil, I in coil_set.coils:
        # Coil cross-section (simplified as small rectangles)
        coil_width = 0.1
        coil_height = 0.1

        # Upper coil
        rect_upper = Rectangle(
            (z_coil - coil_width / 2, r_coil - coil_height / 2), coil_width, coil_height
        )
        coil_patches.append(rect_upper)

        # Lower coil (mirror symmetry)
        rect_lower = Rectangle(
            (z_coil - coil_width / 2, -r_coil - coil_height / 2), coil_width, coil_height
        )
        coil_patches.append(rect_lower)

        # Color by current direction
        color = "red" if I > 0 else "blue"
        colors.extend([color, color])

    # Add coil patches
    pc = PatchCollection(coil_patches, facecolors=colors, edgecolors="black", alpha=0.7)
    ax.add_collection(pc)

    # Plot plasma region
    plasma_rect = Rectangle(
        (-plasma_length / 2, -plasma_radius),
        plasma_length,
        2 * plasma_radius,
        fill=True,
        facecolor="orange",
        alpha=0.3,
        edgecolor="orange",
        linewidth=2,
        label="Plasma",
    )
    ax.add_patch(plasma_rect)

    # Axis line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Axial Position z (m)", fontsize=12)
    ax.set_ylabel("Radial Position r (m)", fontsize=12)
    ax.set_title("Coil Geometry (Cross-Section View)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", edgecolor="black", alpha=0.7, label="Coil (I > 0)"),
        Patch(facecolor="blue", edgecolor="black", alpha=0.7, label="Coil (I < 0)"),
        Patch(facecolor="orange", alpha=0.3, edgecolor="orange", label="Plasma region"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    return ax


def plot_field_contours(config, ax=None):
    """Plot 2D magnetic field magnitude contours."""
    from initial_program import calculate_field

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 5))

    coil_set = config["coil_set"]
    plasma_length = config["plasma_length"]
    plasma_radius = config["plasma_radius"]

    # Create 2D grid
    z = np.linspace(-3, 3, 200)
    r = np.linspace(0, 1.5, 100)
    Z, R = np.meshgrid(z, r)

    # Calculate field on grid
    B_mag = np.zeros_like(Z)
    for i in range(len(r)):
        Bz, Br = calculate_field(coil_set, z, np.full_like(z, r[i]))
        B_mag[i, :] = np.sqrt(Bz**2 + Br**2)

    # Plot contours
    levels = np.linspace(0, np.percentile(B_mag, 95), 20)
    cs = ax.contourf(Z, R, B_mag, levels=levels, cmap="plasma")
    ax.contour(Z, R, B_mag, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.5)

    plt.colorbar(cs, ax=ax, label="|B| (T)")

    # Mark plasma region
    ax.plot(
        [-plasma_length / 2, -plasma_length / 2, plasma_length / 2, plasma_length / 2],
        [0, plasma_radius, plasma_radius, 0],
        "w--",
        linewidth=2,
        label="Plasma boundary",
    )

    # Mark coil positions
    for z_coil, r_coil, I in coil_set.coils:
        marker = "^" if I > 0 else "v"
        ax.plot(z_coil, r_coil, marker, color="white", markersize=10, markeredgecolor="black")

    ax.set_xlabel("Axial Position z (m)", fontsize=12)
    ax.set_ylabel("Radial Position r (m)", fontsize=12)
    ax.set_title("Magnetic Field Magnitude |B|", fontsize=14)

    return ax


def plot_field_lines(config, ax=None, n_lines=15):
    """Plot magnetic field lines."""
    from initial_program import calculate_field

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))

    coil_set = config["coil_set"]
    plasma_length = config["plasma_length"]
    plasma_radius = config["plasma_radius"]

    # Create grid for streamplot
    z = np.linspace(-3, 3, 100)
    r = np.linspace(-1.5, 1.5, 80)
    Z, R = np.meshgrid(z, r)

    # Calculate field components
    Bz_grid = np.zeros_like(Z)
    Br_grid = np.zeros_like(Z)

    for i in range(len(r)):
        Bz, Br = calculate_field(coil_set, z, np.full_like(z, abs(r[i])))
        Bz_grid[i, :] = Bz
        # Br changes sign with r
        Br_grid[i, :] = Br * np.sign(r[i])

    # Plot field lines
    ax.streamplot(Z, R, Bz_grid, Br_grid, density=1.5, color="blue", linewidth=0.8, arrowsize=0.8)

    # Mark plasma region
    plasma_rect = Rectangle(
        (-plasma_length / 2, -plasma_radius),
        plasma_length,
        2 * plasma_radius,
        fill=False,
        edgecolor="orange",
        linewidth=2,
    )
    ax.add_patch(plasma_rect)

    # Mark coils
    for z_coil, r_coil, I in coil_set.coils:
        color = "red" if I > 0 else "blue"
        ax.plot(z_coil, r_coil, "s", color=color, markersize=8, markeredgecolor="black")
        ax.plot(z_coil, -r_coil, "s", color=color, markersize=8, markeredgecolor="black")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Axial Position z (m)", fontsize=12)
    ax.set_ylabel("Radial Position r (m)", fontsize=12)
    ax.set_title("Magnetic Field Lines", fontsize=14)
    ax.set_aspect("equal")

    return ax


def plot_particle_trace(artifacts, ax=None):
    """Plot particle trace statistics."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))

    if "particle_trace" not in artifacts:
        ax.text(0.5, 0.5, "No particle trace data", ha="center", va="center")
        return ax

    trace = artifacts["particle_trace"]
    z_final = np.array(trace.get("z_final", []))
    conf_times = np.array(trace.get("confinement_times", []))

    # Histogram of final positions
    ax.hist(z_final, bins=30, alpha=0.7, color="green", label="Final Axial Positions")
    ax.set_xlabel("Axial Position z (m)", fontsize=12)
    ax.set_ylabel("Particle Count", fontsize=12)
    ax.set_title("Particle Loss Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add confinement stats text
    mean_time = np.mean(conf_times)
    ax.text(
        0.05,
        0.95,
        f"Mean Confinement: {mean_time * 1e6:.1f} μs",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    return ax


def create_summary_figure(config, metrics=None, artifacts=None, save_path=None):
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(15, 12))

    # Panel 1: Field profile
    ax1 = fig.add_subplot(3, 2, 1)
    plot_field_profile(config, ax=ax1)

    # Panel 2: Coil geometry
    ax2 = fig.add_subplot(3, 2, 2)
    plot_coil_geometry(config, ax=ax2)

    # Panel 3: Field contours
    ax3 = fig.add_subplot(3, 2, 3)
    plot_field_contours(config, ax=ax3)

    # Panel 4: Field lines
    ax4 = fig.add_subplot(3, 2, 4)
    plot_field_lines(config, ax=ax4)

    # Panel 5: Particle Traces (New)
    ax5 = fig.add_subplot(3, 2, 5)
    if artifacts:
        plot_particle_trace(artifacts, ax=ax5)
    else:
        ax5.text(0.5, 0.5, "No Artifacts Available", ha="center")

    # Add metrics text if available
    if metrics:
        metrics_text = (
            f"Mirror Ratio: {metrics.get('mirror_ratio', 'N/A'):.2f}\n"
            f"Well Depth: {metrics.get('well_depth', 'N/A'):.3f}\n"
            f"B_center: {metrics.get('B_center', 'N/A'):.3f} T\n"
            f"Confinement: {metrics.get('mean_confinement_time', 0) * 1e6:.1f} μs\n"
            f"Stability: {'Yes' if metrics.get('radial_stability') else 'No'}"
        )
        fig.text(
            0.02,
            0.02,
            metrics_text,
            fontsize=10,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize magnetic mirror configurations")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument(
        "--program", type=str, default="initial_program.py", help="Path to program file"
    )
    parser.add_argument(
        "--output", type=str, default="mirror_config.png", help="Output figure path"
    )
    parser.add_argument("--show", action="store_true", help="Show interactive plot")

    args = parser.parse_args()

    metrics = None
    artifacts = None

    # Determine which program to load
    if args.checkpoint:
        program_path = os.path.join(args.checkpoint, "best_program.py")

        # Try to load info json
        info_path = os.path.join(args.checkpoint, "best_program_info.json")
        if os.path.exists(info_path):
            import json

            with open(info_path) as f:
                info = json.load(f)
                metrics = info.get("metrics")

                # Try to load full program data for artifacts
                prog_id = info.get("id")
                if prog_id:
                    prog_json_path = os.path.join(args.checkpoint, "programs", f"{prog_id}.json")
                    if os.path.exists(prog_json_path):
                        with open(prog_json_path) as pf:
                            prog_data = json.load(pf)
                            # Check metadata for artifacts
                            meta = prog_data.get("metadata", {})
                            artifacts = meta.get("artifacts")
    else:
        program_path = args.program

    # Load the program
    import importlib.util

    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["initial_program"] = module  # For imports
    spec.loader.exec_module(module)

    # Get configuration
    if hasattr(module, "run_design"):
        config = module.run_design()
    else:
        config = module.design_mirror_configuration()

    # Create figure
    fig = create_summary_figure(config, metrics=metrics, artifacts=artifacts, save_path=args.output)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
