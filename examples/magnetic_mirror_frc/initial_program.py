"""
Magnetic Mirror / Field-Reversed Configuration Optimizer

This program defines a magnetic field configuration for plasma confinement.
The goal is to evolve coil parameters to maximize plasma confinement while
maintaining MHD stability.

Physics Background:
- Magnetic mirrors confine plasma by reflecting particles at high-field regions
- The mirror ratio R = B_max/B_min determines the loss cone angle
- Minimum-B configurations (magnetic wells) provide MHD stability
- Field-Reversed Configurations (FRCs) achieve high beta with closed field lines

Key Metrics:
- Mirror ratio: Higher = smaller loss cone = better confinement
- Magnetic well depth: Positive = MHD stable
- Beta limit: Higher achievable beta = more fusion power density
- Confinement parameter: n*tau_E product
"""

from typing import Any

import numpy as np

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]


class CoilSet:
    """Represents a set of circular coils for magnetic field generation."""

    def __init__(self):
        self.coils = []  # List of (z_position, radius, current) tuples

    def add_coil(self, z: float, radius: float, current: float):
        """Add a circular coil at axial position z with given radius and current."""
        self.coils.append((z, radius, current))

    def clear(self):
        """Remove all coils."""
        self.coils = []


def biot_savart_loop(
    z_eval: np.ndarray, r_eval: np.ndarray, z_coil: float, r_coil: float, current: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate magnetic field from a circular current loop using Biot-Savart law.
    Uses elliptic integral formulation for accuracy.

    Args:
        z_eval: Axial positions to evaluate field [m]
        r_eval: Radial positions to evaluate field [m]
        z_coil: Axial position of coil center [m]
        r_coil: Radius of coil [m]
        current: Coil current [A]

    Returns:
        Bz, Br: Axial and radial magnetic field components [T]
    """
    from scipy.special import ellipe, ellipk

    # Relative axial position
    z = z_eval - z_coil
    rho = np.abs(r_eval)  # Radial coordinate

    # Initialize output arrays
    Bz = np.zeros_like(z_eval, dtype=float)
    Br = np.zeros_like(r_eval, dtype=float)

    # On-axis case (rho ≈ 0): Use simple formula
    on_axis = rho < 1e-10
    if np.any(on_axis):
        # B_z = μ₀ * I * a² / (2 * (a² + z²)^(3/2))
        a = r_coil
        denom = (a**2 + z[on_axis] ** 2) ** 1.5
        Bz[on_axis] = MU_0 * current * a**2 / (2 * denom)
        Br[on_axis] = 0.0

    # Off-axis case: Use elliptic integrals
    off_axis = ~on_axis
    if np.any(off_axis):
        z_off = z[off_axis]
        rho_off = rho[off_axis]
        a = r_coil

        # Standard formulation using elliptic integrals
        # See: https://tiggerntatie.github.io/emagnet/offaxis/iloopalioff.htm
        alpha_sq = a**2 + rho_off**2 + z_off**2 - 2 * a * rho_off
        beta_sq = a**2 + rho_off**2 + z_off**2 + 2 * a * rho_off
        beta = np.sqrt(beta_sq)

        # k² parameter for elliptic integrals
        k_sq = 1 - alpha_sq / beta_sq
        k_sq = np.clip(k_sq, 0, 1 - 1e-10)

        # Elliptic integrals K(k²) and E(k²)
        K = ellipk(k_sq)
        E = ellipe(k_sq)

        # Common factor
        C = MU_0 * current / np.pi

        # Axial field B_z
        Bz[off_axis] = (
            C / (2 * np.sqrt(alpha_sq) * beta) * ((a**2 - rho_off**2 - z_off**2) * E / alpha_sq + K)
        )

        # Radial field B_r
        Br[off_axis] = (
            C
            * z_off
            / (2 * np.sqrt(alpha_sq) * beta * rho_off)
            * ((a**2 + rho_off**2 + z_off**2) * E / alpha_sq - K)
        )

    return Bz, Br


def calculate_field(
    coil_set: CoilSet, z: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate total magnetic field from all coils.

    Args:
        coil_set: CoilSet object containing coil definitions
        z: Axial positions [m]
        r: Radial positions [m]

    Returns:
        Bz, Br: Total axial and radial field components [T]
    """
    Bz_total = np.zeros_like(z)
    Br_total = np.zeros_like(r)

    for z_coil, r_coil, current in coil_set.coils:
        Bz, Br = biot_savart_loop(z, r, z_coil, r_coil, current)
        Bz_total += Bz
        Br_total += Br

    return Bz_total, Br_total


# EVOLVE-BLOCK-START
def design_mirror_configuration() -> dict[str, Any]:
    """
    Design a magnetic mirror configuration for plasma confinement.

    This function should return a dictionary containing:
    - 'coil_set': CoilSet object with coil definitions
    - 'plasma_length': Axial extent of plasma region [m]
    - 'plasma_radius': Radial extent of plasma region [m]
    - 'description': String describing the configuration

    The goal is to maximize:
    1. Mirror ratio (B_max/B_min) for particle confinement
    2. Magnetic well depth for MHD stability
    3. Field uniformity in the central region
    4. Achievable plasma beta

    Physical constraints:
    - Coil currents should be < 10 MA (engineering limit)
    - Coil radii should be > plasma radius (no overlap)
    - Mirror ratio > 2 for meaningful confinement
    """

    coils = CoilSet()

    # Basic magnetic mirror: two mirror coils at the ends
    # This is a simple configuration - evolution should improve it!

    # Mirror coils (high field regions at z = +/- 1.5m)
    mirror_current = 2e6  # 2 MA
    mirror_radius = 0.5  # 0.5 m
    mirror_z = 1.5  # 1.5 m from center

    coils.add_coil(z=-mirror_z, radius=mirror_radius, current=mirror_current)
    coils.add_coil(z=+mirror_z, radius=mirror_radius, current=mirror_current)

    # Central coil (lower field for plasma region)
    central_current = 0.5e6  # 0.5 MA
    central_radius = 0.8  # 0.8 m

    coils.add_coil(z=0.0, radius=central_radius, current=central_current)

    return {
        "coil_set": coils,
        "plasma_length": 2.0,  # 2 m plasma length
        "plasma_radius": 0.3,  # 0.3 m plasma radius
        "description": "Basic two-mirror configuration with central solenoid",
    }


# EVOLVE-BLOCK-END


def run_design():
    """Entry point for OpenEvolve evaluation."""
    return design_mirror_configuration()


if __name__ == "__main__":
    # Test the configuration
    config = design_mirror_configuration()
    print(f"Configuration: {config['description']}")
    print(f"Number of coils: {len(config['coil_set'].coils)}")
    print(f"Plasma region: L={config['plasma_length']}m, R={config['plasma_radius']}m")

    # Quick field calculation test
    z_test = np.linspace(-2, 2, 100)
    r_test = np.zeros_like(z_test)
    Bz, Br = calculate_field(config["coil_set"], z_test, r_test)

    print("\nOn-axis field:")
    print(f"  B_min = {np.min(Bz):.4f} T")
    print(f"  B_max = {np.max(Bz):.4f} T")
    print(f"  Mirror ratio = {np.max(Bz) / np.min(Bz):.2f}")
