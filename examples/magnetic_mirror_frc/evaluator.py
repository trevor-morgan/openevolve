"""
Evaluator for Magnetic Mirror / FRC Configurations

This evaluator calculates key plasma physics metrics for magnetic confinement:

1. Mirror Ratio (R = B_max/B_min)
   - Determines loss cone angle: sin²(θ_loss) = 1/R
   - Higher R = better particle confinement

2. Magnetic Well Depth
   - U = (B_edge - B_center) / B_center
   - Positive well = MHD stable (minimum-B)

3. Field Line Curvature
   - Good curvature (toward plasma) = stable
   - Bad curvature = flute instability risk

4. Confinement Figure of Merit
   - Estimated n*tau_E product
   - Accounts for loss cone, instabilities

5. Engineering Feasibility
   - Coil current limits
   - Field strength at coils
   - Geometric constraints

The scoring emphasizes:
- HIGH mirror ratio for classical confinement
- POSITIVE magnetic well for MHD stability
- UNIFORM central field for plasma equilibrium
- PRACTICAL coil parameters for engineering feasibility
"""

import json
import sys
import traceback
from typing import Any

import numpy as np

# =============================================================================
# NOVEL PHYSICS VARIABLES - Discovered by Collaborative Multi-Agent System
# These capture physics beyond textbook MHD stability criteria
# =============================================================================


def compute_bounce_averaged_stability_index(metrics: dict, artifacts: dict) -> float:
    """
    Compute bounce-averaged stability index from evaluation data.

    This metric captures how much of the confined velocity space has
    favorable (negative) bounce-averaged curvature, weighted by the
    time particles spend at each pitch angle.

    Key insight: MHD well depth is volume-averaged, but particles with
    different pitch angles sample the field differently. A configuration
    might have good average stability but poor stability for a critical
    subset of the particle distribution.

    Args:
        metrics: Dictionary containing mirror_ratio, well_depth, B_center, B_mirror
        artifacts: Dictionary containing field profile data

    Returns:
        Bounce-averaged stability index (0-1, higher is better)
    """
    try:
        mirror_ratio = metrics.get("mirror_ratio", 1.0)
        well_depth = metrics.get("well_depth", 0.0)
        center_curvature = metrics.get("center_curvature", 0.0)
        field_uniformity = metrics.get("field_uniformity", 0.5)

        # Loss cone angle determines which pitch angles are confined
        if mirror_ratio > 1:
            sin2_loss = 1.0 / mirror_ratio
            loss_cone_angle = np.arcsin(np.sqrt(sin2_loss))
        else:
            return 0.0  # No confinement

        # Sample pitch angles in confined region
        n_pitch = 50
        # Confined particles have |pitch| > sin(loss_cone_angle)
        pitch_min = np.sin(loss_cone_angle)
        pitch_angles = np.linspace(pitch_min, 1.0, n_pitch)

        # For each pitch angle, estimate bounce-averaged curvature contribution
        stability_contributions = []

        for pitch in pitch_angles:
            # Bounce period scaling: τ_b ∝ 1/sqrt(1 - sin²θ/sin²θ_loss)
            # Particles near loss cone spend more time at turning points
            pitch_factor = 1.0 / np.sqrt(1 - (pitch_min / pitch) ** 2 + 1e-6)

            # Curvature contribution: depends on where particle turns
            # Particles with small pitch (near loss cone) sample more of the field
            # Particles with large pitch stay near center

            # Weight by how much of mirror region particle samples
            mirror_sampling = 1.0 - pitch  # Low pitch = more mirror sampling

            # Stability contribution: positive well depth helps more for particles
            # that sample the edge regions
            curvature_contrib = well_depth * (1 + mirror_sampling) + center_curvature * pitch

            # Weight by bounce time (particles spend more time where they move slowly)
            stability_contributions.append(curvature_contrib * pitch_factor)

        # Average over pitch angle distribution (uniform in cos(pitch))
        weights = np.cos(np.arcsin(pitch_angles))
        weights /= np.sum(weights)

        bounce_avg_stability = np.sum(np.array(stability_contributions) * weights)

        # Normalize to 0-1 range
        # Typical good values: 0.5-2.0, map to 0.5-1.0
        normalized = 0.5 + 0.5 * np.tanh(bounce_avg_stability - 0.5)

        # Penalize low field uniformity (indicates bad regions exist)
        normalized *= field_uniformity

        return float(np.clip(normalized, 0, 1))

    except Exception:
        return 0.5  # Neutral default on error


def compute_adiabatic_coherence_index(metrics: dict, artifacts: dict) -> float:
    """
    Compute adiabatic coherence index measuring invariant spread and conservation quality.

    Higher values indicate better confinement through:
    1. Spread of J/μ ratios (prevents collective modes)
    2. Good adiabatic invariant conservation (prevents Arnold diffusion)

    Key insight: Particles with similar J/μ ratios can collectively break
    confinement through resonant interaction. We want spread in this ratio.

    Args:
        metrics: Dictionary containing mirror_ratio, field_uniformity, center_curvature
        artifacts: Dictionary containing particle tracking data

    Returns:
        Adiabatic coherence index (0-1, higher is better)
    """
    try:
        mirror_ratio = metrics.get("mirror_ratio", 1.0)
        field_uniformity = metrics.get("field_uniformity", 0.5)
        center_curvature = metrics.get("center_curvature", 0.0)
        well_depth = metrics.get("well_depth", 0.0)

        # Estimate J/μ spread from field geometry
        # J = ∮ v_∥ dl (second adiabatic invariant)
        # μ = mv_⊥²/(2B) (magnetic moment)

        # For a mirror, J/μ depends on mirror ratio and field shape
        # Higher mirror ratio = more variation in J/μ across pitch angles

        # Spread metric: how much J/μ varies across confined population
        if mirror_ratio > 1:
            # Approximate J/μ spread from mirror ratio
            # Higher R = more spread = better (prevents collective modes)
            j_mu_spread = 1 - 1.0 / mirror_ratio
        else:
            j_mu_spread = 0.0

        # Adiabatic invariant conservation quality
        # Related to field smoothness (uniformity) and curvature
        # Sharp gradients break adiabaticity

        # Field uniformity directly impacts μ conservation
        mu_conservation = field_uniformity

        # Center curvature affects J conservation
        # Small positive curvature is ideal
        optimal_curvature = 0.05
        j_conservation = np.exp(-10 * (center_curvature - optimal_curvature) ** 2)

        # Combined coherence: want high spread AND good conservation
        spread_factor = 0.3 + 0.7 * j_mu_spread  # 0.3-1.0
        conservation_factor = 0.5 * (mu_conservation + j_conservation)  # 0-1

        # Bonus for positive well (helps maintain invariants)
        well_bonus = 1.0 + 0.2 * np.tanh(5 * well_depth)

        coherence_index = spread_factor * conservation_factor * well_bonus

        return float(np.clip(coherence_index, 0, 1))

    except Exception:
        return 0.5  # Neutral default on error


def compute_drift_resonance_avoidance(metrics: dict, artifacts: dict) -> float:
    """
    Compute how well the configuration avoids dangerous drift-bounce resonances.

    Particles undergo curvature drift (toroidal) and bounce motion (axial).
    When ω_bounce/ω_drift is a low-order rational, resonant energy exchange
    can pump particles into the loss cone.

    Key insight: We want configurations where frequency ratios are irrational
    (non-resonant) for most of the confined population.

    Args:
        metrics: Dictionary containing mirror_ratio, well_depth, B_center, B_mirror
        artifacts: Dictionary containing field geometry

    Returns:
        Drift resonance avoidance score (0-1, higher is better)
    """
    try:
        mirror_ratio = metrics.get("mirror_ratio", 1.0)
        well_depth = metrics.get("well_depth", 0.0)
        B_center = metrics.get("B_center", 1.0)
        B_mirror = metrics.get("B_mirror", 1.0)
        center_curvature = metrics.get("center_curvature", 0.0)
        mean_dB_dr = metrics.get("mean_dB_dr", 0.0)

        if mirror_ratio <= 1:
            return 0.0

        # Estimate frequency ratio ω_bounce/ω_drift for different pitch angles
        # ω_bounce ∝ v_∥ / L (bounce frequency)
        # ω_drift ∝ v_⊥² * κ / (B * Ω) (curvature drift frequency)
        # where κ is field line curvature, Ω is gyrofrequency

        # For mirror: L ~ plasma_length, κ ~ 1/R_curvature
        # Ratio: ω_b/ω_d ∝ (v_∥/v_⊥²) * B / (L * κ)

        # Sample pitch angles
        n_pitch = 30
        sin2_loss = 1.0 / mirror_ratio
        pitch_min = np.sqrt(sin2_loss)
        pitches = np.linspace(pitch_min + 0.01, 0.99, n_pitch)

        # Dangerous resonances: 1:1, 1:2, 2:1, 2:3, 3:2
        dangerous_ratios = [1.0, 0.5, 2.0, 0.667, 1.5]
        resonance_width = 0.1  # How close to resonance is dangerous

        resonance_penalties = []

        for pitch in pitches:
            # Estimate frequency ratio for this pitch angle
            # v_∥/v ~ pitch, v_⊥/v ~ sqrt(1-pitch²)
            v_par_ratio = pitch
            v_perp_ratio_sq = 1 - pitch**2

            # Approximate frequency ratio (normalized units)
            # This depends on specific geometry, but we can estimate
            curvature_factor = max(abs(center_curvature), 0.01)
            field_gradient_factor = max(abs(mean_dB_dr), 0.1)

            # Frequency ratio estimate
            freq_ratio = (v_par_ratio / (v_perp_ratio_sq + 0.1)) * (
                B_center / (curvature_factor * field_gradient_factor + 0.1)
            )

            # Normalize to typical range
            freq_ratio = freq_ratio / 10.0  # Scale to ~0.1-10 range

            # Check proximity to dangerous resonances
            min_distance = min(abs(freq_ratio - r) for r in dangerous_ratios)

            # Penalty for being near resonance
            resonance_penalty = np.exp(-(min_distance**2) / (2 * resonance_width**2))
            resonance_penalties.append(resonance_penalty)

        # Weight by pitch angle distribution
        weights = np.cos(np.arcsin(pitches))
        weights /= np.sum(weights)

        # Average resonance penalty
        avg_penalty = np.sum(np.array(resonance_penalties) * weights)

        # Convert penalty to avoidance score
        avoidance_score = 1 - avg_penalty

        # Bonus for configurations with good stability (less sensitive to resonances)
        stability_bonus = 1.0 + 0.1 * np.tanh(5 * well_depth)
        avoidance_score *= stability_bonus

        return float(np.clip(avoidance_score, 0, 1))

    except Exception:
        return 0.5  # Neutral default on error


def compute_phase_space_confinement_quality(metrics: dict, artifacts: dict) -> float:
    """
    Composite metric combining bounce-averaged stability, adiabatic coherence,
    and drift resonance avoidance.

    This single number captures "how good is the phase space structure for
    confinement" beyond what mirror ratio and well depth tell us.

    Args:
        metrics: Dictionary of current metrics
        artifacts: Dictionary of evaluation artifacts

    Returns:
        Quality score 0-1 (higher is better)
    """
    try:
        # Compute individual components
        bounce_stability = compute_bounce_averaged_stability_index(metrics, artifacts)
        adiabatic_coherence = compute_adiabatic_coherence_index(metrics, artifacts)
        resonance_avoidance = compute_drift_resonance_avoidance(metrics, artifacts)

        # Weighted combination
        # All three are important, but bounce stability is most directly tied to loss
        weights = {"bounce": 0.4, "adiabatic": 0.3, "resonance": 0.3}

        quality = (
            weights["bounce"] * bounce_stability
            + weights["adiabatic"] * adiabatic_coherence
            + weights["resonance"] * resonance_avoidance
        )

        # Slight nonlinear boost for configurations that score well on all three
        # (synergy bonus)
        min_component = min(bounce_stability, adiabatic_coherence, resonance_avoidance)
        synergy_bonus = 0.1 * min_component  # Up to 10% bonus if all components good

        quality += synergy_bonus

        return float(np.clip(quality, 0, 1))

    except Exception:
        return 0.5  # Neutral default on error


def _evaluate_stage1(config: dict[str, Any]) -> dict[str, Any]:
    """
    Stage 1: Quick validation - check basic physics constraints.
    """
    try:
        coil_set = config["coil_set"]
        plasma_length = config["plasma_length"]
        plasma_radius = config["plasma_radius"]

        # Basic sanity checks
        if len(coil_set.coils) < 2:
            return {"score": 0.0, "error": "Need at least 2 coils for mirror"}

        if plasma_length <= 0 or plasma_radius <= 0:
            return {"score": 0.0, "error": "Invalid plasma dimensions"}

        # Check coil parameters are physical
        for z, r, I in coil_set.coils:
            if r <= 0:
                return {"score": 0.0, "error": f"Invalid coil radius: {r}"}
            if r < plasma_radius:
                return {"score": 0.0, "error": f"Coil radius {r} < plasma radius {plasma_radius}"}
            if abs(I) > 20e6:  # 20 MA limit
                return {"score": 0.0, "error": f"Coil current {I / 1e6:.1f} MA exceeds 20 MA limit"}

        return {"score": 1.0, "passed": True}

    except Exception as e:
        return {"score": 0.0, "error": str(e)}


def _evaluate_stage2(config: dict[str, Any]) -> dict[str, Any]:
    """
    Stage 2: Calculate core physics metrics.
    """
    from initial_program import calculate_field

    try:
        coil_set = config["coil_set"]
        plasma_length = config["plasma_length"]
        plasma_radius = config["plasma_radius"]

        # Define evaluation grid
        n_z = 200
        n_r = 50

        z_extent = plasma_length * 1.5
        z = np.linspace(-z_extent, z_extent, n_z)

        # On-axis field profile
        r_axis = np.zeros(n_z)
        Bz_axis, Br_axis = calculate_field(coil_set, z, r_axis)
        B_axis = np.sqrt(Bz_axis**2 + Br_axis**2)

        # Check for valid field
        if np.any(np.isnan(B_axis)) or np.any(np.isinf(B_axis)):
            return {"score": 0.0, "error": "Invalid magnetic field (NaN/Inf)"}

        if np.min(B_axis) <= 0:
            return {"score": 0.0, "error": "Non-positive magnetic field"}

        # === METRIC 1: Mirror Ratio ===
        # Find field in plasma region vs mirror throats
        plasma_mask = np.abs(z) < plasma_length / 2
        mirror_mask = np.abs(z) > plasma_length / 2

        if not np.any(plasma_mask) or not np.any(mirror_mask):
            return {"score": 0.0, "error": "Invalid plasma/mirror regions"}

        B_center = np.mean(B_axis[plasma_mask])
        B_mirror = np.max(B_axis[mirror_mask])
        mirror_ratio = B_mirror / B_center

        # Loss cone fraction: f_loss = 1 - sqrt(1 - 1/R)
        if mirror_ratio > 1:
            loss_cone_fraction = 1 - np.sqrt(1 - 1 / mirror_ratio)
        else:
            loss_cone_fraction = 1.0  # No confinement

        # === METRIC 2: Magnetic Well Depth ===
        # Calculate field at plasma edge vs center
        r_edge = np.full(n_z, plasma_radius)
        Bz_edge, Br_edge = calculate_field(coil_set, z, r_edge)
        B_edge = np.sqrt(Bz_edge**2 + Br_edge**2)

        B_edge_plasma = np.mean(B_edge[plasma_mask])
        well_depth = (B_edge_plasma - B_center) / B_center

        # === METRIC 3: Field Uniformity ===
        B_plasma = B_axis[plasma_mask]
        field_uniformity = 1 - np.std(B_plasma) / np.mean(B_plasma)

        # === METRIC 4: Axial Field Gradient (for stability) ===
        dBdz = np.gradient(B_axis, z)
        d2Bdz2 = np.gradient(dBdz, z)

        # At center, we want d²B/dz² > 0 for minimum-B
        center_idx = n_z // 2
        center_curvature = d2Bdz2[center_idx]

        # === Package stage 2 results ===
        metrics = {
            "mirror_ratio": float(mirror_ratio),
            "loss_cone_fraction": float(loss_cone_fraction),
            "well_depth": float(well_depth),
            "field_uniformity": float(field_uniformity),
            "B_center": float(B_center),
            "B_mirror": float(B_mirror),
            "center_curvature": float(center_curvature),
        }

        # Stage 2 score: must have mirror ratio > 1.5
        if mirror_ratio < 1.5:
            return {"score": 0.3, "metrics": metrics, "error": "Mirror ratio too low"}

        return {"score": 1.0, "metrics": metrics, "passed": True}

    except Exception as e:
        return {"score": 0.0, "error": str(e), "traceback": traceback.format_exc()}


def _evaluate_stage3(config: dict[str, Any], stage2_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Stage 3: Comprehensive evaluation with particle confinement simulation.
    """
    from initial_program import calculate_field

    try:
        coil_set = config["coil_set"]
        plasma_length = config["plasma_length"]
        plasma_radius = config["plasma_radius"]
        metrics = stage2_metrics.copy()

        # === PARTICLE TRACING FOR CONFINEMENT TIME ===
        # Simplified guiding-center approximation
        n_particles = 500
        n_steps = 2000
        dt = 1e-8  # 10 ns time step

        # Initialize particles uniformly in plasma region
        np.random.seed(42)  # Reproducibility
        z_init = np.random.uniform(-plasma_length / 4, plasma_length / 4, n_particles)
        r_init = np.random.uniform(0, plasma_radius * 0.8, n_particles)

        # Random pitch angles (v_parallel / v_total)
        # Uniform in cos(theta) for isotropic distribution
        pitch = np.random.uniform(-1, 1, n_particles)

        # Track survival
        confined = np.ones(n_particles, dtype=bool)
        confinement_times = np.zeros(n_particles)

        z_pos = z_init.copy()
        v_parallel_sign = np.sign(pitch)
        v_parallel_sign[v_parallel_sign == 0] = 1

        # Simplified dynamics: particles follow field lines
        # Bounce when B > B_mirror_local * (1 - pitch²)
        thermal_speed = 1e6  # 1000 km/s typical for fusion plasma

        for step in range(n_steps):
            # Get local field
            Bz, Br = calculate_field(coil_set, z_pos[confined], np.abs(r_init[confined]))
            B_local = np.sqrt(Bz**2 + Br**2)

            # Magnetic moment conservation: v_perp² / B = const
            # Particle reflects when B_local > B_init / cos²(pitch_init)
            B_init_vals = np.sqrt(
                calculate_field(coil_set, z_init[confined], np.abs(r_init[confined]))[0] ** 2
                + calculate_field(coil_set, z_init[confined], np.abs(r_init[confined]))[1] ** 2
            )

            # Reflection condition
            pitch_confined = pitch[confined]
            B_reflect = B_init_vals / (1 - pitch_confined**2 + 1e-10)

            # Check for reflection
            reflecting = B_local > B_reflect * 0.99
            v_parallel_sign[confined] = np.where(
                reflecting, -v_parallel_sign[confined], v_parallel_sign[confined]
            )

            # Advance position
            v_parallel = thermal_speed * np.abs(pitch_confined) * v_parallel_sign[confined]
            z_pos[confined] += v_parallel * dt

            # Check for escape (beyond mirror throats)
            escaped = np.abs(z_pos) > plasma_length * 1.2

            # Update confinement times for escaped particles
            newly_escaped = confined & escaped
            confinement_times[newly_escaped] = step * dt
            confined[escaped] = False

            if not np.any(confined):
                break

        # Particles still confined get max time
        confinement_times[confined] = n_steps * dt

        # Statistics
        mean_confinement_time = np.mean(confinement_times)
        confined_fraction = np.sum(confined) / n_particles

        metrics["mean_confinement_time"] = float(mean_confinement_time)
        metrics["confined_fraction"] = float(confined_fraction)
        metrics["n_particles_traced"] = int(n_particles)

        # === STABILITY ANALYSIS ===
        # Flute instability criterion: integral of dl/B along field line
        # For minimum-B: this should decrease away from axis

        # Simplified: check if field increases radially (good curvature)
        z_test = np.array([0.0])  # At center
        r_test = np.linspace(0, plasma_radius, 20)

        B_radial = []
        for r in r_test:
            Bz, Br = calculate_field(coil_set, z_test, np.array([r]))
            B_radial.append(np.sqrt(Bz[0] ** 2 + Br[0] ** 2))
        B_radial = np.array(B_radial)

        # Want dB/dr > 0 for stability
        dB_dr = np.gradient(B_radial, r_test)
        radial_stability = np.mean(dB_dr) > 0

        metrics["radial_stability"] = bool(radial_stability)
        metrics["mean_dB_dr"] = float(np.mean(dB_dr))

        # === NOVEL PHYSICS METRICS (from Collaborative Discovery) ===
        # These capture phase space physics beyond textbook MHD criteria
        artifacts = {}

        # --- ARTIFACT GENERATION: Full Physics Tensors ---
        # Generate 2D field map for surrogate modeling and visual forensics
        n_map_z, n_map_r = 50, 20
        map_z = np.linspace(-plasma_length, plasma_length, n_map_z)
        map_r = np.linspace(0, plasma_radius, n_map_r)
        Z, R = np.meshgrid(map_z, map_r)

        Bz_map, Br_map = calculate_field(coil_set, Z.flatten(), R.flatten())
        B_map = np.sqrt(Bz_map**2 + Br_map**2).reshape(n_map_r, n_map_z)

        artifacts["field_map"] = {
            "z_grid": map_z.tolist(),
            "r_grid": map_r.tolist(),
            "B_field": B_map.tolist(),  # List of lists for JSON serialization
            "Bz": Bz_map.reshape(n_map_r, n_map_z).tolist(),
            "Br": Br_map.reshape(n_map_r, n_map_z).tolist(),
        }

        # Capture particle phase space data
        artifacts["particle_trace"] = {
            "z_final": z_pos.tolist(),
            "confinement_times": confinement_times.tolist(),
            "initial_pitch": pitch.tolist(),
        }

        metrics["bounce_averaged_stability_index"] = compute_bounce_averaged_stability_index(
            metrics, artifacts
        )
        metrics["adiabatic_coherence_index"] = compute_adiabatic_coherence_index(metrics, artifacts)
        metrics["drift_resonance_avoidance"] = compute_drift_resonance_avoidance(metrics, artifacts)
        metrics["phase_space_confinement_quality"] = compute_phase_space_confinement_quality(
            metrics, artifacts
        )

        # === COMPUTE FINAL SCORE ===
        score = compute_final_score(metrics)

        return {"score": float(score), "metrics": metrics, "passed": True}

    except Exception as e:
        return {"score": 0.0, "error": str(e), "traceback": traceback.format_exc()}


def compute_final_score(metrics: dict[str, Any]) -> float:
    """
    Compute overall score from physics metrics.

    Scoring philosophy:
    - Mirror ratio: logarithmic benefit (diminishing returns above ~10)
    - Well depth: positive is crucial, more is better
    - Confinement: directly impacts fusion gain
    - Stability: binary but essential
    """

    score = 0.0

    # === Mirror Ratio Score (0-30 points) ===
    # R=2: 10 pts, R=5: 20 pts, R=10: 25 pts, R=20: 30 pts
    R = metrics["mirror_ratio"]
    mirror_score = 30 * (1 - np.exp(-0.15 * (R - 1)))
    mirror_score = np.clip(mirror_score, 0, 30)
    score += mirror_score

    # === Magnetic Well Score (0-25 points) ===
    # Positive well is crucial for MHD stability
    # Well depth of 0.1 (10%) is good, 0.3 (30%) is excellent
    well = metrics["well_depth"]
    if well > 0:
        well_score = 25 * (1 - np.exp(-10 * well))
    else:
        well_score = 25 * np.exp(10 * well)  # Penalty for negative well
    well_score = np.clip(well_score, 0, 25)
    score += well_score

    # === Field Uniformity Score (0-10 points) ===
    uniformity = metrics["field_uniformity"]
    uniformity_score = 10 * uniformity
    uniformity_score = np.clip(uniformity_score, 0, 10)
    score += uniformity_score

    # === Confinement Time Score (0-25 points) ===
    # Scale relative to expected loss time
    tau = metrics["mean_confinement_time"]
    tau_ref = 1e-5  # 10 microseconds reference
    conf_score = 25 * (1 - np.exp(-tau / tau_ref))
    conf_score = np.clip(conf_score, 0, 25)
    score += conf_score

    # === Stability Bonus (0-10 points) ===
    if metrics["radial_stability"]:
        score += 10
    else:
        score += 2  # Small credit for attempting

    # === Novel Physics Metrics (0-15 points total) ===
    # These metrics capture phase space physics beyond textbook MHD criteria
    # Added through Collaborative Discovery multi-agent system

    # Phase space confinement quality (composite metric, 0-10 points)
    if "phase_space_confinement_quality" in metrics:
        psq = metrics["phase_space_confinement_quality"]
        # This is already 0-1, scale to 0-10 points
        phase_space_score = 10 * psq
        score += phase_space_score

    # Bonus for excellent individual components (0-5 points)
    # Rewards configurations that excel in all three novel physics aspects
    bonus_score = 0
    if (
        "bounce_averaged_stability_index" in metrics
        and metrics["bounce_averaged_stability_index"] > 0.7
    ):
        bonus_score += 1.5
    if "adiabatic_coherence_index" in metrics and metrics["adiabatic_coherence_index"] > 0.7:
        bonus_score += 1.5
    if "drift_resonance_avoidance" in metrics and metrics["drift_resonance_avoidance"] > 0.7:
        bonus_score += 2.0
    score += bonus_score

    # Normalize to 0-1 (now max is 115 points with novel physics)
    final_score = score / 115.0

    return final_score


def evaluate(program_path: str) -> dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.

    Uses cascade evaluation:
    - Stage 1: Quick validation (threshold: 0.5)
    - Stage 2: Core physics (threshold: 0.6)
    - Stage 3: Full simulation

    Returns:
        Dictionary with 'score' (0-1) and detailed metrics
    """
    import importlib.util

    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("evolved_program", program_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["evolved_program"] = module
        spec.loader.exec_module(module)

        # Get the configuration
        if hasattr(module, "run_design"):
            config = module.run_design()
        elif hasattr(module, "design_mirror_configuration"):
            config = module.design_mirror_configuration()
        else:
            return {"score": 0.0, "combined_score": 0.0, "error": "No design function found"}

        # Stage 1: Quick validation
        stage1 = _evaluate_stage1(config)
        if stage1["score"] < 0.5:
            score = stage1["score"] * 0.1  # Max 0.1 for stage 1 failure
            return {
                "score": score,
                "combined_score": score,  # Required by OpenEvolve
                "stage": 1,
                "error": stage1.get("error", "Stage 1 failed"),
                "metrics": {},
            }

        # Stage 2: Core physics
        stage2 = _evaluate_stage2(config)
        if stage2["score"] < 0.6:
            score = 0.1 + stage2["score"] * 0.2  # 0.1-0.3 for stage 2
            return {
                "score": score,
                "combined_score": score,  # Required by OpenEvolve
                "stage": 2,
                "error": stage2.get("error", "Stage 2 failed"),
                "metrics": stage2.get("metrics", {}),
            }

        # Stage 3: Full simulation
        stage3 = _evaluate_stage3(config, stage2["metrics"])

        # Build comprehensive result
        result = {
            "score": stage3["score"],
            "combined_score": stage3["score"],  # Required by OpenEvolve
            "stage": 3,
            "metrics": stage3.get("metrics", {}),
            "description": config.get("description", "No description"),
            "num_coils": len(config["coil_set"].coils),
        }

        # Add artifacts for visualization
        result["artifacts"] = {
            "config_summary": {
                "num_coils": len(config["coil_set"].coils),
                "plasma_length": config["plasma_length"],
                "plasma_radius": config["plasma_radius"],
                "coils": [(z, r, I / 1e6) for z, r, I in config["coil_set"].coils],  # MA units
            },
            "physics_metrics": stage3.get("metrics", {}),
        }

        return result

    except Exception as e:
        return {
            "score": 0.0,
            "combined_score": 0.0,  # Required by OpenEvolve
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    # Test with the initial program
    result = evaluate("initial_program.py")

    print("=" * 60)
    print("MAGNETIC MIRROR CONFIGURATION EVALUATION")
    print("=" * 60)
    print(f"\nOverall Score: {result['score']:.3f}")
    print(f"Stage Reached: {result.get('stage', 'N/A')}")

    if "error" in result:
        print(f"Error: {result['error']}")

    if result.get("metrics"):
        print("\n--- Physics Metrics ---")
        m = result["metrics"]
        print(f"Mirror Ratio:        {m.get('mirror_ratio', 'N/A'):.2f}")
        print(f"Loss Cone Fraction:  {m.get('loss_cone_fraction', 'N/A'):.3f}")
        print(f"Magnetic Well Depth: {m.get('well_depth', 'N/A'):.3f}")
        print(f"Field Uniformity:    {m.get('field_uniformity', 'N/A'):.3f}")
        print(f"B_center:            {m.get('B_center', 'N/A'):.3f} T")
        print(f"B_mirror:            {m.get('B_mirror', 'N/A'):.3f} T")
        print(f"Mean τ_confinement:  {m.get('mean_confinement_time', 'N/A') * 1e6:.2f} μs")
        print(f"Confined Fraction:   {m.get('confined_fraction', 'N/A'):.1%}")
        print(f"Radial Stability:    {'STABLE' if m.get('radial_stability') else 'UNSTABLE'}")

        print("\n--- Novel Physics Metrics (Collaborative Discovery) ---")
        print(f"Bounce-Averaged Stability:   {m.get('bounce_averaged_stability_index', 'N/A'):.3f}")
        print(f"Adiabatic Coherence:         {m.get('adiabatic_coherence_index', 'N/A'):.3f}")
        print(f"Drift Resonance Avoidance:   {m.get('drift_resonance_avoidance', 'N/A'):.3f}")
        print(f"Phase Space Quality:         {m.get('phase_space_confinement_quality', 'N/A'):.3f}")

    print("\n" + "=" * 60)
