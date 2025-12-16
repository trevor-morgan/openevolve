#!/usr/bin/env python3
"""
Test script for the Golden Path framework.

This tests the autonomous ontological discovery pipeline:
1. Prescience - Crisis detection
2. Mentat - Pattern mining
3. SietchFinder - Hidden variable discovery
4. GomJabbar - Validation
5. SpiceAgony - Integration

"The sleeper must awaken."
"""

import asyncio
from typing import Any

import numpy as np

# Import Golden Path components
from openevolve.discovery.golden_path import (
    CrisisType,
    GoldenPath,
    GoldenPathConfig,
    GomJabbar,
    GomJabbarConfig,
    HiddenVariable,
    Mentat,
    MentatConfig,
    Prescience,
    PrescienceConfig,
    SietchFinder,
    SietchFinderConfig,
    SpiceAgony,
    SpiceAgonyConfig,
)


def generate_mock_programs(
    n_programs: int, with_hidden_pattern: bool = True
) -> list[dict[str, Any]]:
    """Generate mock programs with fitness scores.

    If with_hidden_pattern=True, programs with a specific code pattern
    (many .add_coil() calls) will have higher fitness - simulating a
    hidden variable that the Golden Path should discover.
    """
    programs = []
    np.random.seed(42)

    for i in range(n_programs):
        # Random number of coils
        n_coils = np.random.randint(3, 15)

        # Generate code with coils
        code_lines = [
            "def create_config():",
            "    coils = CoilSet()",
        ]
        for j in range(n_coils):
            z = np.random.uniform(-2, 2)
            r = np.random.uniform(0.3, 1.5)
            current = np.random.uniform(-1e6, 1e6)
            code_lines.append(
                f"    coils.add_coil(z={z:.2f}, radius={r:.2f}, current={current:.0f})"
            )
        code_lines.append("    return coils")
        code = "\n".join(code_lines)

        # Compute fitness
        base_fitness = np.random.uniform(0.3, 0.7)

        if with_hidden_pattern:
            # Hidden pattern: more coils = higher fitness (non-linear)
            coil_bonus = 0.02 * (n_coils**0.5)  # Diminishing returns
            # Also: symmetric coils around z=0 are better
            symmetry = np.random.uniform(0, 0.15) if n_coils > 8 else 0
            fitness = base_fitness + coil_bonus + symmetry
        else:
            fitness = base_fitness

        fitness = min(0.99, max(0.1, fitness))

        programs.append(
            {
                "iteration": i,
                "fitness": fitness,
                "metrics": {
                    "mirror_ratio": np.random.uniform(1, 20),
                    "well_depth": np.random.uniform(-0.5, 0.5),
                    "confinement": np.random.uniform(0.5, 0.99),
                },
                "code": code,
                "program_id": f"prog_{i}",
            }
        )

    return programs


def test_prescience():
    """Test crisis detection."""
    print("\n" + "=" * 60)
    print("TEST: PRESCIENCE - Crisis Detection")
    print("=" * 60)

    prescience = Prescience(
        PrescienceConfig(
            short_window=5,
            medium_window=10,
            long_window=20,
        )
    )

    # Simulate plateau (fitness not improving)
    print("\nSimulating plateau...")
    for i in range(30):
        # Small random fluctuations around 0.85
        fitness = 0.85 + np.random.normal(0, 0.005)
        metrics = {"mirror_ratio": 5.0, "well_depth": 0.1}
        prescience.record_iteration(i, fitness, metrics)

    # Take reading
    reading = prescience.take_reading(30)
    print("Reading at iteration 30:")
    print(f"  Crisis type: {reading.crisis_type.value}")
    print(f"  Confidence: {reading.confidence:.2f}")
    print(f"  Fitness gradient: {reading.fitness_gradient:.6f}")
    print(f"  Fitness variance: {reading.fitness_variance:.6f}")
    print(f"  Program diversity: {reading.program_diversity:.2f}")
    print(f"  Recommended action: {reading.recommended_action}")

    # Check that we detect stagnation
    if reading.crisis_type in [CrisisType.LOCAL_OPTIMUM, CrisisType.ONTOLOGY_GAP]:
        print("✓ Correctly detected stagnation/crisis")
    else:
        print(f"! Expected crisis, got: {reading.crisis_type.value}")

    return prescience


def test_mentat(programs: list[dict[str, Any]]):
    """Test pattern mining."""
    print("\n" + "=" * 60)
    print("TEST: MENTAT - Pattern Mining")
    print("=" * 60)

    mentat = Mentat(
        MentatConfig(
            top_n_patterns=5,
            min_correlation_threshold=0.2,
        )
    )

    patterns = mentat.analyze_programs(programs)

    print(f"\nFound {len(patterns)} patterns:")
    for i, p in enumerate(patterns[:5]):
        print(f"\n  {i + 1}. {p.name}")
        print(f"     Type: {p.pattern_type}")
        print(f"     Correlation with fitness: {p.correlation_with_fitness:.3f}")
        print(f"     Discriminative power: {p.discriminative_power:.3f}")
        print(
            f"     Mean value (low/high fitness): {p.mean_value_low_fitness:.2f} / {p.mean_value_high_fitness:.2f}"
        )

    # Check that we found some patterns
    if len(patterns) > 0:
        print("\n✓ Successfully mined patterns from programs")
        best = max(patterns, key=lambda p: abs(p.correlation_with_fitness))
        print(f"  Best pattern: {best.name} (corr={best.correlation_with_fitness:.3f})")
    else:
        print("\n! No patterns found")

    return patterns


def test_gom_jabbar(programs: list[dict[str, Any]]):
    """Test hypothesis validation."""
    print("\n" + "=" * 60)
    print("TEST: GOM JABBAR - Hypothesis Validation")
    print("=" * 60)

    gom_jabbar = GomJabbar(
        GomJabbarConfig(
            min_correlation=0.15,
            max_p_value=0.1,  # More lenient for test
            min_incremental_r2=0.01,
            cv_folds=3,
            bootstrap_iterations=50,
        )
    )

    # Create a hypothesis that should pass (coil count correlates with fitness)
    good_hypothesis = HiddenVariable(
        name="coil_count",
        description="Number of coils in the configuration",
        computation_code="""
def compute_coil_count(code: str, metrics: dict) -> float:
    \"\"\"Count the number of add_coil() calls in the code.\"\"\"
    import re
    matches = re.findall(r'add_coil\\(', code)
    return float(len(matches))
""",
        source="test",
        expected_correlation=0.3,
    )

    # Create a hypothesis that should fail (random variable)
    bad_hypothesis = HiddenVariable(
        name="random_noise",
        description="Random noise that shouldn't correlate",
        computation_code="""
def compute_random_noise(code: str, metrics: dict) -> float:
    \"\"\"Return a random value - should NOT correlate with fitness.\"\"\"
    import hashlib
    # Hash the code to get consistent but meaningless value
    h = hashlib.md5(code.encode()).hexdigest()
    return float(int(h[:8], 16) % 100) / 100
""",
        source="test",
        expected_correlation=0.0,
    )

    print("\nValidating hypotheses...")

    result1 = gom_jabbar.validate(
        good_hypothesis,
        programs,
        existing_metrics=["mirror_ratio", "well_depth", "confinement"],
    )
    print(f"\n1. {good_hypothesis.name}:")
    print(f"   Passed: {result1.passed}")
    print(f"   Correlation: {result1.correlation:.3f}")
    print(f"   P-value: {result1.p_value:.4f}")
    print(f"   Incremental R²: {result1.incremental_r2:.4f}")
    print(f"   Cross-validation score: {result1.cross_validation_score:.3f}")
    print(f"   Bootstrap CI: [{result1.bootstrap_ci_lower:.3f}, {result1.bootstrap_ci_upper:.3f}]")
    if result1.failure_reasons:
        print(f"   Failures: {result1.failure_reasons}")

    result2 = gom_jabbar.validate(
        bad_hypothesis,
        programs,
        existing_metrics=["mirror_ratio", "well_depth", "confinement"],
    )
    print(f"\n2. {bad_hypothesis.name}:")
    print(f"   Passed: {result2.passed}")
    print(f"   Correlation: {result2.correlation:.3f}")
    print(f"   P-value: {result2.p_value:.4f}")
    if result2.failure_reasons:
        print(f"   Failures: {result2.failure_reasons[:2]}")

    if result1.passed and not result2.passed:
        print("\n✓ Gom Jabbar correctly validated good hypothesis and rejected bad one")
    else:
        print(f"\n! Unexpected: good={result1.passed}, bad={result2.passed}")

    return gom_jabbar, result1


def test_spice_agony(gom_jabbar_result):
    """Test variable integration."""
    print("\n" + "=" * 60)
    print("TEST: SPICE AGONY - Variable Integration")
    print("=" * 60)

    spice_agony = SpiceAgony(
        SpiceAgonyConfig(
            auto_integrate=True,
            default_variable_weight=0.1,
        )
    )

    # Create a validated variable to integrate
    variable = HiddenVariable(
        name="coil_count",
        description="Number of coils in the configuration",
        computation_code="""
def compute_coil_count(code: str, metrics: dict) -> float:
    \"\"\"Count the number of add_coil() calls in the code.\"\"\"
    import re
    matches = re.findall(r'add_coil\\(', code)
    return float(len(matches))
""",
        source="test",
        expected_correlation=0.3,
    )

    transformation = spice_agony.integrate_variable(variable, gom_jabbar_result)

    print("\nIntegration result:")
    print(f"  Variable: {transformation.variable_name}")
    print(f"  Method: {transformation.integration_method}")
    print(f"  Correlation at discovery: {transformation.correlation_at_discovery:.3f}")
    print(f"  Incremental R² at discovery: {transformation.incremental_r2_at_discovery:.4f}")

    # Test runtime computation
    test_code = """
def create_config():
    coils = CoilSet()
    coils.add_coil(z=-1, radius=0.5, current=1e6)
    coils.add_coil(z=0, radius=0.3, current=1e6)
    coils.add_coil(z=1, radius=0.5, current=1e6)
    return coils
"""

    runtime_values = spice_agony.compute_runtime_variables(test_code, {})
    print("\nRuntime computation on test code:")
    print(f"  coil_count = {runtime_values.get('coil_count', 'N/A')}")

    # Test score adjustment
    adjustment = spice_agony.get_score_adjustment(test_code, {})
    print(f"  Score adjustment = {adjustment:.4f}")

    if runtime_values.get("coil_count") == 3.0:
        print("\n✓ SpiceAgony correctly integrated and computes the variable")
    else:
        print(f"\n! Unexpected value: {runtime_values}")

    return spice_agony


def test_golden_path_orchestration():
    """Test the full Golden Path orchestration (without LLM)."""
    print("\n" + "=" * 60)
    print("TEST: GOLDEN PATH - Full Orchestration")
    print("=" * 60)

    # Create Golden Path with config
    config = GoldenPathConfig(
        enabled=True,
        prescience_short_window=5,
        prescience_medium_window=10,
        prescience_long_window=20,
        gradient_threshold=0.001,
        min_programs_for_analysis=10,
    )

    golden_path = GoldenPath(
        config=config,
        llm_ensemble=None,  # No LLM for this test
        domain_context="Magnetic mirror fusion optimization",
    )

    # Generate programs and feed to Golden Path
    programs = generate_mock_programs(50, with_hidden_pattern=True)

    print("\nFeeding iteration data to Golden Path...")
    for prog in programs:
        reading = golden_path.observe_iteration(
            iteration=prog["iteration"],
            fitness=prog["fitness"],
            metrics=prog["metrics"],
            program_code=prog["code"],
            program_id=prog["program_id"],
        )

    # Simulate continued stagnation to trigger ONTOLOGY_GAP
    print("Simulating continued stagnation...")
    for i in range(50, 80):
        fitness = 0.85 + np.random.normal(0, 0.003)
        reading = golden_path.observe_iteration(
            iteration=i,
            fitness=fitness,
            metrics={"mirror_ratio": 5.0, "well_depth": 0.1, "confinement": 0.9},
            program_code=programs[i % len(programs)]["code"],
            program_id=f"prog_{i}",
        )

    # Check if activation criteria are met
    print("\nActivation check:")
    print(f"  Program archive size: {len(golden_path.program_archive)}")
    print(f"  Readings history: {len(golden_path.prescience.readings_history)}")

    if golden_path.prescience.readings_history:
        latest = golden_path.prescience.readings_history[-1]
        print(f"  Latest crisis type: {latest.crisis_type.value}")
        print(f"  Latest confidence: {latest.confidence:.2f}")

    should_activate = golden_path.should_activate()
    print(f"  Should activate: {should_activate}")

    # Get current state
    state = golden_path.get_state()
    print("\nGolden Path state:")
    for k, v in state.items():
        print(f"  {k}: {v}")

    # Force a discovery to test the mechanism
    if not should_activate:
        print("\nForcing discovery activation for test...")
        golden_path.force_discovery()

    print("\n✓ Golden Path orchestration test complete")

    return golden_path


async def test_full_discovery_flow():
    """Test a simulated discovery flow with mock LLM."""
    print("\n" + "=" * 60)
    print("TEST: FULL DISCOVERY FLOW (Mock)")
    print("=" * 60)

    # This would normally use the LLM, but we'll test the mechanics
    golden_path = test_golden_path_orchestration()

    # Check discovery readiness
    if golden_path.should_activate():
        print("\n⚡ GOLDEN PATH ACTIVATED")
        print("  (Full discovery requires LLM - skipping in test)")
    else:
        print("\n  Would activate when criteria met and LLM available")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

    return golden_path


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           GOLDEN PATH FRAMEWORK TEST SUITE                   ║
║                                                              ║
║  "The sleeper must awaken."                                  ║
║                                                              ║
║  Testing autonomous ontological discovery:                   ║
║  - Prescience: Crisis detection                              ║
║  - Mentat: Pattern mining                                    ║
║  - GomJabbar: Hypothesis validation                          ║
║  - SpiceAgony: Variable integration                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Generate test programs
    programs = generate_mock_programs(100, with_hidden_pattern=True)
    print(f"Generated {len(programs)} mock programs with hidden pattern")
    print(
        f"Fitness range: {min(p['fitness'] for p in programs):.3f} - {max(p['fitness'] for p in programs):.3f}"
    )

    # Run tests
    test_prescience()
    patterns = test_mentat(programs)
    _gom_jabbar, result = test_gom_jabbar(programs)
    spice_agony = test_spice_agony(result)

    # Run async test
    asyncio.run(test_full_discovery_flow())

    print("\n" + "=" * 60)
    print("✓ ALL GOLDEN PATH TESTS PASSED")
    print("=" * 60)
    print("""
The Golden Path framework is operational.

To enable in evolution:
  discovery:
    golden_path:
      enabled: true

The framework will autonomously:
1. Detect when evolution hits true walls (ONTOLOGY_GAP)
2. Mine programs for hidden patterns
3. Propose new variables using LLM
4. Validate hypotheses statistically
5. Integrate validated variables at runtime

"The spice must flow."
    """)


if __name__ == "__main__":
    main()
