#!/usr/bin/env python3
"""
Test script to force-trigger collaborative discovery session.

This bypasses the crisis detector to demonstrate the multi-agent debate
between Theorist, Experimentalist, Skeptic, and Synthesizer agents.
"""

import asyncio
import sys

sys.path.insert(0, "../..")

from openevolve.config import Config
from openevolve.discovery.collaborative_discovery import (
    CollaborativeDiscovery,
    CollaborativeDiscoveryConfig,
)
from openevolve.llm import LLMEnsemble


async def run_collaborative_discovery():
    """Run a collaborative discovery session with mock crisis context."""

    # Load config
    config = Config.from_yaml("config.yaml")

    print("=" * 80)
    print("COLLABORATIVE DISCOVERY TEST - Multi-Agent Novel Physics Generation")
    print("=" * 80)
    print()

    # Initialize LLM ensemble
    llm_ensemble = LLMEnsemble(config.llm.models)
    print(f"Initialized LLM ensemble: {config.llm.primary_model}")

    # Create collaborative discovery config
    collab_config = CollaborativeDiscoveryConfig(
        max_debate_rounds=config.discovery.heisenberg.max_debate_rounds,
        min_consensus_for_synthesis=config.discovery.heisenberg.min_consensus_for_synthesis,
        elimination_threshold=config.discovery.heisenberg.elimination_threshold,
    )

    # Initialize collaborative discovery
    collaborative_discovery = CollaborativeDiscovery(
        config=collab_config,
        llm_ensemble=llm_ensemble,
        domain_context=config.discovery.heisenberg.domain_context,
    )

    print(
        f"Initialized CollaborativeDiscovery with {collab_config.max_debate_rounds} debate rounds"
    )
    print()

    # Create mock crisis context (simulating a plateau at 0.9626)
    crisis_context = {
        "best_fitness": 0.9626,
        "fitness_history": [0.9621, 0.9621, 0.9621, 0.9622, 0.9623, 0.9623, 0.9624, 0.9626],
        "current_metrics": [
            "mirror_ratio",
            "loss_cone_fraction",
            "well_depth",
            "field_uniformity",
            "B_center",
            "B_mirror",
            "center_curvature",
            "confined_fraction",
            "radial_stability",
        ],
        "plateau_iterations": 8,
        "variance": 0.00002,
        "crisis_type": "plateau",
        "crisis_description": (
            "Optimization has plateaued at 0.9626 fitness with mirror ratio ~100, "
            "well depth ~1.08, and 99.8% particle confinement. Traditional evolution "
            "is unable to find further improvements. The system needs new physics "
            "variables or fundamentally different approaches to break through this barrier."
        ),
    }

    print("CRISIS CONTEXT:")
    print(f"  Best fitness: {crisis_context['best_fitness']}")
    print(f"  Plateau iterations: {crisis_context['plateau_iterations']}")
    print(f"  Crisis: {crisis_context['crisis_description'][:100]}...")
    print()
    print("=" * 80)
    print("STARTING MULTI-AGENT DISCOVERY SESSION")
    print("=" * 80)
    print()

    # Run the collaborative discovery session
    try:
        syntheses = await collaborative_discovery.run_discovery_session(crisis_context)

        print()
        print("=" * 80)
        print("DISCOVERY SESSION COMPLETE")
        print("=" * 80)
        print()

        if syntheses:
            print(f"Generated {len(syntheses)} synthesis(es):")
            print()
            for i, synthesis in enumerate(syntheses, 1):
                print(f"--- Synthesis {i} ---")
                print(f"Variable Name: {synthesis.variable_name}")
                print(f"Description: {synthesis.description}")
                print(f"Consensus Score: {synthesis.consensus_score:.2f}")
                print()
                print("Computation Code:")
                print("-" * 40)
                print(synthesis.computation_code[:500] if synthesis.computation_code else "None")
                if synthesis.computation_code and len(synthesis.computation_code) > 500:
                    print("... (truncated)")
                print("-" * 40)
                print()
                if synthesis.validation_test:
                    print("Validation Test:")
                    print(synthesis.validation_test[:300] if synthesis.validation_test else "None")
                    print()
                print()
        else:
            print("No syntheses generated (agents may not have reached consensus)")

    except Exception as e:
        print(f"Error during collaborative discovery: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(run_collaborative_discovery())
