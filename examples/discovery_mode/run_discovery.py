#!/usr/bin/env python3
"""
Example: Running OpenEvolve in Discovery Mode

This example demonstrates how to use the Discovery Engine to enable
true scientific discovery - evolving both questions AND answers.

Key differences from standard OpenEvolve:
1. The problem itself evolves as solutions are found
2. Programs are tested adversarially (not just evaluated)
3. Surprise-based curiosity drives exploration of unknown regions

Usage:
    python run_discovery.py initial_program.py evaluator.py --config config.yaml

Or simply use the standard CLI with discovery flags:
    python openevolve-run.py initial.py evaluator.py --discovery --problem-description "Sort numbers"
"""

import argparse
import asyncio
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from openevolve.config import load_config
from openevolve.controller import OpenEvolve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """
    Run OpenEvolve in Discovery Mode.

    The simplest way to use Discovery Mode is through the standard CLI
    with the --discovery flag. This example shows programmatic setup
    for more control over the configuration.
    """
    parser = argparse.ArgumentParser(description="Run OpenEvolve in Discovery Mode")
    parser.add_argument("initial_program", help="Path to initial program file")
    parser.add_argument("evaluator", help="Path to evaluator file")
    parser.add_argument("--config", help="Path to config YAML file")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Maximum iterations (default: 100)"
    )
    parser.add_argument(
        "--problem-description",
        default="Optimize the given code for correctness and efficiency",
        help="Natural language description of the problem",
    )
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--no-skeptic", action="store_true", help="Disable adversarial skeptic")
    parser.add_argument(
        "--evolve-after", type=int, default=5, help="Evolve problem after N solutions (default: 5)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config) if args.config else load_config(None)

    # Enable Discovery Mode in config
    config.discovery.enabled = True
    config.discovery.problem_description = args.problem_description
    config.discovery.skeptic_enabled = not args.no_skeptic
    config.discovery.evolve_problem_after_solutions = args.evolve_after
    config.discovery.surprise_tracking_enabled = True
    config.discovery.curiosity_sampling_enabled = True
    config.discovery.log_discoveries = True

    # Initialize OpenEvolve with discovery-enabled config
    openevolve = OpenEvolve(
        initial_program_path=args.initial_program,
        evaluation_file=args.evaluator,
        config=config,
        output_dir=args.output_dir,
    )

    # Run evolution - discovery engine is automatically initialized and used
    logger.info("Starting Discovery Mode...")
    logger.info(f"Problem: {args.problem_description}")
    logger.info(f"Skeptic: {'enabled' if not args.no_skeptic else 'disabled'}")
    logger.info(f"Problem evolves after: {args.evolve_after} solutions")

    best_program = await openevolve.run(iterations=args.iterations)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERY COMPLETE")
    logger.info("=" * 60)

    if openevolve.discovery_engine:
        stats = openevolve.discovery_engine.get_statistics()
        logger.info(f"Final Problem Generation: {stats['current_problem']['generation']}")
        logger.info(f"Final Difficulty: {stats['current_problem']['difficulty']:.1f}")
        logger.info(f"Total Solutions Found: {stats['successful_solutions']}")
        logger.info(f"Programs Falsified: {stats['falsified_programs']}")
        logger.info(f"Problem Evolutions: {stats['problem_evolutions']}")
        logger.info(f"High Surprise Events: {stats['high_surprise_events']}")

        # Save discovery state
        discovery_state_path = os.path.join(openevolve.output_dir, "discovery_state")
        openevolve.discovery_engine.save_state(discovery_state_path)
        logger.info(f"Discovery state saved to: {discovery_state_path}")

    logger.info(f"\nBest program score: {best_program.metrics.get('combined_score', 'N/A')}")
    logger.info(f"Output directory: {openevolve.output_dir}")

    return best_program


if __name__ == "__main__":
    asyncio.run(main())
