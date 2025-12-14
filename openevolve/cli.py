"""
Command-line interface for OpenEvolve
"""

import argparse
import asyncio
import logging
import os
import sys

from openevolve import OpenEvolve
from openevolve.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument(
        "initial_program",
        nargs="?",
        help="Path to the initial program file (omit when using --init-from-prompt)",
    )

    parser.add_argument(
        "evaluation_file",
        nargs="?",
        help="Path to the evaluation file containing an 'evaluate' function (omit when using --init-from-prompt)",
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    # Discovery Mode arguments
    parser.add_argument(
        "--discovery",
        action="store_true",
        help="Enable Discovery Mode for scientific discovery",
    )

    parser.add_argument(
        "--problem-description",
        help="Genesis problem description for Discovery Mode",
        default=None,
    )

    parser.add_argument(
        "--evolve-after",
        type=int,
        help="Evolve problem after N solutions (default: 5)",
        default=None,
    )

    parser.add_argument(
        "--no-skeptic",
        action="store_true",
        help="Disable adversarial skeptic testing",
    )

    parser.add_argument(
        "--surprise-threshold",
        type=float,
        help="Surprise score threshold for curiosity sampling (default: 0.2)",
        default=None,
    )

    parser.add_argument(
        "--hot-reload",
        action="store_true",
        help="Hot-reload a small set of config knobs from --config while running (timeouts).",
    )

    parser.add_argument(
        "--hot-reload-interval",
        type=float,
        default=2.0,
        help="Config hot-reload polling interval in seconds (default: 2.0).",
    )

    parser.add_argument(
        "--init-from-prompt",
        help="Create a new project skeleton from a short prompt (writes initial_program.py, evaluator.py, config.yaml).",
        default=None,
    )

    parser.add_argument(
        "--init-dir",
        help="Directory to create the project in (default: ./<slug-from-prompt>).",
        default=".",
    )

    parser.add_argument(
        "--init-name",
        help="Project name (used for directory slug; defaults to first line of prompt).",
        default=None,
    )

    parser.add_argument(
        "--init-entrypoint",
        help="Entrypoint function name in the generated program (default: solve).",
        default="solve",
    )

    parser.add_argument(
        "--init-test-cases",
        help="Path to a JSON file of test cases (list of {input, output}). If omitted, a minimal template is created.",
        default=None,
    )

    parser.add_argument(
        "--init-force",
        action="store_true",
        help="Overwrite existing files in the init directory.",
    )

    return parser.parse_args()


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    if args.init_from_prompt:
        from pathlib import Path

        from openevolve.init_project import init_project

        test_cases = None
        if args.init_test_cases:
            import json

            test_cases_path = Path(args.init_test_cases).expanduser()
            test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))

        result = init_project(
            prompt=args.init_from_prompt,
            init_dir=args.init_dir,
            project_name=args.init_name,
            entrypoint=args.init_entrypoint,
            test_cases=test_cases,
            force=bool(args.init_force),
        )

        print("Initialized OpenEvolve project:")
        print(f"  Directory: {result.project_dir}")
        print(f"  Program:   {result.initial_program_path}")
        print(f"  Evaluator: {result.evaluator_path}")
        print(f"  Config:    {result.config_path}")
        print(f"  Tests:     {result.test_cases_path}")
        print("")
        print("Next:")
        print(f"  Edit {result.test_cases_path} to add grounded test cases")
        print(
            f"  Run: ./.venv/bin/openevolve-run {result.initial_program_path} {result.evaluator_path} --config {result.config_path} --iterations 50"
        )
        return 0

    # Check if files exist
    if not args.initial_program or not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not args.evaluation_file or not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    # Load base config from file or defaults
    config = load_config(args.config)

    # Create config object with command-line overrides
    if args.api_base or args.primary_model or args.secondary_model:
        # Apply command-line overrides
        if args.api_base:
            config.llm.api_base = args.api_base
            print(f"Using API base: {config.llm.api_base}")

        if args.primary_model:
            config.llm.primary_model = args.primary_model
            print(f"Using primary model: {config.llm.primary_model}")

        if args.secondary_model:
            config.llm.secondary_model = args.secondary_model
            print(f"Using secondary model: {config.llm.secondary_model}")

        # Rebuild models list to apply CLI overrides
        if args.primary_model or args.secondary_model:
            config.llm.rebuild_models()
            print("Applied CLI model overrides - active models:")
            for i, model in enumerate(config.llm.models):
                print(f"  Model {i + 1}: {model.name} (weight: {model.weight})")

    # Apply discovery mode CLI overrides
    if args.discovery:
        config.discovery.enabled = True
        print("Discovery Mode enabled")

    if args.problem_description:
        config.discovery.problem_description = args.problem_description
        config.discovery.enabled = True
        print(f"Genesis problem: {args.problem_description}")

    if args.evolve_after is not None:
        config.discovery.evolve_problem_after_solutions = args.evolve_after
        print(f"Problem evolves after {args.evolve_after} solutions")

    if args.no_skeptic:
        config.discovery.skeptic_enabled = False
        print("Adversarial skeptic disabled")

    if args.surprise_threshold is not None:
        config.discovery.surprise_bonus_threshold = args.surprise_threshold
        print(f"Surprise threshold: {args.surprise_threshold}")

    # Initialize OpenEvolve
    try:
        openevolve = OpenEvolve(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            output_dir=args.output,
            config_path=args.config,
            hot_reload=bool(args.hot_reload),
            hot_reload_interval=float(args.hot_reload_interval),
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            openevolve.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {openevolve.database.last_iteration})"
            )

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print("\nEvolution complete!")
        print("Best program metrics:")
        for name, value in best_program.metrics.items():
            # Handle mixed types: format numbers as floats, others as strings
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {e!s}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
