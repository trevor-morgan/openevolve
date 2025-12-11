"""
Discovery Engine - Main orchestrator for Open-Ended Scientific Discovery

This is the main integration point that combines:
1. ProblemEvolver - Evolves the questions being asked
2. AdversarialSkeptic - Falsification-based evaluation
3. EpistemicArchive - Behavioral diversity storage

The Discovery Engine runs on TOP of OpenEvolve's existing evolution loop,
adding the three key capabilities needed for true scientific discovery.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openevolve.controller import OpenEvolve
    from openevolve.database import Program

from openevolve.discovery.problem_space import (
    ProblemSpace,
    ProblemEvolver,
    ProblemEvolverConfig,
)
from openevolve.discovery.skeptic import (
    AdversarialSkeptic,
    SkepticConfig,
    FalsificationResult,
)
from openevolve.discovery.epistemic_archive import (
    EpistemicArchive,
    Phenotype,
    SurpriseMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Configuration for the Discovery Engine"""

    # Problem evolution
    evolve_problem_after_solutions: int = 5  # Evolve problem after N successful solutions
    problem_evolution_enabled: bool = True

    # Adversarial skepticism
    skeptic_enabled: bool = True
    skeptic_config: SkepticConfig = field(default_factory=SkepticConfig)

    # Epistemic archive
    surprise_tracking_enabled: bool = True
    curiosity_sampling_enabled: bool = True
    phenotype_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "efficiency"]
    )

    # Thresholds
    solution_threshold: float = 0.8  # Fitness threshold to consider problem "solved"
    surprise_bonus_threshold: float = 0.2  # Surprise level to trigger bonus exploration

    # Logging
    log_discoveries: bool = True
    discovery_log_path: Optional[str] = None


@dataclass
class DiscoveryEvent:
    """Record of a discovery event"""
    timestamp: float
    event_type: str  # "solution", "problem_evolution", "surprise", "falsification"
    problem_id: str
    program_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class DiscoveryEngine:
    """
    Main engine for open-ended scientific discovery.

    This sits on top of OpenEvolve and adds:
    1. Problem space evolution (Explorer)
    2. Adversarial falsification (Skeptic)
    3. Surprise-based curiosity (Archive enhancement)

    Usage:
        engine = DiscoveryEngine(config, openevolve_controller)
        engine.set_genesis_problem(description, evaluator_path)

        # Run discovery loop
        async for discovery in engine.run(max_iterations=1000):
            print(f"Discovery: {discovery}")
    """

    def __init__(
        self,
        config: DiscoveryConfig,
        openevolve: "OpenEvolve",
    ):
        self.config = config
        self.openevolve = openevolve

        # Initialize components
        self.problem_evolver = ProblemEvolver(
            config=ProblemEvolverConfig(),
            llm_ensemble=openevolve.llm_ensemble,
        )

        self.skeptic = AdversarialSkeptic(
            config=config.skeptic_config,
            llm_ensemble=openevolve.llm_ensemble,
        ) if config.skeptic_enabled else None

        self.archive = EpistemicArchive(
            database=openevolve.database,
            phenotype_dimensions=config.phenotype_dimensions,
        )

        # State tracking
        self.current_problem: Optional[ProblemSpace] = None
        self.solutions_since_evolution: int = 0
        self.discovery_events: List[DiscoveryEvent] = []

        # Statistics
        self.stats = {
            "total_iterations": 0,
            "successful_solutions": 0,
            "falsified_programs": 0,
            "problem_evolutions": 0,
            "high_surprise_events": 0,
        }

        logger.info("Initialized DiscoveryEngine")

    def set_genesis_problem(
        self,
        description: str,
        evaluator_path: Optional[str] = None,
    ) -> ProblemSpace:
        """
        Initialize with a genesis problem.

        Args:
            description: Natural language description of the problem
            evaluator_path: Path to the evaluator file (optional)

        Returns:
            The genesis ProblemSpace
        """
        if evaluator_path:
            problem = self.problem_evolver.create_genesis_from_evaluator(
                evaluator_path, description
            )
        else:
            problem = ProblemSpace(
                id="genesis",
                description=description,
            )
            self.problem_evolver.set_genesis_problem(problem)

        self.current_problem = problem

        self._log_event(DiscoveryEvent(
            timestamp=time.time(),
            event_type="genesis",
            problem_id=problem.id,
            details={"description": description},
        ))

        logger.info(f"Set genesis problem: {description[:100]}...")
        return problem

    async def process_program(
        self,
        program: "Program",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a program through the discovery pipeline.

        This is the main integration point with OpenEvolve.
        Call this after OpenEvolve evaluates a program.

        Args:
            program: The program to process

        Returns:
            Tuple of (is_valid_solution, metadata_dict)
        """
        metadata = {
            "problem_id": self.current_problem.id if self.current_problem else None,
            "falsification_passed": True,
            "surprise_score": 0.0,
            "is_solution": False,
        }

        # Step 1: Predict fitness for surprise calculation
        predicted_fitness = self.archive.predict_fitness(program.code)

        # Step 2: Adversarial falsification
        if self.skeptic and self.config.skeptic_enabled:
            description = self.current_problem.description if self.current_problem else ""

            survived, results = await self.skeptic.falsify(
                program,
                description=description,
                language=program.language or "python",
            )

            metadata["falsification_passed"] = survived
            metadata["falsification_results"] = [r.to_dict() for r in results]

            if not survived:
                self.stats["falsified_programs"] += 1
                self._log_event(DiscoveryEvent(
                    timestamp=time.time(),
                    event_type="falsification",
                    problem_id=self.current_problem.id if self.current_problem else "",
                    program_id=program.id,
                    details={
                        "attack_type": results[-1].attack_type if results else "unknown",
                        "error": results[-1].error_message if results else None,
                    },
                ))

                logger.info(f"Program {program.id} FALSIFIED - not a valid solution")
                return False, metadata

        # Step 3: Add to archive with phenotype and surprise tracking
        was_novel, surprise = self.archive.add_with_phenotype(
            program,
            predicted_fitness=predicted_fitness if self.config.surprise_tracking_enabled else None,
        )

        if surprise:
            metadata["surprise_score"] = surprise.surprise_score
            metadata["is_positive_surprise"] = surprise.is_positive_surprise

            if surprise.surprise_score > self.config.surprise_bonus_threshold:
                self.stats["high_surprise_events"] += 1
                self._log_event(DiscoveryEvent(
                    timestamp=time.time(),
                    event_type="surprise",
                    problem_id=self.current_problem.id if self.current_problem else "",
                    program_id=program.id,
                    details={
                        "predicted": surprise.predicted_fitness,
                        "actual": surprise.actual_fitness,
                        "surprise": surprise.surprise_score,
                        "positive": surprise.is_positive_surprise,
                    },
                ))

        # Step 4: Check if this is a solution
        fitness = program.metrics.get("combined_score", 0.0)
        is_solution = fitness >= self.config.solution_threshold

        if is_solution:
            metadata["is_solution"] = True
            self.stats["successful_solutions"] += 1
            self.solutions_since_evolution += 1

            if self.current_problem:
                self.current_problem.solved_by.append(program.id)

            self._log_event(DiscoveryEvent(
                timestamp=time.time(),
                event_type="solution",
                problem_id=self.current_problem.id if self.current_problem else "",
                program_id=program.id,
                details={
                    "fitness": fitness,
                    "phenotype": program.metadata.get("phenotype", {}),
                },
            ))

            logger.info(
                f"SOLUTION FOUND: Program {program.id} solves problem "
                f"{self.current_problem.id if self.current_problem else 'unknown'} "
                f"with fitness {fitness:.3f}"
            )

            # Step 5: Check if we should evolve the problem
            if (
                self.config.problem_evolution_enabled and
                self.solutions_since_evolution >= self.config.evolve_problem_after_solutions
            ):
                await self._evolve_problem()

        self.stats["total_iterations"] += 1
        return is_solution, metadata

    async def _evolve_problem(self) -> None:
        """Evolve the current problem to a more challenging variant"""
        if not self.current_problem:
            return

        # Get characteristics of successful solutions
        solution_characteristics = self._get_solution_characteristics()

        # Evolve the problem
        new_problem = await self.problem_evolver.evolve(
            self.current_problem,
            solution_characteristics,
        )

        self._log_event(DiscoveryEvent(
            timestamp=time.time(),
            event_type="problem_evolution",
            problem_id=new_problem.id,
            details={
                "parent_problem_id": self.current_problem.id,
                "new_constraints": new_problem.constraints,
                "new_difficulty": new_problem.difficulty_level,
                "solution_characteristics": solution_characteristics,
            },
        ))

        self.current_problem = new_problem
        self.solutions_since_evolution = 0
        self.stats["problem_evolutions"] += 1

        logger.info(
            f"PROBLEM EVOLVED: {new_problem.id} "
            f"(generation {new_problem.generation}, difficulty {new_problem.difficulty_level:.1f})"
        )

    def _get_solution_characteristics(self) -> Dict[str, Any]:
        """Get aggregate characteristics of successful solutions"""
        solutions = []

        if self.current_problem:
            for program_id in self.current_problem.solved_by:
                program = self.openevolve.database.programs.get(program_id)
                if program:
                    solutions.append(program)

        if not solutions:
            return {}

        # Aggregate characteristics
        characteristics = {
            "num_solutions": len(solutions),
            "avg_fitness": sum(
                p.metrics.get("combined_score", 0) for p in solutions
            ) / len(solutions),
            "avg_complexity": sum(
                p.metadata.get("phenotype", {}).get("complexity", 0) for p in solutions
            ) / len(solutions),
            "approaches": list(set(
                p.metadata.get("phenotype", {}).get("approach_signature", "")
                for p in solutions
            )),
        }

        return characteristics

    def get_curiosity_samples(self, n: int = 3) -> List["Program"]:
        """
        Get programs that maximize expected information gain.

        Use these as inspirations for the next evolution step.
        """
        if self.config.curiosity_sampling_enabled:
            return self.archive.sample_for_curiosity(n)
        return []

    def get_current_problem_context(self) -> str:
        """Get the current problem formatted for prompt inclusion"""
        if self.current_problem:
            return self.current_problem.to_prompt_context()
        return ""

    def _log_event(self, event: DiscoveryEvent) -> None:
        """Log a discovery event"""
        self.discovery_events.append(event)

        if self.config.log_discoveries and self.config.discovery_log_path:
            try:
                os.makedirs(os.path.dirname(self.config.discovery_log_path), exist_ok=True)
                with open(self.config.discovery_log_path, 'a') as f:
                    f.write(json.dumps({
                        "timestamp": event.timestamp,
                        "type": event.event_type,
                        "problem_id": event.problem_id,
                        "program_id": event.program_id,
                        "details": event.details,
                    }) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log discovery event: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        return {
            **self.stats,
            "current_problem": {
                "id": self.current_problem.id if self.current_problem else None,
                "generation": self.current_problem.generation if self.current_problem else 0,
                "difficulty": self.current_problem.difficulty_level if self.current_problem else 0,
                "solutions": len(self.current_problem.solved_by) if self.current_problem else 0,
            },
            "skeptic_stats": self.skeptic.get_attack_statistics() if self.skeptic else {},
            "surprise_stats": self.archive.get_surprise_statistics(),
            "approach_diversity": self.archive.get_approach_diversity(),
        }

    def save_state(self, path: str) -> None:
        """Save discovery state to disk"""
        os.makedirs(path, exist_ok=True)

        # Save problem history
        self.problem_evolver.save(os.path.join(path, "problems.json"))

        # Save events
        with open(os.path.join(path, "events.json"), 'w') as f:
            json.dump([
                {
                    "timestamp": e.timestamp,
                    "type": e.event_type,
                    "problem_id": e.problem_id,
                    "program_id": e.program_id,
                    "details": e.details,
                }
                for e in self.discovery_events
            ], f, indent=2)

        # Save statistics
        with open(os.path.join(path, "stats.json"), 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)

        logger.info(f"Saved discovery state to {path}")

    def load_state(self, path: str) -> None:
        """Load discovery state from disk"""
        # Load problem history
        problems_path = os.path.join(path, "problems.json")
        if os.path.exists(problems_path):
            self.problem_evolver.load(problems_path)
            self.current_problem = self.problem_evolver.current_problem

        # Load events
        events_path = os.path.join(path, "events.json")
        if os.path.exists(events_path):
            with open(events_path, 'r') as f:
                events_data = json.load(f)
                self.discovery_events = [
                    DiscoveryEvent(
                        timestamp=e["timestamp"],
                        event_type=e["type"],
                        problem_id=e["problem_id"],
                        program_id=e.get("program_id"),
                        details=e.get("details", {}),
                    )
                    for e in events_data
                ]

        logger.info(f"Loaded discovery state from {path}")


# Convenience function for integration
def create_discovery_engine(
    openevolve: "OpenEvolve",
    problem_description: str,
    config: Optional[DiscoveryConfig] = None,
) -> DiscoveryEngine:
    """
    Create a DiscoveryEngine and attach it to an OpenEvolve instance.

    Args:
        openevolve: The OpenEvolve controller
        problem_description: Natural language description of the initial problem
        config: Optional configuration

    Returns:
        Configured DiscoveryEngine
    """
    config = config or DiscoveryConfig()

    engine = DiscoveryEngine(config, openevolve)
    engine.set_genesis_problem(
        problem_description,
        evaluator_path=openevolve.evaluation_file,
    )

    return engine
