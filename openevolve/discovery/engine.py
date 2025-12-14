"""
Discovery Engine - Main orchestrator for Open-Ended Scientific Discovery

This is the main integration point that combines:
1. ProblemEvolver - Evolves the questions being asked
2. AdversarialSkeptic - Falsification-based evaluation
3. EpistemicArchive - Behavioral diversity storage
4. HeisenbergEngine - Ontological expansion (discover new variables)

The Discovery Engine runs on TOP of OpenEvolve's existing evolution loop,
adding the key capabilities needed for true scientific discovery.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openevolve.controller import OpenEvolve
    from openevolve.database import Program

from openevolve.config import DiscoveryConfig
from openevolve.discovery.code_instrumenter import (
    CodeInstrumenter,
)
from openevolve.discovery.crisis_detector import (
    CrisisDetector,
    CrisisDetectorConfig,
    EpistemicCrisis,
)
from openevolve.discovery.epistemic_archive import (
    EpistemicArchive,
)
from openevolve.discovery.instrument_synthesizer import (
    InstrumentSynthesizer,
    InstrumentSynthesizerConfig,
)

# Heisenberg Engine imports (Ontological Expansion)
from openevolve.discovery.ontology import (
    Ontology,
    OntologyManager,
    Variable,
)
from openevolve.discovery.problem_archive import (
    ProblemArchive,
    ProblemArchiveConfig,
)
from openevolve.discovery.problem_space import (
    ProblemEvolver,
    ProblemEvolverConfig,
    ProblemSpace,
)
from openevolve.discovery.skeptic import AdversarialSkeptic

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryEvent:
    """Record of a discovery event"""

    timestamp: float
    event_type: str  # "solution", "problem_evolution", "surprise", "falsification"
    problem_id: str
    program_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


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

        # Multi-problem co-evolution archive (optional)
        self.problem_archive: ProblemArchive | None = None
        if self.config.coevolution_enabled:
            num_islands = getattr(openevolve.config.database, "num_islands", 1)
            archive_cfg = ProblemArchiveConfig(
                enabled=True,
                max_active_problems=self.config.max_active_problems,
                spawn_after_solutions=self.config.evolve_problem_after_solutions,
                novelty_threshold=self.config.novelty_threshold,
                min_difficulty=self.config.min_problem_difficulty,
                max_difficulty=self.config.max_problem_difficulty,
                min_islands_per_problem=self.config.min_islands_per_problem,
            )
            self.problem_archive = ProblemArchive(archive_cfg, num_islands=num_islands)

        # Initialize components
        self.problem_evolver = ProblemEvolver(
            config=ProblemEvolverConfig(),
            llm_ensemble=openevolve.llm_ensemble,
        )

        self.skeptic = (
            AdversarialSkeptic(
                config=config.skeptic,
                llm_ensemble=openevolve.llm_ensemble,
            )
            if config.skeptic_enabled
            else None
        )

        self.archive = EpistemicArchive(
            database=openevolve.database,
            phenotype_dimensions=config.phenotype_dimensions,
            custom_phenotype_extractor=getattr(
                getattr(openevolve, "evaluator", None),
                "custom_phenotype_extractor",
                None,
            ),
            mirror_dimensions=getattr(config, "phenotype_feature_dimensions", None),
        )

        # State tracking
        self.current_problem: ProblemSpace | None = None
        self.solutions_since_evolution: int = 0
        self.discovery_events: list[DiscoveryEvent] = []
        self._last_spawn_solution_count: dict[str, int] = {}

        # Statistics
        self.stats = {
            "total_iterations": 0,
            "successful_solutions": 0,
            "falsified_programs": 0,
            "problem_evolutions": 0,
            "high_surprise_events": 0,
            "epistemic_crises": 0,
            "ontology_expansions": 0,
        }

        # Heisenberg Engine components (initialized if enabled)
        self.ontology_manager: OntologyManager | None = None
        self.crisis_detector: CrisisDetector | None = None
        self.instrument_synthesizer: InstrumentSynthesizer | None = None
        self.code_instrumenter: CodeInstrumenter | None = None

        if getattr(config, "heisenberg", None) and config.heisenberg.enabled:
            self._init_heisenberg_engine()

        logger.info("Initialized DiscoveryEngine")

    def _init_heisenberg_engine(self) -> None:
        """Initialize Heisenberg Engine components for ontological expansion"""
        logger.info("Initializing Heisenberg Engine for ontological expansion")

        # Initialize Ontology Manager
        self.ontology_manager = OntologyManager()

        # Initialize Crisis Detector
        crisis_config = CrisisDetectorConfig(
            min_plateau_iterations=self.config.heisenberg.min_plateau_iterations,
            fitness_improvement_threshold=self.config.heisenberg.fitness_improvement_threshold,
            variance_window=self.config.heisenberg.variance_window,
            confidence_threshold=self.config.heisenberg.crisis_confidence_threshold,
        )
        self.crisis_detector = CrisisDetector(crisis_config)

        # Initialize Instrument Synthesizer
        synth_config = InstrumentSynthesizerConfig(
            max_probes_per_crisis=self.config.heisenberg.max_probes_per_crisis,
            probe_timeout=self.config.heisenberg.probe_timeout,
            validation_trials=self.config.heisenberg.validation_trials,
        )
        self.instrument_synthesizer = InstrumentSynthesizer(
            synth_config,
            self.openevolve.llm_ensemble,
        )

        # Initialize Code Instrumenter
        self.code_instrumenter = CodeInstrumenter()

        logger.info("Heisenberg Engine initialized successfully")

    def set_genesis_problem(
        self,
        description: str,
        evaluator_path: str | None = None,
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
        if self.problem_archive is not None:
            self.problem_archive.initialize_genesis(problem)

        self._log_event(
            DiscoveryEvent(
                timestamp=time.time(),
                event_type="genesis",
                problem_id=problem.id,
                details={"description": description},
            )
        )

        logger.info(f"Set genesis problem: {description[:100]}...")
        return problem

    async def process_program(
        self,
        program: "Program",
    ) -> tuple[bool, dict[str, Any]]:
        """
        Process a program through the discovery pipeline.

        This is the main integration point with OpenEvolve.
        Call this after OpenEvolve evaluates a program.

        Args:
            program: The program to process

        Returns:
            Tuple of (is_valid_solution, metadata_dict)
        """
        # Determine which problem this program was evaluated under.
        problem_id = None
        if program.metadata:
            problem_id = program.metadata.get("problem_id")

        problem = self.current_problem
        if self.problem_archive is not None and problem_id:
            rec = self.problem_archive.get_record(problem_id)
            if rec:
                problem = rec.problem
        if problem_id is None and problem is not None:
            problem_id = problem.id

        metadata = {
            "problem_id": problem_id,
            "falsification_passed": True,
            "surprise_score": 0.0,
            "is_solution": False,
        }

        # Step 1: Predict fitness for surprise calculation
        predicted_fitness = self.archive.predict_fitness(program.code)

        # Step 2: Adversarial falsification
        if self.skeptic and self.config.skeptic_enabled:
            description = problem.description if problem else ""

            survived, results = await self.skeptic.falsify(
                program,
                description=description,
                language=program.language or "python",
                fitness=program.metrics.get("combined_score", 0.0),
            )

            metadata["falsification_passed"] = survived
            metadata["falsification_results"] = [r.to_dict() for r in results]

            if not survived:
                self.stats["falsified_programs"] += 1
                self._log_event(
                    DiscoveryEvent(
                        timestamp=time.time(),
                        event_type="falsification",
                        problem_id=problem_id or "",
                        program_id=program.id,
                        details={
                            "attack_type": results[-1].attack_type if results else "unknown",
                            "error": results[-1].error_message if results else None,
                        },
                    )
                )

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
                self._log_event(
                    DiscoveryEvent(
                        timestamp=time.time(),
                        event_type="surprise",
                        problem_id=problem_id or "",
                        program_id=program.id,
                        details={
                            "predicted": surprise.predicted_fitness,
                            "actual": surprise.actual_fitness,
                            "surprise": surprise.surprise_score,
                            "positive": surprise.is_positive_surprise,
                        },
                    )
                )

        # Step 4: Check if this is a solution
        fitness = program.metrics.get("combined_score", 0.0)
        is_solution = fitness >= self.config.solution_threshold

        if is_solution:
            metadata["is_solution"] = True
            self.stats["successful_solutions"] += 1
            if self.problem_archive is None:
                # Single-problem mode
                self.solutions_since_evolution += 1
                if self.current_problem:
                    if program.id not in self.current_problem.solved_by:
                        self.current_problem.solved_by.append(program.id)
            else:
                # Co-evolution mode: mark solution on the specific problem
                if problem and program.id not in problem.solved_by:
                    problem.solved_by.append(program.id)

            self._log_event(
                DiscoveryEvent(
                    timestamp=time.time(),
                    event_type="solution",
                    problem_id=problem_id or "",
                    program_id=program.id,
                    details={
                        "fitness": fitness,
                        "phenotype": program.metadata.get("phenotype", {}),
                    },
                )
            )

            logger.info(
                f"SOLUTION FOUND: Program {program.id} solves problem "
                f"{problem_id or 'unknown'} "
                f"with fitness {fitness:.3f}"
            )

            # Step 5: Check if we should evolve the problem
            if self.problem_archive is None:
                if (
                    self.config.problem_evolution_enabled
                    and self.solutions_since_evolution >= self.config.evolve_problem_after_solutions
                ):
                    await self._evolve_problem()

        # Step 6: Heisenberg Engine - Check for epistemic crisis
        if self.crisis_detector is not None:
            # Record this evaluation for crisis analysis
            artifacts = {}
            if program.metadata and "artifacts" in program.metadata:
                artifacts = program.metadata.get("artifacts") or {}
            else:
                try:
                    artifacts = self.openevolve.database.get_artifacts(program.id) or {}
                except Exception:
                    artifacts = {}
                if artifacts:
                    if program.metadata is None:
                        program.metadata = {}
                    program.metadata["artifacts"] = artifacts
            self.crisis_detector.record_evaluation(
                iteration=self.stats["total_iterations"],
                metrics=program.metrics,
                artifacts=artifacts,
            )

            # Check for crisis
            crisis = self.crisis_detector.detect_crisis()
            if crisis:
                await self._handle_epistemic_crisis(crisis, program)
                metadata["crisis_detected"] = crisis.to_dict()

        # Record per-problem progress in co-evolution mode.
        if self.problem_archive is not None and problem_id:
            iter_idx = getattr(program, "iteration_found", self.stats["total_iterations"])
            try:
                self.problem_archive.record_program_result(
                    problem_id=problem_id,
                    fitness=float(fitness),
                    iteration=int(iter_idx),
                    program_id=program.id,
                    solution_threshold=self.config.solution_threshold,
                )
            except Exception as e:
                logger.debug(f"ProblemArchive record failed: {e}")

            # Spawn new problems POET-style when a problem is repeatedly solved.
            if (
                is_solution
                and self.config.problem_evolution_enabled
                and self.problem_archive.should_spawn_child(problem_id)
            ):
                rec = self.problem_archive.get_record(problem_id)
                if rec is not None:
                    last_spawn = self._last_spawn_solution_count.get(problem_id, 0)
                    spawn_every = self.config.evolve_problem_after_solutions
                    if rec.solutions - last_spawn >= spawn_every:
                        spawned = await self._spawn_child_problem(rec.problem)
                        if spawned is not None:
                            self._last_spawn_solution_count[problem_id] = rec.solutions

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

        self._log_event(
            DiscoveryEvent(
                timestamp=time.time(),
                event_type="problem_evolution",
                problem_id=new_problem.id,
                details={
                    "parent_problem_id": self.current_problem.id,
                    "new_constraints": new_problem.constraints,
                    "new_difficulty": new_problem.difficulty_level,
                    "solution_characteristics": solution_characteristics,
                },
            )
        )

        self.current_problem = new_problem
        self.solutions_since_evolution = 0
        self.stats["problem_evolutions"] += 1

        logger.info(
            f"PROBLEM EVOLVED: {new_problem.id} "
            f"(generation {new_problem.generation}, difficulty {new_problem.difficulty_level:.1f})"
        )

    async def _spawn_child_problem(self, parent_problem: ProblemSpace) -> ProblemSpace | None:
        """Spawn a mutated child problem and assign an island (coevolution mode)."""
        solution_characteristics = self._get_solution_characteristics(parent_problem)
        new_problem = await self.problem_evolver.evolve(
            parent_problem,
            solution_characteristics,
        )

        if self.problem_archive is None:
            # Fallback to legacy behavior
            self.current_problem = new_problem
            self.solutions_since_evolution = 0
            self.stats["problem_evolutions"] += 1
            return new_problem

        if not await self._passes_minimal_criterion(new_problem):
            logger.info(
                f"ProblemArchive minimal-criterion rejected candidate {new_problem.id} "
                f"(parent {parent_problem.id})"
            )
            return None

        if not self.problem_archive.admit_candidate(new_problem):
            logger.info(
                f"ProblemArchive rejected candidate {new_problem.id} "
                f"(parent {parent_problem.id})"
            )
            return None

        self.problem_archive.add_problem(new_problem)
        island = self.problem_archive.allocate_island_for_new_problem()
        if island is not None:
            self.problem_archive.assign_island(new_problem.id, island)

        self.problem_evolver.current_problem = new_problem
        self.current_problem = new_problem
        self.stats["problem_evolutions"] += 1

        self._log_event(
            DiscoveryEvent(
                timestamp=time.time(),
                event_type="problem_evolution",
                problem_id=new_problem.id,
                details={
                    "parent_problem_id": parent_problem.id,
                    "new_constraints": new_problem.constraints,
                    "new_difficulty": new_problem.difficulty_level,
                    "solution_characteristics": solution_characteristics,
                    "assigned_island": island,
                },
            )
        )

        logger.info(
            f"PROBLEM SPAWNED: {new_problem.id} "
            f"(parent {parent_problem.id}, difficulty {new_problem.difficulty_level:.1f}, "
            f"island {island})"
        )
        return new_problem

    async def _passes_minimal_criterion(self, candidate: ProblemSpace) -> bool:
        """Screen candidate problems by testing transfer from existing solvers.

        A candidate is admitted if the best transferred solver fitness is:
            min_transfer_fitness <= best < max_transfer_fitness.
        """
        if self.problem_archive is None:
            return True

        try:
            trial_n = max(1, int(self.config.transfer_trial_programs))
        except Exception:
            trial_n = 3

        # If we have no solvers yet, accept.
        top_programs = self.openevolve.database.get_top_programs(trial_n)
        if not top_programs:
            return True

        context = candidate.to_prompt_context()

        best_fitness = 0.0
        trial_results: list[float] = []

        import uuid

        from openevolve.utils.metrics_utils import get_fitness_score

        for prog in top_programs:
            try:
                metrics = await self.openevolve.evaluator.evaluate_program(
                    prog.code,
                    program_id=f"transfer_{prog.id}_{candidate.id}_{uuid.uuid4().hex[:6]}",
                    problem_context=context,
                    max_stage=self.config.transfer_max_stage,
                    use_llm_feedback=False,
                )

                if "combined_score_raw" in metrics:
                    fitness = float(metrics["combined_score_raw"])
                elif "combined_score" in metrics:
                    fitness = float(metrics["combined_score"])
                else:
                    fitness = float(
                        get_fitness_score(
                            metrics, self.openevolve.database.config.feature_dimensions
                        )
                    )

                trial_results.append(fitness)
                best_fitness = max(best_fitness, fitness)
            except Exception as e:
                logger.debug(f"Transfer screening failed for {prog.id}: {e}")

        ceiling = (
            float(self.config.max_transfer_fitness)
            if self.config.max_transfer_fitness is not None
            else float(self.config.solution_threshold)
        )
        floor = float(self.config.min_transfer_fitness)

        passed = (best_fitness >= floor) and (best_fitness < ceiling)

        logger.info(
            f"Transfer screening candidate {candidate.id} (parent {candidate.parent_id}): "
            f"best={best_fitness:.3f}, floor={floor:.3f}, ceiling={ceiling:.3f} -> "
            f"{'PASS' if passed else 'REJECT'}"
        )

        # Log a screening event for transparency
        self._log_event(
            DiscoveryEvent(
                timestamp=time.time(),
                event_type="candidate_screening",
                problem_id=candidate.id,
                details={
                    "parent_problem_id": candidate.parent_id,
                    "transfer_best_fitness": best_fitness,
                    "transfer_trials": trial_results,
                    "floor": floor,
                    "ceiling": ceiling,
                    "passed": passed,
                },
            )
        )

        return passed

    async def _handle_epistemic_crisis(
        self,
        crisis: EpistemicCrisis,
        triggering_program: "Program",
    ) -> None:
        """
        Handle a detected epistemic crisis through ontological expansion.

        This is the core of the Heisenberg Engine - when optimization is stuck
        due to missing variables, we:
        1. Synthesize probes to discover hidden variables
        2. Execute probes and validate discoveries
        3. Expand the ontology with validated variables
        4. Perform a soft reset to continue with new knowledge
        """
        logger.warning(
            f"EPISTEMIC CRISIS DETECTED: {crisis.crisis_type} "
            f"(confidence: {crisis.confidence:.2f}, severity: {crisis.get_severity()})"
        )

        self.stats["epistemic_crises"] += 1

        # Log the crisis event
        self._log_event(
            DiscoveryEvent(
                timestamp=time.time(),
                event_type="epistemic_crisis",
                problem_id=self.current_problem.id if self.current_problem else "",
                program_id=triggering_program.id,
                details={
                    "crisis_type": crisis.crisis_type,
                    "confidence": crisis.confidence,
                    "evidence": crisis.evidence,
                    "suggested_probes": crisis.suggested_probes,
                },
            )
        )

        # Get current ontology (or create genesis if none exists)
        if self.ontology_manager.current_ontology is None:
            self.ontology_manager.create_genesis_ontology()

        # Synthesize probes
        logger.info("Synthesizing probes for hidden variable discovery...")
        probes = await self.instrument_synthesizer.synthesize_probes(
            crisis=crisis,
            current_ontology=self.ontology_manager.current_ontology,
            evaluation_artifacts=triggering_program.metadata.get("artifacts", {}),
        )

        # Execute probes and collect discoveries
        discovered_variables: list[Variable] = []

        for probe in probes:
            logger.info(f"Executing probe: {probe.id} ({probe.probe_type})")

            result = await self.instrument_synthesizer.execute_probe(
                probe=probe,
                evaluation_context={
                    "artifacts": triggering_program.metadata.get("artifacts", {}),
                    "metrics": triggering_program.metrics,
                },
            )

            if result.success and result.discovered_variables:
                # Validate each discovered variable
                for var in result.discovered_variables:
                    logger.info(f"Validating discovered variable: '{var.name}'")

                    is_valid, confidence = await self.instrument_synthesizer.validate_discovery(
                        variable=var,
                        evaluation_context={
                            "artifacts": triggering_program.metadata.get("artifacts", {}),
                            "metrics": triggering_program.metrics,
                        },
                    )

                    if is_valid:
                        var.confidence = confidence
                        discovered_variables.append(var)
                        logger.info(
                            f"VARIABLE VALIDATED: '{var.name}' "
                            f"(type: {var.var_type}, confidence: {confidence:.2f})"
                        )
                    else:
                        logger.info(f"Variable '{var.name}' failed validation")

        # Expand ontology if we discovered valid variables
        if discovered_variables:
            logger.info(
                f"ONTOLOGY EXPANSION: Adding {len(discovered_variables)} variables: "
                f"{[v.name for v in discovered_variables]}"
            )

            new_ontology = self.ontology_manager.expand_ontology(
                new_variables=discovered_variables,
                discovered_via=crisis.id,
            )

            self.stats["ontology_expansions"] += 1

            # Log the expansion event
            self._log_event(
                DiscoveryEvent(
                    timestamp=time.time(),
                    event_type="ontology_expansion",
                    problem_id=self.current_problem.id if self.current_problem else "",
                    details={
                        "new_variables": [v.name for v in discovered_variables],
                        "ontology_generation": new_ontology.generation,
                        "triggered_by_crisis": crisis.id,
                        "variable_details": [v.to_dict() for v in discovered_variables],
                    },
                )
            )

            # Perform soft reset
            await self._perform_soft_reset(new_ontology)

        else:
            logger.info("No valid variables discovered - continuing without expansion")

    async def _perform_soft_reset(self, new_ontology: Ontology) -> None:
        """
        Perform a soft reset after ontology expansion.

        Keeps top programs but resets crisis detector and updates problem context
        with new ontology information.
        """
        logger.info("Performing soft reset after ontology expansion...")

        # Keep a small, high-quality working set to restart exploration.
        keep_n = int(getattr(self.config.heisenberg, "programs_to_keep_on_reset", 0) or 0)
        if keep_n > 0 and getattr(self.openevolve, "database", None) is not None:
            db = self.openevolve.database
            get_top = getattr(db, "get_top_programs", None)
            retain = getattr(db, "retain_programs", None)
            if callable(get_top) and callable(retain):
                try:
                    before = len(getattr(db, "programs", {}) or {})
                    keep_ids = {p.id for p in get_top(keep_n) if getattr(p, "id", None)}
                    if keep_ids:
                        retain(keep_ids)
                        after = len(getattr(db, "programs", {}) or {})
                        logger.info(
                            "Soft reset pruned programs: %d -> %d (kept %d)",
                            before,
                            after,
                            len(keep_ids),
                        )
                except Exception as e:
                    logger.debug(f"Soft reset program pruning failed: {e}")

        # Reset crisis detector to start fresh analysis with new ontology
        self.crisis_detector.reset()

        # Update archive for new ontology (if it supports it)
        if hasattr(self.archive, "update_for_ontology"):
            new_vars = new_ontology.metadata.get("new_variables", [])
            self.archive.update_for_ontology(
                ontology_generation=new_ontology.generation,
                new_variables=new_vars,
            )

        # Update problem context with new ontology
        if self.current_problem:
            updated_problem = self.current_problem.update_for_ontology(
                ontology_id=new_ontology.id,
                ontology_generation=new_ontology.generation,
                variable_names=new_ontology.get_variable_names(),
                variable_descriptions={v.name: v.description for v in new_ontology.variables},
            )
            # Track and switch to updated problem context
            self.problem_evolver.problem_history[updated_problem.id] = updated_problem
            self.problem_evolver.current_problem = updated_problem
            self.current_problem = updated_problem

        # Log the soft reset
        self._log_event(
            DiscoveryEvent(
                timestamp=time.time(),
                event_type="soft_reset",
                problem_id=self.current_problem.id if self.current_problem else "",
                details={
                    "new_ontology_generation": new_ontology.generation,
                    "new_variables": [
                        v.name for v in new_ontology.variables if v.source == "probe"
                    ],
                    "programs_kept": self.config.heisenberg.programs_to_keep_on_reset,
                },
            )
        )

        logger.info(f"Soft reset complete. Ontology now at generation {new_ontology.generation}")

    def _get_solution_characteristics(self, problem: ProblemSpace | None = None) -> dict[str, Any]:
        """Get aggregate characteristics of successful solutions for a problem."""
        solutions = []

        target_problem = problem or self.current_problem
        if target_problem:
            for program_id in target_problem.solved_by:
                program = self.openevolve.database.programs.get(program_id)
                if program:
                    solutions.append(program)

        if not solutions:
            return {}

        # Aggregate characteristics
        characteristics = {
            "num_solutions": len(solutions),
            "avg_fitness": sum(p.metrics.get("combined_score", 0) for p in solutions)
            / len(solutions),
            "avg_complexity": sum(
                p.metadata.get("phenotype", {}).get("complexity", 0) for p in solutions
            )
            / len(solutions),
            "approaches": list(
                set(
                    p.metadata.get("phenotype", {}).get("approach_signature", "") for p in solutions
                )
            ),
        }

        return characteristics

    def get_curiosity_samples(self, n: int = 3) -> list["Program"]:
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

    def get_problem_context_for_island(self, island_idx: int) -> str:
        """Get problem context for a specific island (coevolution mode)."""
        if self.problem_archive is not None:
            prob = self.problem_archive.get_problem_for_island(island_idx)
            if prob is not None:
                return prob.to_prompt_context()
        return self.get_current_problem_context()

    def get_problem_id_for_island(self, island_idx: int) -> str | None:
        if self.problem_archive is not None:
            prob = self.problem_archive.get_problem_for_island(island_idx)
            if prob is not None:
                return prob.id
        return self.current_problem.id if self.current_problem else None

    def _log_event(self, event: DiscoveryEvent) -> None:
        """Log a discovery event"""
        self.discovery_events.append(event)

        if self.config.log_discoveries and self.config.discovery_log_path:
            try:
                os.makedirs(os.path.dirname(self.config.discovery_log_path), exist_ok=True)
                with open(self.config.discovery_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": event.timestamp,
                                "type": event.event_type,
                                "problem_id": event.problem_id,
                                "program_id": event.program_id,
                                "details": event.details,
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                logger.warning(f"Failed to log discovery event: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get discovery statistics"""
        stats = {
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

        if self.problem_archive is not None:
            ordered = sorted(self.problem_archive.records.values(), key=lambda r: r.created_at)
            stats["active_problems"] = [
                {
                    "id": rec.problem.id,
                    "parent_id": rec.problem.parent_id,
                    "generation": rec.problem.generation,
                    "difficulty": rec.problem.difficulty_level,
                    "solutions": rec.solutions,
                    "best_fitness": rec.best_fitness,
                    "islands": rec.island_ids,
                }
                for rec in ordered
            ]
            stats["island_problem_map"] = dict(self.problem_archive.island_to_problem)

        # Add Heisenberg stats if enabled
        if self.ontology_manager is not None:
            stats["ontology_stats"] = self.ontology_manager.get_statistics()
        if self.crisis_detector is not None:
            stats["crisis_detector_stats"] = self.crisis_detector.get_statistics()
        if self.instrument_synthesizer is not None:
            stats["probe_stats"] = self.instrument_synthesizer.get_statistics()

        return stats

    def save_state(self, path: str) -> None:
        """Save discovery state to disk"""
        os.makedirs(path, exist_ok=True)

        # Save problem history
        self.problem_evolver.save(os.path.join(path, "problems.json"))

        # Save co-evolution archive state if enabled
        if self.problem_archive is not None:
            archive_data = {
                "records": {
                    pid: {
                        "best_fitness": rec.best_fitness,
                        "solutions": rec.solutions,
                        "last_improvement_iteration": rec.last_improvement_iteration,
                        "island_ids": rec.island_ids,
                        "created_at": rec.created_at,
                    }
                    for pid, rec in self.problem_archive.records.items()
                },
                "island_to_problem": dict(self.problem_archive.island_to_problem),
                "current_problem_id": self.current_problem.id if self.current_problem else None,
            }
            with open(os.path.join(path, "problem_archive.json"), "w") as f:
                json.dump(archive_data, f, indent=2)

        # Save events
        with open(os.path.join(path, "events.json"), "w") as f:
            json.dump(
                [
                    {
                        "timestamp": e.timestamp,
                        "type": e.event_type,
                        "problem_id": e.problem_id,
                        "program_id": e.program_id,
                        "details": e.details,
                    }
                    for e in self.discovery_events
                ],
                f,
                indent=2,
            )

        # Save statistics
        with open(os.path.join(path, "stats.json"), "w") as f:
            json.dump(self.get_statistics(), f, indent=2)

        # Save Heisenberg state if enabled
        if self.ontology_manager is not None:
            heisenberg_path = os.path.join(path, "heisenberg")
            os.makedirs(heisenberg_path, exist_ok=True)

            # Save ontology history
            self.ontology_manager.save(os.path.join(heisenberg_path, "ontology.json"))

            # Save crisis history
            if self.crisis_detector is not None:
                crisis_data = {
                    "fitness_history": self.crisis_detector.fitness_history,
                    "artifact_history": [
                        {k: str(v)[:1000] for k, v in art.items()}
                        for art in self.crisis_detector.artifact_history[-100:]  # Last 100
                    ],
                    "crisis_history": [
                        crisis.to_dict() for crisis in self.crisis_detector.crisis_history
                    ],
                    "last_crisis_iteration": self.crisis_detector.last_crisis_iteration,
                }
                with open(os.path.join(heisenberg_path, "crisis_history.json"), "w") as f:
                    json.dump(crisis_data, f, indent=2)

            # Save probe history
            if self.instrument_synthesizer is not None:
                probe_data = {
                    "executed_probes": self.instrument_synthesizer.executed_probes,
                    "discovered_variables": [
                        {
                            "name": v.name,
                            "var_type": v.var_type,
                            "source": v.source,
                            "confidence": v.confidence,
                        }
                        for v in self.instrument_synthesizer.discovered_variables
                    ],
                }
                with open(os.path.join(heisenberg_path, "probe_history.json"), "w") as f:
                    json.dump(probe_data, f, indent=2)

            logger.info(f"Saved Heisenberg state to {heisenberg_path}")

        logger.info(f"Saved discovery state to {path}")

    def load_state(self, path: str) -> None:
        """Load discovery state from disk"""
        # Load problem history
        problems_path = os.path.join(path, "problems.json")
        if os.path.exists(problems_path):
            self.problem_evolver.load(problems_path)
            self.current_problem = self.problem_evolver.current_problem

        # Load archive state if coevolution enabled
        archive_path = os.path.join(path, "problem_archive.json")
        if os.path.exists(archive_path) and self.problem_archive is not None:
            with open(archive_path) as f:
                archive_data = json.load(f)

            # Rebuild archive records from problem history
            self.problem_archive.records.clear()
            for pid, rec_data in archive_data.get("records", {}).items():
                prob = self.problem_evolver.problem_history.get(pid)
                if prob is None:
                    continue
                self.problem_archive.add_problem(prob)
                rec = self.problem_archive.records[pid]
                rec.best_fitness = float(rec_data.get("best_fitness", 0.0))
                rec.solutions = int(rec_data.get("solutions", len(prob.solved_by)))
                rec.last_improvement_iteration = int(rec_data.get("last_improvement_iteration", 0))
                rec.island_ids = list(rec_data.get("island_ids", []))
                rec.created_at = float(rec_data.get("created_at", time.time()))

            self.problem_archive.island_to_problem = {
                int(k): v for k, v in archive_data.get("island_to_problem", {}).items()
            }

            current_pid = archive_data.get("current_problem_id")
            if current_pid and current_pid in self.problem_evolver.problem_history:
                self.current_problem = self.problem_evolver.problem_history[current_pid]

        # Load events
        events_path = os.path.join(path, "events.json")
        if os.path.exists(events_path):
            with open(events_path) as f:
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

        # Load Heisenberg state if it exists
        heisenberg_path = os.path.join(path, "heisenberg")
        if os.path.exists(heisenberg_path):
            # Load ontology history
            ontology_path = os.path.join(heisenberg_path, "ontology.json")
            if os.path.exists(ontology_path) and self.ontology_manager is not None:
                self.ontology_manager.load(ontology_path)

            # Load crisis history
            crisis_path = os.path.join(heisenberg_path, "crisis_history.json")
            if os.path.exists(crisis_path) and self.crisis_detector is not None:
                with open(crisis_path) as f:
                    crisis_data = json.load(f)
                    self.crisis_detector.fitness_history = crisis_data.get("fitness_history", [])
                    self.crisis_detector.last_crisis_iteration = crisis_data.get(
                        "last_crisis_iteration", 0
                    )
                    # Restore crisis history
                    self.crisis_detector.crisis_history = [
                        EpistemicCrisis.from_dict(c) for c in crisis_data.get("crisis_history", [])
                    ]

            # Load probe history
            probe_path = os.path.join(heisenberg_path, "probe_history.json")
            if os.path.exists(probe_path) and self.instrument_synthesizer is not None:
                with open(probe_path) as f:
                    probe_data = json.load(f)
                    self.instrument_synthesizer.executed_probes = probe_data.get(
                        "executed_probes", 0
                    )
                    # Note: discovered_variables are derived from ontology, don't reload

            logger.info(f"Loaded Heisenberg state from {heisenberg_path}")

        logger.info(f"Loaded discovery state from {path}")


# Convenience function for integration
def create_discovery_engine(
    openevolve: "OpenEvolve",
    problem_description: str,
    config: DiscoveryConfig | None = None,
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
