"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.utils.metrics_utils import get_fitness_score, safe_numeric_average

logger = logging.getLogger(__name__)

_worker_thread_local = threading.local()


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: dict[str, Any] | None = None
    parent_id: str | None = None
    iteration_time: float = 0.0
    prompt: dict[str, str] | None = None
    llm_response: str | None = None
    artifacts: dict[str, Any] | None = None
    iteration: int = 0
    error: str | None = None
    # Meta-prompting attribution (captured in worker, rewarded in main)
    meta_prompt_strategy: str | None = None
    meta_prompt_context: dict[str, Any] | None = None
    meta_prompt_island: int | None = None


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict = None) -> None:
    """Initialize worker process with necessary components"""
    import os

    # Set environment from parent process
    if parent_env:
        os.environ.update(parent_env)

    global _worker_config
    global _worker_evaluation_file
    # Store config for later use
    _worker_config = Config.from_worker_dict(config_dict)
    _worker_evaluation_file = evaluation_file

    # Clear any cached thread-local components for this worker/thread.
    try:
        del _worker_thread_local.components
    except AttributeError:
        pass


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use (thread-safe)."""
    components = getattr(_worker_thread_local, "components", None)
    if components is not None:
        return components

    from openevolve.config import EvaluatorConfig
    from openevolve.evaluator import Evaluator
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.prompt.sampler import PromptSampler

    llm_ensemble = LLMEnsemble(_worker_config.llm.models)
    prompt_sampler = PromptSampler(_worker_config.prompt)

    # Create evaluator-specific components
    evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
    evaluator_prompt = PromptSampler(_worker_config.prompt)
    evaluator_prompt.set_templates("evaluator_system_message")

    # Copy evaluator config per thread to avoid cross-thread mutation and
    # allow per-iteration runtime overrides (e.g., timeout hot reload).
    evaluator_cfg = EvaluatorConfig(**asdict(_worker_config.evaluator))
    evaluator = Evaluator(
        evaluator_cfg,
        _worker_evaluation_file,
        evaluator_llm,
        evaluator_prompt,
        database=None,  # No shared database in worker
        suffix=getattr(_worker_config, "file_suffix", ".py"),
    )

    components = (evaluator, llm_ensemble, prompt_sampler)
    _worker_thread_local.components = components
    return components


def _run_iteration_worker(iteration: int, snapshot: dict[str, Any]) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        evaluator, llm_ensemble, prompt_sampler = _lazy_init_worker_components()

        runtime_overrides = snapshot.get("runtime_overrides") or {}
        try:
            evaluator_timeout = runtime_overrides.get("evaluator_timeout")
            if isinstance(evaluator_timeout, (int, float)) and float(evaluator_timeout) > 0:
                evaluator.config.timeout = int(evaluator_timeout)
        except Exception:
            pass

        parent = Program(**snapshot["parent"])
        inspirations = snapshot.get("inspirations", [])
        top_programs = snapshot.get("top_programs", [])
        previous_programs = snapshot.get("previous_programs", [])

        # Parent artifacts if available
        parent_artifacts = snapshot.get("program_artifacts")

        parent_island = parent.metadata.get("island", snapshot.get("island_idx", 0))

        # Build prompt (with discovery mode support)
        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=previous_programs,
            top_programs=top_programs,
            inspirations=inspirations,
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=snapshot.get("feature_dimensions", []),
            problem_context=snapshot.get("problem_context"),
            discovery_mode=snapshot.get("discovery_mode", False),
            island_idx=parent_island,
            generation=parent.generation,
        )

        # Capture meta-prompt strategy selection for main-process reward attribution
        meta_strategy = getattr(prompt_sampler, "_last_strategy_name", None)
        meta_context = getattr(prompt_sampler, "_last_strategy_context", None)
        meta_island = getattr(prompt_sampler, "_last_island_idx", None)

        iteration_start = time.time()

        # Generate code modification (sync wrapper for async)
        try:
            llm_response = asyncio.run(
                llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SerializableResult(error=f"LLM generation failed: {e!s}", iteration=iteration)

        # Check for None response
        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import apply_diff, extract_diffs, format_diff_summary

            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(
                    error="No valid diffs found in response", iteration=iteration
                )

            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error="No valid code found in response", iteration=iteration
                )

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        # Evaluate the child program
        import uuid

        child_id = str(uuid.uuid4())
        problem_context = snapshot.get("problem_context")
        max_stage = snapshot.get("max_evaluation_stage")
        if problem_context:
            child_metrics = asyncio.run(
                evaluator.evaluate_program(
                    child_code,
                    child_id,
                    problem_context=problem_context,
                    max_stage=max_stage,
                )
            )
        else:
            child_metrics = asyncio.run(
                evaluator.evaluate_program(child_code, child_id, max_stage=max_stage)
            )

        # Get artifacts
        artifacts = evaluator.get_pending_artifacts(child_id)

        # Update worker-local meta-prompting statistics for within-run learning
        if meta_strategy and getattr(prompt_sampler, "meta_prompt_evolver", None):
            try:
                feature_dims = snapshot.get("feature_dimensions", [])
                parent_fitness = get_fitness_score(parent.metrics, feature_dims)
                child_fitness = get_fitness_score(child_metrics, feature_dims)
                prompt_sampler.report_outcome(
                    parent_fitness=parent_fitness,
                    child_fitness=child_fitness,
                    island_idx=parent_island,
                )
            except Exception as e:
                logger.warning(f"Worker meta-prompt update failed: {e}")

        # Create child program
        child_program = Program(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
                "problem_id": snapshot.get("problem_id"),
            },
        )

        iteration_time = time.time() - iteration_start

        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
            meta_prompt_strategy=meta_strategy,
            meta_prompt_context=meta_context,
            meta_prompt_island=meta_island,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(error=str(e), iteration=iteration)


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        evolution_tracer=None,
        prompt_sampler=None,
        discovery_engine=None,
        file_suffix: str = ".py",
        config_path: str | None = None,
        hot_reload: bool = False,
        hot_reload_interval: float = 2.0,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.prompt_sampler = prompt_sampler
        self.discovery_engine = discovery_engine
        self.file_suffix = file_suffix
        self.config_path = config_path
        self.hot_reload_enabled = bool(hot_reload) and bool(config_path)
        self.hot_reload_interval = float(hot_reload_interval)
        self._hot_last_check: float = 0.0
        self._hot_last_mtime: float | None = None

        # Track last problem generation for single-problem discovery context updates.
        self._last_problem_generation: int = 0
        if self.discovery_engine is not None:
            current_problem = getattr(self.discovery_engine, "current_problem", None)
            if current_problem is not None:
                try:
                    self._last_problem_generation = int(current_problem.generation)
                except Exception:
                    self._last_problem_generation = 0

        self.executor: Executor | None = None
        self.executor_mode: str | None = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Track RL selections per iteration for safe parallel attribution
        self._rl_selections: dict[int, dict[str, Any]] = {}

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

    def _maybe_hot_reload_config(self) -> None:
        """Hot-reload a small set of config knobs from config_path during a run.

        This is intentionally conservative: it only applies settings that are safe
        to change mid-run (timeouts and skeptic budget knobs).
        """
        if not self.hot_reload_enabled or not self.config_path:
            return

        now = time.time()
        if now - self._hot_last_check < self.hot_reload_interval:
            return
        self._hot_last_check = now

        try:
            import os

            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return

        if self._hot_last_mtime is not None and mtime <= self._hot_last_mtime:
            return
        self._hot_last_mtime = mtime

        try:
            import yaml

            raw = Path(self.config_path).read_text(encoding="utf-8")
            cfg = yaml.safe_load(raw) or {}
        except Exception as e:
            logger.warning(f"Hot-reload skipped (failed to read {self.config_path}): {e}")
            return

        changed: list[str] = []

        # evaluator.timeout (affects worker-side asyncio timeouts and main future wait buffer).
        try:
            new_eval_timeout = cfg.get("evaluator", {}).get("timeout")
            if isinstance(new_eval_timeout, (int, float)):
                new_eval_timeout_i = int(new_eval_timeout)
                if new_eval_timeout_i > 0 and new_eval_timeout_i != int(
                    self.config.evaluator.timeout
                ):
                    old = self.config.evaluator.timeout
                    self.config.evaluator.timeout = new_eval_timeout_i
                    changed.append(f"evaluator.timeout {old} -> {new_eval_timeout_i}")
        except Exception:
            pass

        # discovery.skeptic.attack_timeout / num_attack_rounds.
        try:
            disc = cfg.get("discovery", {}) or {}
            sk = disc.get("skeptic", {}) or {}
            new_attack_timeout = sk.get("attack_timeout")
            if isinstance(new_attack_timeout, (int, float)):
                new_attack_timeout_f = float(new_attack_timeout)
                if new_attack_timeout_f > 0 and new_attack_timeout_f != float(
                    self.config.discovery.skeptic.attack_timeout
                ):
                    old = self.config.discovery.skeptic.attack_timeout
                    self.config.discovery.skeptic.attack_timeout = new_attack_timeout_f
                    changed.append(
                        f"discovery.skeptic.attack_timeout {old} -> {new_attack_timeout_f}"
                    )

            new_rounds = sk.get("num_attack_rounds")
            if (
                isinstance(new_rounds, int)
                and new_rounds >= 0
                and new_rounds != int(self.config.discovery.skeptic.num_attack_rounds)
            ):
                old = self.config.discovery.skeptic.num_attack_rounds
                self.config.discovery.skeptic.num_attack_rounds = int(new_rounds)
                changed.append(f"discovery.skeptic.num_attack_rounds {old} -> {new_rounds}")
        except Exception:
            pass

        if changed:
            logger.info(f"Hot-reloaded config ({self.config_path}): {', '.join(changed)}")

    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        config_dict = self.config.to_worker_dict()

        # Pass current environment to worker processes
        import os
        import sys

        current_env = dict(os.environ)

        executor_kwargs = {
            "max_workers": self.num_workers,
            "initializer": _worker_init,
            "initargs": (config_dict, self.evaluation_file, current_env),
        }
        if sys.version_info >= (3, 11):
            logger.info(f"Set max {self.config.max_tasks_per_child} tasks per child")
            executor_kwargs["max_tasks_per_child"] = self.config.max_tasks_per_child
        elif self.config.max_tasks_per_child is not None:
            logger.warning(
                "max_tasks_per_child is only supported in Python 3.11+. "
                "Ignoring max_tasks_per_child and using spawn start method."
            )
            executor_kwargs["mp_context"] = mp.get_context("spawn")

        # Create process pool with initializer (fallback to threads on restricted systems).
        try:
            self.executor = ProcessPoolExecutor(**executor_kwargs)
            self.executor_mode = "process"
            logger.info(f"Started process pool with {self.num_workers} processes")
            return
        except (PermissionError, OSError) as e:
            logger.warning(
                f"Process pool creation failed ({type(e).__name__}: {e}). "
                "Falling back to ThreadPoolExecutor."
            )

        thread_kwargs = {
            "max_workers": self.num_workers,
            "initializer": _worker_init,
            "initargs": (config_dict, self.evaluation_file, current_env),
        }
        self.executor = ThreadPoolExecutor(**thread_kwargs)
        self.executor_mode = "thread"
        logger.info(f"Started thread pool with {self.num_workers} workers")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def set_problem_context(self, problem_context: str) -> None:
        """Set the current problem context for discovery mode"""
        self._problem_context = problem_context

    def _create_iteration_snapshot(
        self,
        parent: Program,
        inspirations: list[Program],
        island_idx: int,
    ) -> dict[str, Any]:
        """Create a lightweight snapshot for a single worker iteration."""
        # Top programs for prompt context
        num_top = self.config.prompt.num_top_programs
        num_diverse = self.config.prompt.num_diverse_programs
        top_programs = self.database.get_top_programs(num_top, island_idx=island_idx)

        # Diverse elites sampled from different MAP-Elites cells
        exclude_ids = {parent.id}.union(p.id for p in top_programs)
        diverse_programs = self.database.get_diverse_programs(
            num_diverse, island_idx=island_idx, exclude_ids=exclude_ids
        )

        programs_for_prompt = top_programs + diverse_programs

        parent_artifacts = self.database.get_artifacts(parent.id)

        # Discovery mode: add curiosity samples as extra inspirations
        if (
            self.discovery_engine is not None
            and self.config.discovery.enabled
            and self.config.discovery.curiosity_sampling_enabled
        ):
            try:
                curiosity = self.discovery_engine.get_curiosity_samples(
                    n=self.config.prompt.num_diverse_programs
                )
                # Filter to same island to preserve isolation
                existing_ids = {p.id for p in inspirations}
                limit = (
                    self.config.prompt.num_top_programs + self.config.prompt.num_diverse_programs
                )
                for prog in curiosity:
                    if (
                        prog.id != parent.id
                        and prog.id not in existing_ids
                        and prog.metadata.get("island") == island_idx
                    ):
                        inspirations.append(prog)
                        existing_ids.add(prog.id)
                        if len(inspirations) >= limit:
                            break
            except Exception as e:
                logger.debug(f"Curiosity sampling failed: {e}")

        # Per-island problem context (coevolution mode) if available.
        problem_context = getattr(self, "_problem_context", None)
        problem_id = None
        if self.discovery_engine is not None:
            try:
                if hasattr(self.discovery_engine, "get_problem_context_for_island"):
                    problem_context = self.discovery_engine.get_problem_context_for_island(
                        island_idx
                    )
                if hasattr(self.discovery_engine, "get_problem_id_for_island"):
                    problem_id = self.discovery_engine.get_problem_id_for_island(island_idx)
            except Exception as e:
                logger.debug(f"Problem context lookup failed: {e}")

        # Compute-budgeted cascade: cap max evaluation stage for workers based on parent fitness.
        max_eval_stage = None
        if getattr(self.config.evaluator, "budgeted_cascade_enabled", False):
            try:
                parent_fitness = get_fitness_score(
                    parent.metrics, self.database.config.feature_dimensions
                )
                threshold = getattr(self.config.evaluator, "budget_stage3_parent_threshold", 0.6)
                if parent_fitness >= threshold:
                    max_eval_stage = int(getattr(self.config.evaluator, "budget_max_stage_high", 3))
                else:
                    max_eval_stage = int(getattr(self.config.evaluator, "budget_max_stage_low", 2))

                # If discovery is enabled, allocate more compute to high-surprise areas.
                if (
                    self.discovery_engine is not None
                    and getattr(self.config, "discovery", None)
                    and self.config.discovery.surprise_tracking_enabled
                ):
                    try:
                        predicted = self.discovery_engine.archive.predict_fitness(parent.code)
                        surprise = abs(parent_fitness - float(predicted))
                        if surprise > float(self.config.discovery.surprise_bonus_threshold):
                            max_eval_stage = int(
                                getattr(
                                    self.config.evaluator, "budget_max_stage_high", max_eval_stage
                                )
                            )
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Budgeted cascade selection failed: {e}")
            else:
                try:
                    logger.debug(
                        f"Budgeted cascade: island={island_idx}, parent={parent.id}, "
                        f"fitness={parent_fitness:.3f}, threshold={threshold:.3f} -> "
                        f"max_stage={max_eval_stage}"
                    )
                except Exception:
                    pass

        return {
            "parent": parent.to_dict(),
            "inspirations": [p.to_dict() for p in inspirations],
            "top_programs": [p.to_dict() for p in programs_for_prompt],
            "previous_programs": [p.to_dict() for p in top_programs],
            "feature_dimensions": self.database.config.feature_dimensions,
            "program_artifacts": parent_artifacts,
            "problem_context": problem_context,
            "problem_id": problem_id,
            "discovery_mode": self.config.discovery.enabled
            if hasattr(self.config, "discovery")
            else False,
            "island_idx": island_idx,
            "max_evaluation_stage": max_eval_stage,
            "runtime_overrides": {
                "evaluator_timeout": int(getattr(self.config.evaluator, "timeout", 0) or 0),
            },
        }

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: float | None = None,
        checkpoint_callback=None,
        program_callback=None,
    ):
        """Run evolution with process-based parallelism

        Args:
            start_iteration: Starting iteration number
            max_iterations: Maximum number of iterations to run
            target_score: Optional target score to stop at
            checkpoint_callback: Callback for checkpointing
            program_callback: Optional async callback called for each new program
                              Signature: async def callback(program: Program) -> None
        """
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures by island to maintain distribution
        pending_futures: dict[int, Future] = {}
        island_pending: dict[int, list[int]] = {i: [] for i in range(self.num_islands)}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch - distribute across islands
        batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        current_iteration = start_iteration

        # Round-robin distribution across islands
        for island_id in range(self.num_islands):
            for _ in range(batch_per_island):
                if current_iteration < total_iterations:
                    future = self._submit_iteration(current_iteration, island_id)
                    if future:
                        pending_futures[current_iteration] = future
                        island_pending[island_id].append(current_iteration)
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        # Island management
        programs_per_island = self.config.database.programs_per_island or max(
            1, max_iterations // (self.config.database.num_islands * 10)
        )
        current_island_counter = 0

        # Early stopping tracking
        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
            iterations_without_improvement = 0
            logger.info(
                f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.convergence_threshold}, "
                f"metric={self.config.early_stopping_metric}"
            )
        else:
            logger.info("Early stopping disabled")

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)

            try:
                # Use evaluator timeout + buffer to gracefully handle stuck processes
                timeout_seconds = self.config.evaluator.timeout + 30
                result = future.result(timeout=timeout_seconds)

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Attach artifacts early so discovery/Heisenberg can see them
                    if result.artifacts:
                        if child_program.metadata is None:
                            child_program.metadata = {}
                        # Keep a copy in metadata for downstream consumers
                        child_program.metadata.setdefault("artifacts", result.artifacts)

                    # Discovery mode pre-admission processing:
                    # Run skeptic/falsification BEFORE admitting to the database.
                    discovery_metadata: dict[str, Any] | None = None
                    falsification_passed = True
                    is_solution = False
                    if (
                        self.discovery_engine is not None
                        and hasattr(self.config, "discovery")
                        and self.config.discovery.enabled
                    ):
                        try:
                            (
                                is_solution,
                                discovery_metadata,
                            ) = await self.discovery_engine.process_program(child_program)
                            falsification_passed = discovery_metadata.get(
                                "falsification_passed", True
                            )

                            # Persist a compact discovery summary on admitted programs for debugging.
                            if falsification_passed and discovery_metadata:
                                if child_program.metadata is None:
                                    child_program.metadata = {}
                                existing = child_program.metadata.get("discovery")
                                discovery_summary = existing if isinstance(existing, dict) else {}
                                for k in (
                                    "problem_id",
                                    "falsification_passed",
                                    "surprise_score",
                                    "is_positive_surprise",
                                    "is_solution",
                                ):
                                    if k in discovery_metadata:
                                        discovery_summary[k] = discovery_metadata[k]
                                try:
                                    fr = discovery_metadata.get("falsification_results")
                                    if isinstance(fr, list) and fr:
                                        discovery_summary["falsification_attack_types"] = [
                                            r.get("attack_type") for r in fr if isinstance(r, dict)
                                        ][:10]
                                except Exception:
                                    pass
                                child_program.metadata["discovery"] = discovery_summary

                            # In single-problem discovery mode, refresh problem context
                            # for future worker prompts when a new generation appears.
                            if (
                                falsification_passed
                                and is_solution
                                and getattr(self.discovery_engine, "problem_archive", None) is None
                            ):
                                current_problem = getattr(
                                    self.discovery_engine, "current_problem", None
                                )
                                if current_problem is not None:
                                    try:
                                        current_gen = int(current_problem.generation)
                                    except Exception:
                                        current_gen = self._last_problem_generation
                                    if current_gen > self._last_problem_generation:
                                        self._last_problem_generation = current_gen
                                        self.set_problem_context(
                                            self.discovery_engine.get_current_problem_context()
                                        )
                                        logger.info(
                                            f"PROBLEM EVOLVED to generation {current_gen} "
                                            f"(difficulty: {getattr(current_problem, 'difficulty_level', 0.0):.1f})"
                                        )
                        except Exception as e:
                            logger.warning(f"Discovery processing failed: {e}")
                            falsification_passed = True

                    if falsification_passed:
                        # Admit to database if discovery engine didn't already add it
                        if child_program.id not in self.database.programs:
                            # Add to database (will auto-inherit parent's island)
                            self.database.add(child_program, iteration=completed_iteration)

                        # Store artifacts after admission
                        if result.artifacts:
                            self.database.store_artifacts(child_program.id, result.artifacts)
                    else:
                        logger.info(
                            f"Program {child_program.id} FALSIFIED - not admitted to database"
                        )

                    # Meta-prompting reward attribution (parallel-safe)
                    if (
                        self.prompt_sampler is not None
                        and getattr(result, "meta_prompt_strategy", None)
                        and getattr(self.prompt_sampler, "meta_prompt_evolver", None)
                    ):
                        parent_program = (
                            self.database.get(result.parent_id) if result.parent_id else None
                        )
                        if parent_program:
                            feature_dims = self.database.config.feature_dimensions
                            parent_fitness = get_fitness_score(parent_program.metrics, feature_dims)
                            # Penalize falsified children by zeroing fitness
                            raw_child_fitness = get_fitness_score(
                                child_program.metrics, feature_dims
                            )
                            child_fitness = raw_child_fitness if falsification_passed else 0.0
                            try:
                                self.prompt_sampler.report_explicit_outcome(
                                    strategy_name=result.meta_prompt_strategy,
                                    parent_fitness=parent_fitness,
                                    child_fitness=child_fitness,
                                    island_idx=result.meta_prompt_island
                                    or child_program.metadata.get(
                                        "island", self.database.current_island
                                    ),
                                    context=result.meta_prompt_context,
                                )
                            except Exception as e:
                                logger.warning(f"Meta-prompt reward attribution failed: {e}")

                    # RL reward attribution (parallel-safe)
                    rl_info = self._rl_selections.pop(completed_iteration, None)
                    if rl_info and self.database.rl_policy and self.database.rl_policy.enabled:
                        try:
                            feature_dims = self.database.config.feature_dimensions
                            raw_child_fitness = get_fitness_score(
                                child_program.metrics, feature_dims
                            )
                            child_fitness = raw_child_fitness if falsification_passed else 0.0
                            rl_policy = self.database.rl_policy
                            rl_policy.last_action = rl_info["action"]
                            rl_policy.last_state = rl_info["state"]
                            rl_policy.report_outcome(
                                parent_fitness=rl_info["parent_fitness"],
                                child_fitness=child_fitness,
                                island_idx=rl_info["island_idx"],
                            )
                        except Exception as e:
                            logger.warning(f"RL outcome report failed: {e}")

                    # Call user program callback if provided (not used for discovery now)
                    if program_callback is not None:
                        try:
                            await program_callback(child_program)
                        except Exception as e:
                            logger.warning(f"Program callback failed: {e}")

                    # Log evolution trace
                    if self.evolution_tracer:
                        # Retrieve parent program for trace logging
                        parent_program = (
                            self.database.get(result.parent_id) if result.parent_id else None
                        )
                        if parent_program:
                            # Determine island ID
                            island_id = child_program.metadata.get(
                                "island", self.database.current_island
                            )

                            self.evolution_tracer.log_trace(
                                iteration=completed_iteration,
                                parent_program=parent_program,
                                child_program=child_program,
                                prompt=result.prompt,
                                llm_response=result.llm_response,
                                artifacts=result.artifacts,
                                island_id=island_id,
                                metadata={
                                    "iteration_time": result.iteration_time,
                                    "changes": child_program.metadata.get("changes", ""),
                                },
                            )

                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # Island management
                    if (
                        completed_iteration > start_iteration
                        and current_island_counter >= programs_per_island
                    ):
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )

                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(
                            f"Checkpoint interval reached at iteration {completed_iteration}"
                        )
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        if (
                            "combined_score" in child_program.metrics
                            and child_program.metrics["combined_score"] >= target_score
                        ):
                            logger.info(
                                f"Target score {target_score} reached at iteration {completed_iteration}"
                            )
                            break

                    # Check early stopping
                    if early_stopping_enabled and child_program.metrics:
                        # Get the metric to track for early stopping
                        current_score = None
                        if self.config.early_stopping_metric in child_program.metrics:
                            current_score = child_program.metrics[self.config.early_stopping_metric]
                        elif self.config.early_stopping_metric == "combined_score":
                            # Default metric not found, use safe average (standard pattern)
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            # User specified a custom metric that doesn't exist
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' not found, using safe numeric average"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if current_score is not None and isinstance(current_score, (int, float)):
                            # Check for improvement
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                                logger.debug(
                                    f"New best score: {best_score:.4f} (improvement: {improvement:+.4f})"
                                )
                            else:
                                iterations_without_improvement += 1
                                logger.debug(
                                    f"No improvement: {iterations_without_improvement}/{self.config.early_stopping_patience}"
                                )

                            # Check if we should stop
                            if (
                                iterations_without_improvement
                                >= self.config.early_stopping_patience
                            ):
                                self.early_stopping_triggered = True
                                logger.info(
                                    f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                    f"No improvement for {iterations_without_improvement} iterations "
                                    f"(best score: {best_score:.4f})"
                                )
                                break

            except FutureTimeoutError:
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s "
                    f"(evaluator timeout: {self.config.evaluator.timeout}s + 30s buffer). "
                    f"Canceling future and continuing with next iteration."
                )
                # Cancel the future to clean up the process
                future.cancel()
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            # Remove completed iteration from island tracking
            for island_id, iteration_list in island_pending.items():
                if completed_iteration in iteration_list:
                    iteration_list.remove(completed_iteration)
                    break

            # Submit next iterations maintaining island balance
            for island_id in range(self.num_islands):
                if (
                    len(island_pending[island_id]) < batch_per_island
                    and next_iteration < total_iterations
                    and not self.shutdown_event.is_set()
                ):
                    future = self._submit_iteration(next_iteration, island_id)
                    if future:
                        pending_futures[next_iteration] = future
                        island_pending[island_id].append(next_iteration)
                        next_iteration += 1
                        break  # Only submit one iteration per completion to maintain balance

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Log completion reason
        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(self, iteration: int, island_id: int | None = None) -> Future | None:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            self._maybe_hot_reload_config()

            # Use specified island or current island
            target_island = island_id if island_id is not None else self.database.current_island

            # Use thread-safe sampling that doesn't modify shared state
            # This fixes the race condition from GitHub issue #246
            parent, inspirations = self.database.sample_from_island(
                island_id=target_island, num_inspirations=self.config.prompt.num_top_programs
            )

            # Capture RL selection info for this iteration if RL is enabled and used
            if (
                self.database.rl_policy
                and self.database.rl_policy.enabled
                and getattr(self.database, "_last_rl_action", None) is not None
            ):
                import copy

                feature_dims = self.database.config.feature_dimensions
                self._rl_selections[iteration] = {
                    "action": self.database.rl_policy.last_action,
                    "state": copy.deepcopy(self.database.rl_policy.last_state),
                    "parent_fitness": get_fitness_score(parent.metrics, feature_dims),
                    "island_idx": target_island,
                }

            # Create lightweight snapshot for this iteration
            snapshot = self._create_iteration_snapshot(parent, inspirations, target_island)

            # Submit to process pool
            future = self.executor.submit(_run_iteration_worker, iteration, snapshot)

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
