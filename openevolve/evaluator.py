"""
Evaluation system for OpenEvolve
"""

import asyncio
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from typing import Any

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.async_utils import TaskPool
from openevolve.utils.dependency_manager import DependencyManager
from openevolve.utils.format_utils import format_metrics_safe

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates programs and assigns scores

    The evaluator is responsible for executing programs, measuring their performance,
    and assigning scores based on the evaluation criteria.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: LLMEnsemble | None = None,
        prompt_sampler: PromptSampler | None = None,
        database: ProgramDatabase | None = None,
        suffix: str | None = ".py",
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.program_suffix = suffix
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database

        # Initialize dependency manager
        self.dependency_manager = DependencyManager(
            getattr(config, "auto_install_dependencies", False)
        )

        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: dict[str, dict[str, str | bytes]] = {}

        logger.info(f"Initialized evaluator with {evaluation_file}")

    def _load_evaluation_function(self) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            eval_path = os.path.abspath(self.evaluation_file)
            digest = hashlib.md5(eval_path.encode("utf-8")).hexdigest()[:12]
            module_name = f"openevolve_evaluation_{digest}"
            self.evaluation_module_name = module_name

            spec = importlib.util.spec_from_file_location(module_name, self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Keep a reference to the loaded module for optional hooks (e.g., phenotype extractors).
            self.evaluation_module = module

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self.evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")

            # Detect whether evaluator supports evolving problem context.
            self.supports_problem_context = self._supports_problem_context(self.evaluate_function)

            # Optional phenotype extractor hook for discovery mode.
            self.custom_phenotype_extractor = None
            if hasattr(module, "extract_phenotype") and callable(module.extract_phenotype):
                self.custom_phenotype_extractor = module.extract_phenotype
            elif hasattr(module, "phenotype_extractor"):
                pe = module.phenotype_extractor
                if inspect.isclass(pe):
                    try:
                        pe = pe()
                    except Exception:
                        pe = None
                if pe is not None and callable(getattr(pe, "extract", None)):
                    self.custom_phenotype_extractor = pe.extract

            # Validate cascade configuration
            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {e!s}")
            raise

    def _supports_problem_context(self, func) -> bool:
        """Return True if evaluator function can accept problem_context."""
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if "problem_context" in sig.parameters:
                return True
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
                return True
            positional = [
                p
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            return len(positional) >= 2
        except Exception:
            return False

    def _validate_cascade_configuration(self, module) -> None:
        """
        Validate cascade evaluation configuration and warn about potential issues

        Args:
            module: The loaded evaluation module
        """
        if self.config.cascade_evaluation:
            # Check if cascade functions exist
            has_stage1 = hasattr(module, "evaluate_stage1")
            has_stage2 = hasattr(module, "evaluate_stage2")
            has_stage3 = hasattr(module, "evaluate_stage3")

            if not has_stage1:
                logger.warning(
                    f"Configuration has 'cascade_evaluation: true' but evaluator "
                    f"'{self.evaluation_file}' does not define 'evaluate_stage1' function. "
                    f"This will fall back to direct evaluation, making the cascade setting useless. "
                    f"Consider setting 'cascade_evaluation: false' or implementing cascade functions."
                )
            elif not (has_stage2 or has_stage3):
                logger.warning(
                    f"Evaluator '{self.evaluation_file}' defines 'evaluate_stage1' but no additional "
                    f"cascade stages (evaluate_stage2, evaluate_stage3). Consider implementing "
                    f"multi-stage evaluation for better cascade benefits."
                )
            else:
                logger.debug(
                    "Cascade evaluation properly configured with available stage functions"
                )

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
        problem_context: str | None = None,
        max_stage: int | None = None,
        use_llm_feedback: bool | None = None,
    ) -> dict[str, float]:
        """
        Evaluate a program and return scores

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging
            problem_context: Optional evolving problem context
            max_stage: If cascade enabled, cap evaluation to this stage.
            use_llm_feedback: Override LLM feedback for this evaluation.

        Returns:
            Dictionary of metric name to score
        """
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check dependencies if enabled
        # We do this before creating the temp file to ensure environment is ready
        self.dependency_manager.check_and_install(program_code)

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

        # Retry logic for evaluation
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            # Create a temporary file for the program
            with tempfile.NamedTemporaryFile(suffix=self.program_suffix, delete=False) as temp_file:
                temp_file.write(program_code.encode("utf-8"))
                temp_file_path = temp_file.name

            try:
                # Run evaluation
                if self.config.cascade_evaluation:
                    # Run cascade evaluation
                    result = await self._cascade_evaluate(
                        temp_file_path,
                        problem_context=problem_context,
                        max_stage=max_stage,
                    )
                else:
                    # Run direct evaluation
                    result = await self._direct_evaluate(
                        temp_file_path, problem_context=problem_context
                    )

                # Process the result based on type
                eval_result = self._process_evaluation_result(result)

                # Check if this was a timeout and capture artifacts if enabled
                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    self._pending_artifacts[program_id].update(
                        {
                            "timeout": True,
                            "timeout_duration": self.config.timeout,
                            "failure_stage": "evaluation",
                            "error_type": "timeout",
                        }
                    )

                # Add LLM feedback if configured
                llm_eval_result = None
                llm_feedback_enabled = (
                    self.config.use_llm_feedback
                    if use_llm_feedback is None
                    else bool(use_llm_feedback)
                )
                if llm_feedback_enabled and self.llm_ensemble:
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    llm_eval_result = self._process_evaluation_result(llm_result)

                    # Combine metrics from the processed result
                    llm_scores = []
                    for name, value in llm_eval_result.metrics.items():
                        weighted_value = value * self.config.llm_feedback_weight
                        eval_result.metrics[f"llm_{name}"] = weighted_value
                        llm_scores.append(value)  # Use unweighted value for average

                    # Add average of LLM metrics
                    if llm_scores:
                        llm_average = sum(llm_scores) / len(llm_scores)
                        eval_result.metrics["llm_average"] = (
                            llm_average * self.config.llm_feedback_weight
                        )

                        # Recalculate combined_score if it exists
                        if "combined_score" in eval_result.metrics:
                            # Preserve raw score and blend with LLM quality.
                            try:
                                accuracy = float(eval_result.metrics["combined_score"])
                                eval_result.metrics["combined_score_raw"] = accuracy
                                blend = float(self.config.llm_feedback_weight)
                                blend = max(0.0, min(blend, 1.0))
                                eval_result.metrics["combined_score"] = (
                                    accuracy * (1.0 - blend) + llm_average * blend
                                )
                            except (TypeError, ValueError):
                                logger.warning(
                                    "Skipping LLM blend due to non-numeric combined_score"
                                )

                # Store artifacts if enabled and present
                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    # Merge eval_result artifacts with llm artifacts if they exist
                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: {eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )

                # Return just metrics for backward compatibility
                return eval_result.metrics

            except asyncio.TimeoutError:
                # Handle timeout specially - don't retry, just return timeout result
                logger.warning(f"Evaluation timed out after {self.config.timeout}s")

                # Capture timeout artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

                return {"error": 0.0, "timeout": True}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {e!s}"
                )
                traceback.print_exc()

                # Capture failure artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }

                # If this is not the last attempt, wait a bit before retrying
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)  # Wait 1 second before retry

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        # All retries failed
        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {last_exception!s}"
        )
        return {"error": 0.0}

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance

        Handles three formats:
            1. EvaluationResult: Use directly
            2. Structured dict: {"score": ..., "metrics": {...}, "artifacts": {...}}
            3. Flat dict (legacy): {"metric1": 0.5, "metric2": 0.8}
        """
        if isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        elif isinstance(result, dict):
            # Check for structured format with nested metrics/artifacts
            if "metrics" in result and isinstance(result["metrics"], dict):
                # Structured format - extract metrics and artifacts
                metrics = dict(result["metrics"])

                # Also include top-level "score" if present
                if "score" in result and isinstance(result["score"], (int, float)):
                    # Use score as combined_score if not already present
                    if "combined_score" not in metrics:
                        metrics["combined_score"] = float(result["score"])

                # Extract artifacts if present
                artifacts = {}
                if "artifacts" in result and isinstance(result["artifacts"], dict):
                    artifacts = result["artifacts"]

                return EvaluationResult(metrics=metrics, artifacts=artifacts)
            else:
                # Flat dict format (legacy) - wrap in EvaluationResult
                return EvaluationResult.from_dict(result)
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> dict[str, str | bytes] | None:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

    def _call_evaluate(
        self, func, program_path: str, problem_context: str | None = None
    ) -> dict[str, float] | EvaluationResult:
        """Call an evaluator function with optional problem context.

        Backward compatible: if the evaluator doesn't accept a second argument,
        we call it with just program_path.
        """
        if problem_context is None:
            return func(program_path)

        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            positional = [
                p
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]

            if has_var_kw:
                try:
                    return func(program_path, problem_context=problem_context)
                except TypeError:
                    pass

            if len(positional) >= 2 or has_var_pos:
                return func(program_path, problem_context)
        except Exception:
            # If signature inspection fails, fall back to permissive call
            try:
                return func(program_path, problem_context)
            except TypeError:
                return func(program_path)

        return func(program_path)

    async def _direct_evaluate(
        self, program_path: str, problem_context: str | None = None
    ) -> dict[str, float] | EvaluationResult:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """

        # Create a coroutine that runs the evaluation function in an executor
        async def run_evaluation():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._call_evaluate, self.evaluate_function, program_path, problem_context
            )

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return result

    async def _cascade_evaluate(
        self,
        program_path: str,
        problem_context: str | None = None,
        max_stage: int | None = None,
    ) -> dict[str, float] | EvaluationResult:
        """
        Run cascade evaluation with increasingly challenging test cases

        Args:
            program_path: Path to the program file
            max_stage: Optional cap on cascade stage (1/2/3)

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts
        """
        try:
            module = getattr(self, "evaluation_module", None)
            if module is None or not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(program_path, problem_context=problem_context)

            # Run first stage with timeout
            try:

                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self._call_evaluate,
                        module.evaluate_stage1,
                        program_path,
                        problem_context,
                    )

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {e!s}")
                # Capture stage 1 failure with enhanced context
                error_context = self._create_cascade_error_context("stage1", e)
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        **error_context,
                    },
                )

            if max_stage is not None and max_stage <= 1:
                return stage1_eval_result

            # Check threshold
            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:

                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self._call_evaluate,
                        module.evaluate_stage2,
                        program_path,
                        problem_context,
                    )

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {e!s}")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result

            # Merge results from stage 1 and 2
            merged_metrics = {}
            # Convert all values to float to avoid type errors
            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            # Merge artifacts
            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

            if max_stage is not None and max_stage <= 2:
                return merged_result

            # Check threshold for stage 3
            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result

            # Check if third stage exists
            if not hasattr(module, "evaluate_stage3"):
                return merged_result

            # Run third stage with timeout
            try:

                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        self._call_evaluate,
                        module.evaluate_stage3,
                        program_path,
                        problem_context,
                    )

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {e!s}")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result

            # Merge stage 3 results
            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

        except Exception as e:
            logger.error(f"Error in cascade evaluation: {e!s}")
            # Return proper cascade failure result with enhanced context
            error_context = self._create_cascade_error_context("cascade_setup", e)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

    async def _llm_evaluate(
        self, program_code: str, program_id: str = ""
    ) -> EvaluationResult | dict[str, float]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            EvaluationResult on success, empty dict on error
        """
        if not self.llm_ensemble:
            return {}

        try:
            # Create prompt for LLM
            feature_dimensions = self.database.config.feature_dimensions if self.database else []
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code,
                template_key="evaluation",
                feature_dimensions=feature_dimensions,
            )

            # Get LLM response
            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )

            # Log prompt and response to database
            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )

            # Extract JSON from response
            try:
                # Try to find JSON block
                json_pattern = r"```json\n(.*?)\n```"
                import re

                artifacts = {}
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
                        # Remove non-JSON parts
                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                    # Parse JSON
                    result = json.loads(json_str)

                    # All non-numeric values are artifacts, all numeric values are metrics
                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)

                    # Weight of the model in the ensemble
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0

                    # Average the metrics
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {e!s}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e!s}")
            traceback.print_exc()
            return {}

    def _create_cascade_error_context(self, stage: str, error: Exception) -> dict:
        """
        Create rich error context for cascade failures

        Args:
            stage: The stage where the error occurred
            error: The exception that was raised

        Returns:
            Dictionary with enhanced error context
        """
        import time

        return {
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "cascade_config": self.config.cascade_evaluation,
            "cascade_thresholds": getattr(self.config, "cascade_thresholds", []),
            "timeout_config": self.config.timeout,
            "evaluation_file": self.evaluation_file,
        }

    def _passes_threshold(self, metrics: dict[str, float], threshold: float) -> bool:
        """
        Check if metrics pass a threshold

        Uses 'combined_score' if available (for consistency with evolution),
        otherwise falls back to averaging all numeric metrics except 'error'

        Args:
            metrics: Dictionary of metric name to score
            threshold: Threshold to pass

        Returns:
            True if metrics pass threshold
        """
        if not metrics:
            return False

        # Use combined_score if available - this is what evolution uses
        if "combined_score" in metrics:
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        # Fallback: average all numeric metrics except 'error'
        # This maintains backward compatibility
        valid_metrics = []
        for name, value in metrics.items():
            # Skip 'error' keys and ensure values are numeric
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """
        Evaluate multiple programs in parallel

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)
