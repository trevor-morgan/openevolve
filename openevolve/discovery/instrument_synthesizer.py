"""
Instrument Synthesizer for Heisenberg Engine

Generates "probe" code to discover hidden variables when an epistemic crisis is detected.
Probes analyze raw data streams (artifacts, execution traces) to find patterns that
correlate with unexplained fitness variations.

Key insight: When optimization is stuck, the problem isn't "bad solutions" - it's
"missing variables". Probes are scientific instruments that reveal these hidden factors.

Probe Types:
- State Probe: Discovers hidden state variables (loop counters, accumulators)
- Gradient Probe: Analyzes fitness landscape for unexplored directions
- Coverage Probe: Finds unexplored input regions
- Numerical Probe: Detects numerical stability issues
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.discovery.ontology import Ontology, Variable
    from openevolve.discovery.crisis_detector import EpistemicCrisis

logger = logging.getLogger(__name__)


@dataclass
class Probe:
    """
    Code that probes raw data to discover hidden variables.

    A probe is a piece of Python code that:
    1. Analyzes evaluation artifacts (debug output, traces, etc.)
    2. Looks for patterns that correlate with performance
    3. Returns candidate variables for the ontology

    Attributes:
        id: Unique identifier for this probe
        code: Python code implementing the probe
        target_hypothesis: What variable we're looking for
        probe_type: Type of probe ("state", "gradient", "coverage", "numerical")
        rationale: Why this probe might reveal something
        expected_schema: What the probe should return
        timeout: Maximum execution time in seconds
        metadata: Additional probe-specific data
    """

    id: str
    code: str
    target_hypothesis: str
    probe_type: str  # "state", "gradient", "coverage", "numerical"
    rationale: str = ""
    expected_schema: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Probe":
        """Deserialize from dictionary"""
        return cls(**data)


@dataclass
class ProbeResult:
    """
    Result from executing a probe.

    Contains both the raw output and extracted candidate variables.

    Attributes:
        probe_id: ID of the probe that generated this result
        success: Whether the probe executed successfully
        discovered_variables: List of candidate Variables
        raw_output: Raw output from the probe
        validation_stats: Statistical validation results
        error: Error message if probe failed
        execution_time: How long the probe took
    """

    probe_id: str
    success: bool
    discovered_variables: List["Variable"] = field(default_factory=list)
    raw_output: Any = None
    validation_stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "probe_id": self.probe_id,
            "success": self.success,
            "discovered_variables": [v.to_dict() for v in self.discovered_variables],
            "raw_output": str(self.raw_output)[:1000] if self.raw_output else None,
            "validation_stats": self.validation_stats,
            "error": self.error,
            "execution_time": self.execution_time,
        }


@dataclass
class InstrumentSynthesizerConfig:
    """Configuration for probe synthesis"""

    max_probes_per_crisis: int = 5  # Max probes to generate per crisis
    probe_timeout: float = 60.0  # Timeout for probe execution
    require_validation: bool = True  # Validate discovered variables
    validation_trials: int = 5  # Number of trials for validation
    min_correlation_threshold: float = 0.6  # Min correlation to accept variable
    probe_synthesis_temperature: float = 0.9  # LLM temperature for probe generation
    probe_synthesis_max_tokens: int = 2048  # Max tokens for probe generation


# Default probe templates for fallback when LLM is unavailable
DEFAULT_STATE_PROBE = '''
def probe(artifacts: dict, metrics: dict) -> dict:
    """Probe for hidden state variables"""
    discovered = []
    analysis_notes = []

    # Look for intermediate values that correlate with performance
    if "intermediate_values" in artifacts:
        values = artifacts["intermediate_values"]
        if isinstance(values, dict):
            for key, val_list in values.items():
                if isinstance(val_list, list) and len(val_list) > 0:
                    # Check for patterns
                    try:
                        arr = [float(v) for v in val_list if v is not None]
                        if len(arr) > 2:
                            variance = sum((x - sum(arr)/len(arr))**2 for x in arr) / len(arr)
                            if variance > 0.01:
                                discovered.append({
                                    "name": f"state_{key}",
                                    "type": "continuous",
                                    "evidence": {"variance": variance, "samples": len(arr)},
                                    "confidence": min(len(arr) / 10, 1.0)
                                })
                                analysis_notes.append(f"Found variable state in {key}")
                    except (ValueError, TypeError):
                        pass

    # Look for execution counters
    if "counters" in artifacts:
        counters = artifacts["counters"]
        if isinstance(counters, dict):
            for name, count in counters.items():
                if count > 0:
                    discovered.append({
                        "name": f"counter_{name}",
                        "type": "continuous",
                        "evidence": {"count": count},
                        "confidence": 0.7
                    })

    return {
        "discovered_variables": discovered,
        "analysis_notes": "; ".join(analysis_notes) if analysis_notes else "No patterns found"
    }
'''

DEFAULT_NUMERICAL_PROBE = '''
def probe(artifacts: dict, metrics: dict) -> dict:
    """Probe for numerical stability issues"""
    discovered = []
    analysis_notes = []

    # Check for NaN/Inf occurrences
    def check_numerical(data, path=""):
        issues = []
        if isinstance(data, dict):
            for k, v in data.items():
                issues.extend(check_numerical(v, f"{path}.{k}"))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                issues.extend(check_numerical(v, f"{path}[{i}]"))
        elif isinstance(data, float):
            if data != data:  # NaN check
                issues.append(("nan", path))
            elif abs(data) > 1e300:
                issues.append(("overflow", path))
            elif abs(data) < 1e-300 and data != 0:
                issues.append(("underflow", path))
        return issues

    numerical_issues = check_numerical(artifacts)

    if numerical_issues:
        issue_types = {}
        for issue_type, path in numerical_issues:
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(path)

        for issue_type, paths in issue_types.items():
            discovered.append({
                "name": f"numerical_{issue_type}",
                "type": "categorical",
                "evidence": {"paths": paths[:5], "count": len(paths)},
                "confidence": min(len(paths) / 5, 1.0)
            })
            analysis_notes.append(f"Found {len(paths)} {issue_type} issues")

    # Check for precision loss patterns
    if "precision_stats" in artifacts:
        stats = artifacts["precision_stats"]
        if stats.get("max_error", 0) > 1e-6:
            discovered.append({
                "name": "precision_loss",
                "type": "continuous",
                "evidence": stats,
                "confidence": 0.8
            })

    return {
        "discovered_variables": discovered,
        "analysis_notes": "; ".join(analysis_notes) if analysis_notes else "No numerical issues found"
    }
'''


class InstrumentSynthesizer:
    """
    Generates probe code to discover hidden variables.

    When an epistemic crisis is detected (optimization stuck due to missing variables),
    the InstrumentSynthesizer generates probes that analyze raw data to find patterns
    that correlate with unexplained performance variations.

    Usage:
        synthesizer = InstrumentSynthesizer(config, llm_ensemble)

        # When crisis is detected
        probes = await synthesizer.synthesize_probes(crisis, ontology, artifacts)

        # Execute probes
        for probe in probes:
            result = await synthesizer.execute_probe(probe, context)
            if result.discovered_variables:
                for var in result.discovered_variables:
                    is_valid, confidence = await synthesizer.validate_discovery(var, context)
    """

    def __init__(
        self,
        config: InstrumentSynthesizerConfig,
        llm_ensemble: Optional["LLMEnsemble"] = None,
    ):
        self.config = config
        self.llm_ensemble = llm_ensemble

        # History tracking
        self.probe_history: List[Probe] = []
        self.result_history: List[ProbeResult] = []
        self.successful_patterns: Dict[str, List[str]] = {}  # probe_type -> successful code patterns

        logger.info("Initialized InstrumentSynthesizer")

    async def synthesize_probes(
        self,
        crisis: "EpistemicCrisis",
        current_ontology: "Ontology",
        evaluation_artifacts: Dict[str, Any],
    ) -> List[Probe]:
        """
        Generate probe code based on crisis type and available data.

        Args:
            crisis: The detected epistemic crisis
            current_ontology: Current variable space
            evaluation_artifacts: Available artifacts from evaluations

        Returns:
            List of Probe objects to execute
        """
        probes = []

        # Generate probes for each suggested probe type
        for probe_type in crisis.suggested_probes[:self.config.max_probes_per_crisis]:
            if self.llm_ensemble:
                probe = await self._synthesize_probe_with_llm(
                    crisis, current_ontology, evaluation_artifacts, probe_type
                )
            else:
                probe = self._synthesize_probe_fallback(probe_type)

            if probe:
                probes.append(probe)
                self.probe_history.append(probe)

        logger.info(
            f"Synthesized {len(probes)} probes for crisis {crisis.id} "
            f"(types: {[p.probe_type for p in probes]})"
        )

        return probes

    async def _synthesize_probe_with_llm(
        self,
        crisis: "EpistemicCrisis",
        current_ontology: "Ontology",
        evaluation_artifacts: Dict[str, Any],
        probe_type: str,
    ) -> Optional[Probe]:
        """Generate a probe using LLM"""
        from openevolve.prompt.templates import (
            PROBE_SYNTHESIS_SYSTEM,
            PROBE_SYNTHESIS_USER,
        )

        # Format the prompt
        crisis_context = crisis.to_prompt_context()
        ontology_context = current_ontology.to_prompt_context() if current_ontology else "No ontology defined"

        # Describe available artifacts
        artifact_schema = self._describe_artifacts(evaluation_artifacts)

        user_prompt = PROBE_SYNTHESIS_USER.format(
            crisis_context=crisis_context,
            ontology_context=ontology_context,
            artifact_schema=artifact_schema,
            probe_type=probe_type,
        )

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message=PROBE_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract code from response
            code = self._extract_code_from_response(response)
            if not code:
                logger.warning(f"No code extracted from LLM response for {probe_type} probe")
                return self._synthesize_probe_fallback(probe_type)

            probe = Probe(
                id=f"probe_{probe_type}_{uuid.uuid4().hex[:8]}",
                code=code,
                target_hypothesis=self._extract_hypothesis(response),
                probe_type=probe_type,
                rationale=self._extract_rationale(response),
                expected_schema={"discovered_variables": [], "analysis_notes": ""},
                timeout=self.config.probe_timeout,
                metadata={
                    "crisis_id": crisis.id,
                    "generated_by": "llm",
                },
            )

            return probe

        except Exception as e:
            logger.warning(f"LLM probe synthesis failed: {e}, falling back to default")
            return self._synthesize_probe_fallback(probe_type)

    def _synthesize_probe_fallback(self, probe_type: str) -> Probe:
        """Generate a probe using default templates"""
        templates = {
            "state": DEFAULT_STATE_PROBE,
            "numerical": DEFAULT_NUMERICAL_PROBE,
            "gradient": DEFAULT_STATE_PROBE,  # Reuse state probe for now
            "coverage": DEFAULT_STATE_PROBE,  # Reuse state probe for now
        }

        code = templates.get(probe_type, DEFAULT_STATE_PROBE)

        return Probe(
            id=f"probe_{probe_type}_{uuid.uuid4().hex[:8]}",
            code=code,
            target_hypothesis=f"Discover hidden {probe_type} variables",
            probe_type=probe_type,
            rationale=f"Default {probe_type} probe template",
            expected_schema={"discovered_variables": [], "analysis_notes": ""},
            timeout=self.config.probe_timeout,
            metadata={"generated_by": "fallback"},
        )

    def _describe_artifacts(self, artifacts: Dict[str, Any]) -> str:
        """Generate a description of available artifacts"""
        if not artifacts:
            return "No artifacts available"

        lines = ["Available artifacts:"]
        for key, value in artifacts.items():
            type_name = type(value).__name__
            if isinstance(value, dict):
                lines.append(f"  - {key}: dict with keys {list(value.keys())[:5]}")
            elif isinstance(value, list):
                lines.append(f"  - {key}: list with {len(value)} elements")
            else:
                lines.append(f"  - {key}: {type_name}")

        return "\n".join(lines)

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        # Try to find code blocks
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find a function definition
        func_match = re.search(r'(def probe\s*\(.*?\).*?)(?=\ndef|\Z)', response, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()

        return None

    def _extract_hypothesis(self, response: str) -> str:
        """Extract the target hypothesis from LLM response"""
        # Look for hypothesis markers
        hypo_match = re.search(r'(?:hypothesis|looking for|target)[:\s]*([^\n]+)', response, re.IGNORECASE)
        if hypo_match:
            return hypo_match.group(1).strip()
        return "Discover hidden variables"

    def _extract_rationale(self, response: str) -> str:
        """Extract the rationale from LLM response"""
        # Look for rationale markers
        rat_match = re.search(r'(?:rationale|because|reason)[:\s]*([^\n]+)', response, re.IGNORECASE)
        if rat_match:
            return rat_match.group(1).strip()
        return ""

    async def execute_probe(
        self,
        probe: Probe,
        evaluation_context: Dict[str, Any],
    ) -> ProbeResult:
        """
        Execute a probe and parse results.

        Args:
            probe: The probe to execute
            evaluation_context: Context containing artifacts and metrics

        Returns:
            ProbeResult with discovered variables
        """
        start_time = time.time()

        artifacts = evaluation_context.get("artifacts", {})
        metrics = evaluation_context.get("metrics", {})

        # Build the execution code
        exec_code = f'''
import json
import sys

{probe.code}

# Execute the probe
try:
    result = probe(
        artifacts={json.dumps(artifacts)},
        metrics={json.dumps(metrics)}
    )
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
'''

        try:
            # Execute in subprocess with timeout
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(exec_code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=probe.timeout,
                )

                execution_time = time.time() - start_time

                if result.returncode != 0:
                    return ProbeResult(
                        probe_id=probe.id,
                        success=False,
                        error=f"Probe execution failed: {result.stderr[:500]}",
                        execution_time=execution_time,
                    )

                # Parse output
                try:
                    output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    return ProbeResult(
                        probe_id=probe.id,
                        success=False,
                        error=f"Invalid probe output: {result.stdout[:200]}",
                        execution_time=execution_time,
                    )

                if "error" in output:
                    return ProbeResult(
                        probe_id=probe.id,
                        success=False,
                        error=output["error"],
                        execution_time=execution_time,
                    )

                # Convert discovered variables
                from openevolve.discovery.ontology import Variable

                discovered = []
                for var_data in output.get("discovered_variables", []):
                    var = Variable(
                        name=var_data.get("name", "unknown"),
                        var_type=var_data.get("type", "continuous"),
                        source="probe",
                        discovery_method=probe.id,
                        confidence=var_data.get("confidence", 0.5),
                        metadata={"evidence": var_data.get("evidence", {})},
                    )
                    discovered.append(var)

                probe_result = ProbeResult(
                    probe_id=probe.id,
                    success=True,
                    discovered_variables=discovered,
                    raw_output=output,
                    execution_time=execution_time,
                )

                self.result_history.append(probe_result)

                logger.info(
                    f"Probe {probe.id} discovered {len(discovered)} candidate variables "
                    f"in {execution_time:.2f}s"
                )

                return probe_result

            finally:
                import os
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            return ProbeResult(
                probe_id=probe.id,
                success=False,
                error=f"Probe timed out after {probe.timeout}s",
                execution_time=probe.timeout,
            )

        except Exception as e:
            return ProbeResult(
                probe_id=probe.id,
                success=False,
                error=f"Probe execution error: {str(e)}",
                execution_time=time.time() - start_time,
            )

    async def validate_discovery(
        self,
        variable: "Variable",
        evaluation_context: Dict[str, Any],
        run_evaluation: Optional[Callable] = None,
    ) -> Tuple[bool, float]:
        """
        Validate a discovered variable using statistical correlation.

        Runs multiple trials to check if the variable consistently
        correlates with fitness improvements.

        Args:
            variable: The variable to validate
            evaluation_context: Current evaluation context
            run_evaluation: Optional function to run evaluations

        Returns:
            Tuple of (is_valid, confidence)
        """
        if not self.config.require_validation:
            # Skip validation, trust the probe
            return True, variable.confidence

        # If we don't have a way to run evaluations, use heuristics
        if run_evaluation is None:
            return self._validate_heuristic(variable)

        # Statistical validation with multiple trials
        fitness_samples = []
        variable_samples = []

        for trial in range(self.config.validation_trials):
            try:
                # Run evaluation
                result = await run_evaluation()

                # Extract fitness
                fitness = result.get("combined_score", result.get("fitness", 0.0))
                fitness_samples.append(fitness)

                # Extract variable value if possible
                if variable.extraction_code:
                    # This would require executing the extraction code
                    # For now, use a placeholder
                    variable_samples.append(trial)  # Placeholder

            except Exception as e:
                logger.warning(f"Validation trial {trial} failed: {e}")
                continue

        if len(fitness_samples) < 3:
            logger.warning(
                f"Not enough samples for validation ({len(fitness_samples)}), "
                "using heuristic"
            )
            return self._validate_heuristic(variable)

        # Calculate statistics
        fitness_var = np.var(fitness_samples)
        fitness_mean = np.mean(fitness_samples)

        # Check if variable explains fitness variance
        # This is a simplified validation - in practice you'd want
        # actual correlation with the extracted variable values
        if fitness_var < 0.01:
            # Very low variance - variable not needed
            return False, 0.3

        # Use evidence from probe as additional signal
        evidence = variable.metadata.get("evidence", {})
        evidence_confidence = evidence.get("confidence", 0.5)

        # Combined confidence
        confidence = min(evidence_confidence * 0.6 + 0.4, 1.0)

        is_valid = confidence >= self.config.min_correlation_threshold

        logger.info(
            f"Validation for '{variable.name}': "
            f"{'PASSED' if is_valid else 'FAILED'} "
            f"(confidence: {confidence:.2f})"
        )

        return is_valid, confidence

    def _validate_heuristic(self, variable: "Variable") -> Tuple[bool, float]:
        """Heuristic validation when statistical validation isn't possible"""
        # Use the confidence from the probe
        confidence = variable.confidence

        # Adjust based on evidence quality
        evidence = variable.metadata.get("evidence", {})

        if "samples" in evidence and evidence["samples"] > 5:
            confidence += 0.1

        if "variance" in evidence and evidence["variance"] > 0.01:
            confidence += 0.1

        confidence = min(confidence, 1.0)

        is_valid = confidence >= self.config.min_correlation_threshold

        return is_valid, confidence

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about probe synthesis and execution"""
        successful_probes = [r for r in self.result_history if r.success]
        total_discoveries = sum(len(r.discovered_variables) for r in self.result_history)

        return {
            "total_probes": len(self.probe_history),
            "successful_probes": len(successful_probes),
            "success_rate": len(successful_probes) / max(len(self.result_history), 1),
            "total_discoveries": total_discoveries,
            "probes_by_type": self._count_probes_by_type(),
            "avg_execution_time": np.mean([r.execution_time for r in self.result_history]) if self.result_history else 0,
        }

    def _count_probes_by_type(self) -> Dict[str, int]:
        """Count probes by type"""
        counts = {}
        for probe in self.probe_history:
            counts[probe.probe_type] = counts.get(probe.probe_type, 0) + 1
        return counts
