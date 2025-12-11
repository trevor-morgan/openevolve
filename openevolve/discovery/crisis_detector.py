"""
Crisis Detector for Heisenberg Engine

Detects "epistemic crises" - situations where optimization is fundamentally stuck
because the model is missing a variable, not just because it hasn't found a good solution.

Key insight: There's a difference between:
- "We haven't optimized well enough" (keep trying)
- "Our model cannot represent the solution" (need new variables)

The Crisis Detector distinguishes between these by analyzing:
1. Fitness plateau patterns (stuck at local optimum)
2. Systematic bias in residuals (consistent errors suggest missing factor)
3. Unexplained variance (high irreducible error)
"""

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpistemicCrisis:
    """
    Represents detection of a fundamental model limitation.

    An epistemic crisis signals that the current ontology (variable space)
    is insufficient to solve the problem - optimization alone won't help.

    Attributes:
        id: Unique identifier for this crisis
        crisis_type: Type of crisis detected
            - "plateau": Fitness has stopped improving despite continued evolution
            - "systematic_bias": Residuals show consistent patterns (missing explanatory variable)
            - "unexplained_variance": High variance that optimization cannot reduce
        confidence: Confidence in this crisis detection (0-1)
        evidence: Supporting data for the diagnosis
        suggested_probes: List of probe types to try ("state", "gradient", "coverage", "numerical")
        timestamp: When this crisis was detected
        metadata: Additional crisis-specific data
    """

    id: str
    crisis_type: str
    confidence: float
    evidence: Dict[str, Any]
    suggested_probes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpistemicCrisis":
        """Deserialize from dictionary"""
        return cls(**data)

    def get_severity(self) -> str:
        """Get severity level based on confidence"""
        if self.confidence >= 0.9:
            return "critical"
        elif self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        else:
            return "low"

    def to_prompt_context(self) -> str:
        """Format crisis for inclusion in LLM prompts"""
        lines = [
            f"## Epistemic Crisis Detected",
            f"",
            f"**Type**: {self.crisis_type}",
            f"**Confidence**: {self.confidence:.2f} ({self.get_severity()})",
            f"**Suggested Probes**: {', '.join(self.suggested_probes)}",
            f"",
            f"### Evidence:",
        ]

        for key, value in self.evidence.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)


@dataclass
class CrisisDetectorConfig:
    """Configuration for crisis detection"""

    # Plateau detection
    min_plateau_iterations: int = 50  # Minimum iterations before declaring plateau
    fitness_improvement_threshold: float = 0.001  # Below this = no improvement
    variance_window: int = 20  # Window size for variance calculation

    # Systematic bias detection
    systematic_error_threshold: float = 0.1  # Threshold for systematic residuals
    min_residual_samples: int = 10  # Minimum samples to detect bias

    # Unexplained variance detection
    min_variance_samples: int = 20  # Minimum samples for variance analysis
    variance_reduction_threshold: float = 0.1  # Expected variance reduction per window

    # General settings
    confidence_threshold: float = 0.7  # Min confidence to trigger crisis
    cooldown_iterations: int = 30  # Iterations to wait after a crisis before detecting another


class CrisisDetector:
    """
    Detects epistemic crises indicating missing variables in the ontology.

    The detector analyzes the history of fitness values and evaluation artifacts
    to determine when optimization is stuck due to model limitations rather than
    insufficient exploration.

    Usage:
        detector = CrisisDetector(CrisisDetectorConfig())

        # Record each evaluation
        detector.record_evaluation(
            iteration=42,
            metrics={"combined_score": 0.75},
            artifacts={"residuals": {...}}
        )

        # Check for crisis
        crisis = detector.detect_crisis()
        if crisis:
            print(f"Crisis detected: {crisis.crisis_type}")
    """

    def __init__(self, config: CrisisDetectorConfig):
        self.config = config

        # History tracking
        self.fitness_history: List[Tuple[int, float]] = []  # (iteration, fitness)
        self.residual_history: List[Dict[str, float]] = []  # Residuals from artifacts
        self.artifact_history: List[Dict[str, Any]] = []  # Raw artifacts

        # Crisis tracking
        self.crisis_history: List[EpistemicCrisis] = []
        self.last_crisis_iteration: int = -999999  # Track cooldown

        logger.info("Initialized CrisisDetector")

    def record_evaluation(
        self,
        iteration: int,
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an evaluation result for analysis.

        Args:
            iteration: Current iteration number
            metrics: Evaluation metrics (must include fitness metric)
            artifacts: Optional evaluation artifacts (may include residuals)
        """
        # Extract fitness
        fitness = metrics.get("combined_score", metrics.get("fitness", 0.0))
        self.fitness_history.append((iteration, fitness))

        # Extract residuals if available
        if artifacts:
            self.artifact_history.append(artifacts)

            residuals = artifacts.get("residuals")
            if residuals and isinstance(residuals, dict):
                self.residual_history.append(residuals)

        # Limit history size to prevent memory issues
        max_history = self.config.min_plateau_iterations * 3
        if len(self.fitness_history) > max_history:
            self.fitness_history = self.fitness_history[-max_history:]
        if len(self.residual_history) > max_history:
            self.residual_history = self.residual_history[-max_history:]
        if len(self.artifact_history) > max_history:
            self.artifact_history = self.artifact_history[-max_history:]

    def detect_crisis(self) -> Optional[EpistemicCrisis]:
        """
        Analyze history for crisis patterns.

        Returns:
            EpistemicCrisis if crisis detected, None otherwise
        """
        if not self.fitness_history:
            return None

        current_iteration = self.fitness_history[-1][0]

        # Check cooldown
        if current_iteration - self.last_crisis_iteration < self.config.cooldown_iterations:
            return None

        # Check for each crisis type
        plateau = self._detect_plateau()
        if plateau and plateau.get("confidence", 0) >= self.config.confidence_threshold:
            crisis = EpistemicCrisis(
                id=f"crisis_plateau_{uuid.uuid4().hex[:8]}",
                crisis_type="plateau",
                confidence=plateau["confidence"],
                evidence=plateau,
                suggested_probes=["state", "gradient"],
                metadata={"detection_iteration": current_iteration},
            )
            self._record_crisis(crisis, current_iteration)
            return crisis

        bias = self._detect_systematic_bias()
        if bias and bias.get("confidence", 0) >= self.config.confidence_threshold:
            crisis = EpistemicCrisis(
                id=f"crisis_bias_{uuid.uuid4().hex[:8]}",
                crisis_type="systematic_bias",
                confidence=bias["confidence"],
                evidence=bias,
                suggested_probes=["coverage", "numerical"],
                metadata={"detection_iteration": current_iteration},
            )
            self._record_crisis(crisis, current_iteration)
            return crisis

        variance = self._detect_unexplained_variance()
        if variance and variance.get("confidence", 0) >= self.config.confidence_threshold:
            crisis = EpistemicCrisis(
                id=f"crisis_variance_{uuid.uuid4().hex[:8]}",
                crisis_type="unexplained_variance",
                confidence=variance["confidence"],
                evidence=variance,
                suggested_probes=["state", "numerical"],
                metadata={"detection_iteration": current_iteration},
            )
            self._record_crisis(crisis, current_iteration)
            return crisis

        return None

    def _detect_plateau(self) -> Optional[Dict[str, Any]]:
        """
        Detect fitness plateau (stuck at local optimum).

        A plateau is detected when:
        1. We have enough history (min_plateau_iterations)
        2. Recent fitness improvement is below threshold
        3. Variance is low (truly stuck, not just noisy)

        Returns:
            Evidence dict with confidence, or None
        """
        if len(self.fitness_history) < self.config.min_plateau_iterations:
            return None

        window = self.config.variance_window

        # Need at least 2 windows of data
        if len(self.fitness_history) < 2 * window:
            return None

        # Get recent and older fitness values
        recent = [f for _, f in self.fitness_history[-window:]]
        older = [f for _, f in self.fitness_history[-2 * window : -window]]

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        improvement = recent_mean - older_mean

        # Check if improvement is below threshold
        if abs(improvement) >= self.config.fitness_improvement_threshold:
            return None

        # Calculate variance - low variance = true plateau
        recent_var = np.var(recent)
        recent_std = np.std(recent)

        # Confidence calculation:
        # - Lower variance = higher confidence (truly stuck)
        # - Smaller improvement = higher confidence
        # - More iterations at plateau = higher confidence
        variance_factor = 1.0 - min(recent_var * 100, 0.5)  # 0.5 to 1.0
        improvement_factor = 1.0 - min(abs(improvement) / self.config.fitness_improvement_threshold, 0.5)

        # Check how long we've been at plateau
        plateau_iterations = self._count_plateau_iterations()
        duration_factor = min(plateau_iterations / self.config.min_plateau_iterations, 1.0)

        confidence = (variance_factor * 0.4 + improvement_factor * 0.3 + duration_factor * 0.3)

        return {
            "improvement": improvement,
            "recent_mean": recent_mean,
            "older_mean": older_mean,
            "recent_variance": recent_var,
            "recent_std": recent_std,
            "plateau_iterations": plateau_iterations,
            "confidence": confidence,
        }

    def _count_plateau_iterations(self) -> int:
        """Count how many iterations we've been at plateau"""
        if len(self.fitness_history) < 2:
            return 0

        threshold = self.config.fitness_improvement_threshold
        count = 0

        # Go backwards through history
        for i in range(len(self.fitness_history) - 1, 0, -1):
            current = self.fitness_history[i][1]
            previous = self.fitness_history[i - 1][1]

            if abs(current - previous) < threshold:
                count += 1
            else:
                break

        return count

    def _detect_systematic_bias(self) -> Optional[Dict[str, Any]]:
        """
        Detect systematic bias in residuals.

        Systematic bias indicates the model is consistently wrong in the same way,
        suggesting a missing explanatory variable.

        For this to work, the evaluator must return residuals in artifacts:
        {
            "residuals": {"input_1": 0.05, "input_2": -0.12, ...}
        }

        Returns:
            Evidence dict with confidence, or None
        """
        if len(self.residual_history) < self.config.min_residual_samples:
            return None

        # Aggregate all residuals
        all_residuals = []
        for residual_dict in self.residual_history[-self.config.min_residual_samples:]:
            all_residuals.extend(residual_dict.values())

        if not all_residuals:
            return None

        residuals = np.array(all_residuals)

        # Check for systematic bias (mean significantly different from 0)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Calculate z-score of mean
        n = len(residuals)
        se = std_residual / np.sqrt(n) if n > 0 else 1
        z_score = abs(mean_residual) / se if se > 0 else 0

        # Check for consistent sign
        positive_ratio = np.mean(residuals > 0)
        sign_consistency = max(positive_ratio, 1 - positive_ratio)

        # Bias is significant if:
        # 1. Mean is far from 0 (high z-score)
        # 2. Residuals consistently have same sign
        if abs(mean_residual) < self.config.systematic_error_threshold:
            return None

        # Confidence based on z-score and sign consistency
        z_factor = min(z_score / 3, 1.0)  # z > 3 = high confidence
        sign_factor = (sign_consistency - 0.5) * 2  # 0.5 = random, 1.0 = all same sign

        confidence = z_factor * 0.6 + sign_factor * 0.4

        return {
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "z_score": z_score,
            "positive_ratio": positive_ratio,
            "sign_consistency": sign_consistency,
            "num_samples": n,
            "confidence": confidence,
        }

    def _detect_unexplained_variance(self) -> Optional[Dict[str, Any]]:
        """
        Detect high unexplained variance.

        If fitness variance stays high despite optimization, it suggests
        there's a hidden variable causing the fluctuation.

        Returns:
            Evidence dict with confidence, or None
        """
        if len(self.fitness_history) < self.config.min_variance_samples:
            return None

        window = self.config.variance_window

        # Need multiple windows to compare variance reduction
        if len(self.fitness_history) < 3 * window:
            return None

        # Calculate variance in different time periods
        early = [f for _, f in self.fitness_history[:window]]
        middle = [f for _, f in self.fitness_history[window : 2 * window]]
        recent = [f for _, f in self.fitness_history[-window:]]

        var_early = np.var(early)
        var_middle = np.var(middle)
        var_recent = np.var(recent)

        # Expected: variance should decrease over time as we optimize
        # If it doesn't, there might be unexplained factors

        # Check if variance has reduced
        expected_reduction = self.config.variance_reduction_threshold
        actual_reduction = (var_early - var_recent) / max(var_early, 0.001)

        if actual_reduction >= expected_reduction:
            # Variance is reducing as expected - no crisis
            return None

        # Variance not reducing - possible unexplained factor
        # Calculate confidence based on how little variance has reduced
        reduction_deficit = expected_reduction - actual_reduction
        confidence = min(reduction_deficit / expected_reduction, 1.0) * 0.8

        # Additional check: is variance high in absolute terms?
        mean_fitness = np.mean(recent)
        cv = np.sqrt(var_recent) / max(mean_fitness, 0.001)  # Coefficient of variation

        if cv > 0.1:  # >10% variation relative to mean
            confidence += 0.2

        confidence = min(confidence, 1.0)

        return {
            "variance_early": var_early,
            "variance_middle": var_middle,
            "variance_recent": var_recent,
            "expected_reduction": expected_reduction,
            "actual_reduction": actual_reduction,
            "coefficient_of_variation": cv,
            "confidence": confidence,
        }

    def _record_crisis(self, crisis: EpistemicCrisis, iteration: int) -> None:
        """Record a detected crisis"""
        self.crisis_history.append(crisis)
        self.last_crisis_iteration = iteration

        logger.warning(
            f"EPISTEMIC CRISIS DETECTED: {crisis.crisis_type} "
            f"(confidence: {crisis.confidence:.2f}, severity: {crisis.get_severity()})"
        )

    def reset(self) -> None:
        """
        Reset the detector state.

        Called after an ontology expansion to start fresh analysis
        with the new variable space.
        """
        self.fitness_history = []
        self.residual_history = []
        self.artifact_history = []
        self.last_crisis_iteration = -999999

        logger.info("CrisisDetector reset - starting fresh analysis")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about crisis detection"""
        if not self.fitness_history:
            return {
                "total_evaluations": 0,
                "total_crises": len(self.crisis_history),
                "current_fitness": None,
            }

        recent_fitness = [f for _, f in self.fitness_history[-10:]] if self.fitness_history else []

        return {
            "total_evaluations": len(self.fitness_history),
            "total_crises": len(self.crisis_history),
            "crises_by_type": self._count_crises_by_type(),
            "current_fitness": self.fitness_history[-1][1] if self.fitness_history else None,
            "recent_mean_fitness": np.mean(recent_fitness) if recent_fitness else None,
            "recent_fitness_std": np.std(recent_fitness) if recent_fitness else None,
            "iterations_since_last_crisis": (
                self.fitness_history[-1][0] - self.last_crisis_iteration
                if self.fitness_history else 0
            ),
        }

    def _count_crises_by_type(self) -> Dict[str, int]:
        """Count crises by type"""
        counts = {}
        for crisis in self.crisis_history:
            counts[crisis.crisis_type] = counts.get(crisis.crisis_type, 0) + 1
        return counts
