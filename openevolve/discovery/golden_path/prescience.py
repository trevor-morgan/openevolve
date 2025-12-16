"""
Prescience - The ability to see when evolution has truly hit a wall.

"I must not fear. Fear is the mind-killer." - Bene Gesserit Litany

Unlike simple plateau detection, Prescience understands the NATURE of stagnation:
- Is this a local optimum? (fitness gradient is zero in all directions)
- Is this a representation limit? (programs are diverse but scores aren't)
- Is this a complexity barrier? (simple solutions exhausted)
- Is this an ontology gap? (hidden variables needed)

Prescience doesn't just detect problems - it diagnoses them.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CrisisType(Enum):
    """Types of evolutionary crises requiring different interventions."""

    # Evolution is making progress - no crisis
    NONE = "none"

    # Local optimum - fitness flat, programs similar
    LOCAL_OPTIMUM = "local_optimum"

    # Representation limit - programs diverse but scores clustered
    REPRESENTATION_LIMIT = "representation_limit"

    # Complexity barrier - simple solutions exhausted, complex ones fail
    COMPLEXITY_BARRIER = "complexity_barrier"

    # Ontology gap - success patterns exist but aren't captured by metrics
    ONTOLOGY_GAP = "ontology_gap"

    # Catastrophic - everything is failing
    CATASTROPHIC = "catastrophic"


@dataclass
class PrescienceReading:
    """A reading from the Prescience module - what it sees about the future."""

    crisis_type: CrisisType
    confidence: float  # 0-1, how certain are we?

    # Diagnostic details
    fitness_gradient: float  # Rate of improvement
    fitness_variance: float  # Spread of recent fitness values
    program_diversity: float  # How different are the programs?
    score_clustering: float  # Are scores bunched together?

    # Pattern indicators
    success_pattern_strength: float  # Do successful programs share hidden patterns?
    ontology_coverage: float  # What fraction of variance is explained by current metrics?

    # Recommendations
    recommended_action: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrescienceConfig:
    """Configuration for the Prescience module."""

    # Window sizes for analysis
    short_window: int = 10  # Recent history
    medium_window: int = 30  # Medium-term trends
    long_window: int = 100  # Long-term patterns

    # Thresholds for crisis detection
    gradient_threshold: float = 0.001  # Below this = stagnant
    variance_threshold: float = 0.0001  # Below this = clustered
    diversity_threshold: float = 0.3  # Below this = converged

    # Confidence thresholds
    min_samples_for_diagnosis: int = 20
    crisis_confidence_threshold: float = 0.7

    # Cooldown to prevent thrashing
    min_iterations_between_readings: int = 5


class Prescience:
    """
    The Prescience module - sees the nature of evolutionary stagnation.

    "The mystery of life isn't a problem to solve, but a reality to experience."
    But we can still diagnose when that experience has stalled.
    """

    def __init__(self, config: PrescienceConfig | None = None):
        self.config = config or PrescienceConfig()

        # History buffers
        self.fitness_history: list[float] = []
        self.program_history: list[dict[str, Any]] = []  # Program metadata
        self.metrics_history: list[dict[str, float]] = []

        # State
        self.last_reading_iteration: int = 0
        self.readings_history: list[PrescienceReading] = []

    def record_iteration(
        self,
        iteration: int,
        fitness: float,
        metrics: dict[str, float],
        program_code: str | None = None,
        program_id: str | None = None,
    ) -> None:
        """Record data from an iteration for future analysis."""
        self.fitness_history.append(fitness)
        self.metrics_history.append(metrics)

        if program_code or program_id:
            self.program_history.append(
                {
                    "iteration": iteration,
                    "fitness": fitness,
                    "program_id": program_id,
                    "code_hash": hash(program_code) if program_code else None,
                    "code_length": len(program_code) if program_code else 0,
                    "metrics": metrics,
                }
            )

    def take_reading(self, current_iteration: int) -> PrescienceReading:
        """
        Take a prescient reading of the evolutionary state.

        Returns a diagnosis of what's happening and recommendations.
        """
        # Check cooldown
        if (
            current_iteration - self.last_reading_iteration
            < self.config.min_iterations_between_readings
        ):
            return PrescienceReading(
                crisis_type=CrisisType.NONE,
                confidence=0.0,
                fitness_gradient=0.0,
                fitness_variance=0.0,
                program_diversity=0.0,
                score_clustering=0.0,
                success_pattern_strength=0.0,
                ontology_coverage=0.0,
                recommended_action="cooldown_active",
            )

        self.last_reading_iteration = current_iteration

        # Need enough data to diagnose
        if len(self.fitness_history) < self.config.min_samples_for_diagnosis:
            return PrescienceReading(
                crisis_type=CrisisType.NONE,
                confidence=0.0,
                fitness_gradient=0.0,
                fitness_variance=0.0,
                program_diversity=0.0,
                score_clustering=0.0,
                success_pattern_strength=0.0,
                ontology_coverage=0.0,
                recommended_action="insufficient_data",
            )

        # Compute diagnostics
        gradient = self._compute_fitness_gradient()
        variance = self._compute_fitness_variance()
        diversity = self._compute_program_diversity()
        clustering = self._compute_score_clustering()
        pattern_strength = self._compute_success_pattern_strength()
        ontology_coverage = self._compute_ontology_coverage()

        # Diagnose crisis type
        crisis_type, confidence = self._diagnose_crisis(
            gradient, variance, diversity, clustering, pattern_strength, ontology_coverage
        )

        # Generate recommendation
        action, details = self._generate_recommendation(crisis_type, confidence)

        reading = PrescienceReading(
            crisis_type=crisis_type,
            confidence=confidence,
            fitness_gradient=gradient,
            fitness_variance=variance,
            program_diversity=diversity,
            score_clustering=clustering,
            success_pattern_strength=pattern_strength,
            ontology_coverage=ontology_coverage,
            recommended_action=action,
            details=details,
        )

        self.readings_history.append(reading)

        if crisis_type != CrisisType.NONE:
            logger.info(
                f"Prescience detected {crisis_type.value} crisis (confidence: {confidence:.2f})"
            )
            logger.info(f"Recommendation: {action}")

        return reading

    def _compute_fitness_gradient(self) -> float:
        """Compute the rate of fitness improvement."""
        window = min(self.config.short_window, len(self.fitness_history))
        if window < 3:
            return 0.0

        recent = self.fitness_history[-window:]

        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        return float(slope)

    def _compute_fitness_variance(self) -> float:
        """Compute variance in recent fitness values."""
        window = min(self.config.short_window, len(self.fitness_history))
        if window < 2:
            return 0.0

        recent = self.fitness_history[-window:]
        return float(np.var(recent))

    def _compute_program_diversity(self) -> float:
        """
        Compute diversity of recent programs.

        Uses code length variance and hash uniqueness as proxies.
        Real implementation would use semantic similarity.
        """
        window = min(self.config.medium_window, len(self.program_history))
        if window < 2:
            return 1.0  # Assume diverse if no data

        recent = self.program_history[-window:]

        # Hash uniqueness
        hashes = [p.get("code_hash") for p in recent if p.get("code_hash")]
        if hashes:
            unique_ratio = len(set(hashes)) / len(hashes)
        else:
            unique_ratio = 1.0

        # Length variance
        lengths = [p.get("code_length", 0) for p in recent]
        if lengths and np.mean(lengths) > 0:
            length_cv = np.std(lengths) / np.mean(lengths)
        else:
            length_cv = 0.0

        # Combine metrics
        diversity = 0.7 * unique_ratio + 0.3 * min(length_cv, 1.0)

        return float(diversity)

    def _compute_score_clustering(self) -> float:
        """
        Compute how clustered the fitness scores are.

        High clustering + high diversity = representation limit.
        """
        window = min(self.config.medium_window, len(self.fitness_history))
        if window < 5:
            return 0.0

        recent = self.fitness_history[-window:]

        # Coefficient of variation (inverted - high clustering = low CV)
        mean = np.mean(recent)
        if mean > 0:
            cv = np.std(recent) / mean
            clustering = max(0, 1 - cv * 10)  # Scale to 0-1
        else:
            clustering = 1.0

        return float(clustering)

    def _compute_success_pattern_strength(self) -> float:
        """
        Compute whether successful programs share hidden patterns.

        This is the key indicator for ontology gaps - if high-fitness
        programs share patterns not captured by metrics.
        """
        window = min(self.config.long_window, len(self.program_history))
        if window < 10:
            return 0.0

        recent = self.program_history[-window:]

        # Split into successful and unsuccessful
        fitnesses = [p["fitness"] for p in recent]
        median_fitness = np.median(fitnesses)

        successful = [p for p in recent if p["fitness"] > median_fitness]
        unsuccessful = [p for p in recent if p["fitness"] <= median_fitness]

        if not successful or not unsuccessful:
            return 0.0

        # Compare code length distributions
        success_lengths = [p.get("code_length", 0) for p in successful]
        fail_lengths = [p.get("code_length", 0) for p in unsuccessful]

        # If successful programs cluster in length but unsuccessful don't,
        # there's a hidden pattern
        success_length_var = np.var(success_lengths) if success_lengths else 0
        fail_length_var = np.var(fail_lengths) if fail_lengths else 0

        if fail_length_var > 0:
            pattern_ratio = 1 - (success_length_var / fail_length_var)
            pattern_strength = max(0, min(1, pattern_ratio))
        else:
            pattern_strength = 0.5

        return float(pattern_strength)

    def _compute_ontology_coverage(self) -> float:
        """
        Compute what fraction of fitness variance is explained by current metrics.

        Low coverage = ontology gap (hidden variables exist).
        """
        window = min(self.config.long_window, len(self.metrics_history))
        if window < 10:
            return 1.0  # Assume covered if no data

        recent_metrics = self.metrics_history[-window:]
        recent_fitness = self.fitness_history[-window:]

        if not recent_metrics or not recent_fitness:
            return 1.0

        # Get all metric names
        all_keys = set()
        for m in recent_metrics:
            all_keys.update(m.keys())

        # Build feature matrix
        X = []
        for m in recent_metrics:
            row = [m.get(k, 0.0) for k in sorted(all_keys)]
            X.append(row)

        X = np.array(X)
        y = np.array(recent_fitness)

        # Compute R² using simple linear regression
        try:
            if X.shape[1] > 0 and len(y) > X.shape[1]:
                # Add intercept
                X_with_intercept = np.column_stack([np.ones(len(X)), X])

                # Solve least squares
                coeffs, _residuals, _rank, _s = np.linalg.lstsq(X_with_intercept, y, rcond=None)

                # Compute R²
                y_pred = X_with_intercept @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                if ss_tot > 0:
                    r_squared = 1 - (ss_res / ss_tot)
                    return float(max(0, min(1, r_squared)))
        except Exception:
            pass

        return 0.5  # Default assumption

    def _diagnose_crisis(
        self,
        gradient: float,
        variance: float,
        diversity: float,
        clustering: float,
        pattern_strength: float,
        ontology_coverage: float,
    ) -> tuple[CrisisType, float]:
        """
        Diagnose the type of crisis based on all indicators.

        Returns (crisis_type, confidence).
        """
        # No crisis if making progress
        if gradient > self.config.gradient_threshold:
            return CrisisType.NONE, 0.0

        # Catastrophic if everything is failing
        recent_mean = np.mean(self.fitness_history[-10:]) if len(self.fitness_history) >= 10 else 0
        if recent_mean < 0.1:
            return CrisisType.CATASTROPHIC, 0.9

        # Now diagnose the type of stagnation
        scores = {
            CrisisType.LOCAL_OPTIMUM: 0.0,
            CrisisType.REPRESENTATION_LIMIT: 0.0,
            CrisisType.COMPLEXITY_BARRIER: 0.0,
            CrisisType.ONTOLOGY_GAP: 0.0,
        }

        # Local optimum: low diversity, low variance
        if diversity < self.config.diversity_threshold:
            scores[CrisisType.LOCAL_OPTIMUM] += 0.5
        if variance < self.config.variance_threshold:
            scores[CrisisType.LOCAL_OPTIMUM] += 0.3

        # Representation limit: high diversity but clustered scores
        if diversity > 0.5 and clustering > 0.7:
            scores[CrisisType.REPRESENTATION_LIMIT] += 0.6

        # Complexity barrier: recent programs getting longer but not better
        if len(self.program_history) >= 20:
            recent_lengths = [p.get("code_length", 0) for p in self.program_history[-20:]]
            length_trend = np.polyfit(np.arange(len(recent_lengths)), recent_lengths, 1)[0]
            if length_trend > 0 and gradient <= 0:
                scores[CrisisType.COMPLEXITY_BARRIER] += 0.5

        # Ontology gap: strong success patterns but low coverage
        if pattern_strength > 0.5 and ontology_coverage < 0.7:
            scores[CrisisType.ONTOLOGY_GAP] += 0.7
        if ontology_coverage < 0.5:
            scores[CrisisType.ONTOLOGY_GAP] += 0.3

        # Find highest scoring crisis
        best_crisis = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_crisis]

        if best_score < self.config.crisis_confidence_threshold:
            return CrisisType.NONE, best_score

        return best_crisis, min(1.0, best_score)

    def _generate_recommendation(
        self,
        crisis_type: CrisisType,
        confidence: float,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a recommendation based on the crisis type."""

        recommendations = {
            CrisisType.NONE: ("continue_evolution", {"reason": "Evolution is making progress"}),
            CrisisType.LOCAL_OPTIMUM: (
                "increase_exploration",
                {
                    "reason": "Stuck in local optimum",
                    "suggestions": [
                        "Increase mutation rate",
                        "Add random restarts",
                        "Try island migration",
                    ],
                },
            ),
            CrisisType.REPRESENTATION_LIMIT: (
                "expand_representation",
                {
                    "reason": "Current representation cannot express better solutions",
                    "suggestions": [
                        "Add new code constructs",
                        "Expand allowed operations",
                        "Try different parameterizations",
                    ],
                },
            ),
            CrisisType.COMPLEXITY_BARRIER: (
                "simplify_or_decompose",
                {
                    "reason": "Complex solutions failing, simple ones exhausted",
                    "suggestions": [
                        "Add intermediate fitness signals",
                        "Decompose problem into subproblems",
                        "Reduce solution complexity",
                    ],
                },
            ),
            CrisisType.ONTOLOGY_GAP: (
                "discover_hidden_variables",
                {
                    "reason": "Success patterns exist but aren't captured by metrics",
                    "suggestions": [
                        "Mine programs for hidden patterns (Mentat)",
                        "Propose new variables (SietchFinder)",
                        "Validate and integrate (GomJabbar + SpiceAgony)",
                    ],
                    "trigger_golden_path": True,
                },
            ),
            CrisisType.CATASTROPHIC: (
                "reset_and_diagnose",
                {
                    "reason": "Evolution has catastrophically failed",
                    "suggestions": [
                        "Check evaluator for bugs",
                        "Verify initial program works",
                        "Reset population with new seed",
                    ],
                },
            ),
        }

        return recommendations.get(crisis_type, ("unknown", {}))

    def should_trigger_golden_path(self) -> bool:
        """Check if we should trigger the full Golden Path discovery."""
        if not self.readings_history:
            return False

        recent = self.readings_history[-1]
        return (
            recent.crisis_type == CrisisType.ONTOLOGY_GAP
            and recent.confidence >= self.config.crisis_confidence_threshold
            and recent.details.get("trigger_golden_path", False)
        )
