"""
Causal Discovery Tool - Find causal relationships, not just correlations.

"Correlation is not causation, but it sure is a hint."

Uses causal discovery algorithms to find:
- What code features CAUSE better fitness?
- What interventions would improve performance?
- Hidden confounders affecting both code and fitness

This goes beyond correlation to find actionable insights.
"""

import itertools
import logging
import re
from typing import Any, ClassVar

import numpy as np

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class CausalDiscoveryTool(DiscoveryTool):
    """
    Discovers causal relationships using causal inference methods.

    Primary: DoWhy + causal-learn
    Fallback: Simple causal heuristics with numpy
    """

    name = "causal_discovery"
    description = (
        "Find causal relationships (not just correlations) between code features and fitness"
    )
    discovery_types: ClassVar[list[DiscoveryType]] = [
        DiscoveryType.CAUSAL_RELATIONSHIP,
        DiscoveryType.HYPOTHESIS,
    ]
    dependencies: ClassVar[list[str]] = []  # Check dynamically

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._backend: str | None = None

        def _check_dependencies(self) -> bool:
            """Check for causal discovery libraries."""

            import importlib.util

            # Try DoWhy + causal-learn

            if importlib.util.find_spec("dowhy") and importlib.util.find_spec("causallearn"):
                self._backend = "dowhy"

                return True

            # Try just causal-learn

            if importlib.util.find_spec("causallearn"):
                self._backend = "causallearn"

                return True

            # Fallback to numpy-based heuristics

            self._backend = "numpy"

            return True

    async def discover(self, context: ToolContext) -> list[Discovery]:
        """Run causal discovery."""
        logger.info(f"Running causal discovery ({self._backend})...")

        programs = context.programs
        if len(programs) < 20:
            logger.warning("Not enough programs for causal discovery")
            return []

        # Extract features
        features, feature_names = self._extract_features(programs)
        fitnesses = np.array([p.get("fitness", 0.0) for p in programs])

        if features.shape[1] == 0:
            return []

        discoveries = []

        if self._backend == "dowhy":
            discoveries = await self._run_dowhy(features, fitnesses, feature_names)
        elif self._backend == "causallearn":
            discoveries = await self._run_causallearn(features, fitnesses, feature_names)
        else:
            discoveries = await self._run_numpy_fallback(
                features, fitnesses, feature_names, programs
            )

        logger.info(f"Causal discovery found {len(discoveries)} relationships")
        return discoveries

    def _extract_features(self, programs: list[dict[str, Any]]) -> tuple:
        """Extract features for causal analysis."""
        feature_names = []
        feature_matrix = []

        for prog in programs:
            features = {}

            # From metrics
            for k, v in prog.get("metrics", {}).items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    features[f"m_{k}"] = v

            # From code
            code = prog.get("code", "")
            if code:
                features["n_coils"] = len(re.findall(r"add_coil\(", code))
                features["n_functions"] = len(re.findall(r"def \w+\(", code))
                features["code_len"] = len(code)

                # Extract numeric patterns
                numbers = [
                    float(x) for x in re.findall(r"[-+]?\d*\.?\d+", code) if abs(float(x)) < 1e10
                ]
                if numbers:
                    features["num_mean"] = np.mean(numbers)
                    features["num_std"] = np.std(numbers) if len(numbers) > 1 else 0

            feature_matrix.append(features)

        # Build matrix with common features
        if feature_matrix:
            all_keys = set()
            for f in feature_matrix:
                all_keys.update(f.keys())

            feature_names = sorted(all_keys)

            X = []
            for f in feature_matrix:
                row = [f.get(k, 0.0) for k in feature_names]
                X.append(row)

            X = np.array(X)

            # Remove constant/near-constant columns
            stds = np.std(X, axis=0)
            mask = stds > 1e-6
            X = X[:, mask]
            feature_names = [n for n, m in zip(feature_names, mask) if m]

            return X, feature_names

        return np.array([]), []

    async def _run_dowhy(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> list[Discovery]:
        """Run DoWhy causal inference."""
        import dowhy
        import pandas as pd

        discoveries = []

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["fitness"] = y

        for feature in feature_names[:10]:  # Limit features to test
            try:
                # Define causal model
                model = dowhy.CausalModel(
                    data=df,
                    treatment=feature,
                    outcome="fitness",
                    common_causes=[f for f in feature_names if f != feature][:5],
                )

                # Identify effect
                identified = model.identify_effect(proceed_when_unidentifiable=True)

                # Estimate effect
                estimate = model.estimate_effect(
                    identified, method_name="backdoor.linear_regression"
                )

                effect = estimate.value

                if abs(effect) > 0.01:  # Non-trivial effect
                    # Refute (robustness check)
                    refute = model.refute_estimate(
                        identified, estimate, method_name="random_common_cause"
                    )

                    # Check if estimate is robust
                    is_robust = abs(refute.new_effect - effect) < abs(effect) * 0.5

                    discoveries.append(
                        Discovery(
                            name=f"causal_{feature}_to_fitness",
                            description=f"Causal effect: {feature} -> fitness (effect={effect:.3f})",
                            discovery_type=DiscoveryType.CAUSAL_RELATIONSHIP,
                            content={
                                "treatment": feature,
                                "outcome": "fitness",
                                "effect": float(effect),
                                "robust": is_robust,
                                "method": "dowhy_backdoor",
                            },
                            confidence=0.7 if is_robust else 0.4,
                            evidence=[
                                f"Estimated effect: {effect:.3f}",
                                f"Robust to random common cause: {is_robust}",
                            ],
                            testable=True,
                            validation_criteria=f"Intervening on {feature} should change fitness by ~{effect:.3f}",
                        )
                    )

            except Exception as e:
                logger.debug(f"DoWhy failed for {feature}: {e}")

        return discoveries

    async def _run_causallearn(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> list[Discovery]:
        """Run causal-learn PC algorithm."""
        from causallearn.search.ConstraintBased.PC import pc

        discoveries = []

        # Combine features and outcome
        data = np.column_stack([X, y])

        try:
            # Run PC algorithm
            cg = pc(data, alpha=0.05, indep_test="fisherz")

            # Extract edges to fitness
            fitness_idx = len(feature_names)

            for i, name in enumerate(feature_names):
                # Check if there's an edge to fitness
                if cg.G.graph[i, fitness_idx] != 0 or cg.G.graph[fitness_idx, i] != 0:
                    # Determine direction
                    if cg.G.graph[i, fitness_idx] == -1 and cg.G.graph[fitness_idx, i] == 1:
                        direction = f"{name} -> fitness"
                    elif cg.G.graph[i, fitness_idx] == 1 and cg.G.graph[fitness_idx, i] == -1:
                        direction = f"fitness -> {name}"
                    else:
                        direction = f"{name} -- fitness (undirected)"

                    # Compute correlation for effect size
                    corr = np.corrcoef(X[:, i], y)[0, 1]

                    discoveries.append(
                        Discovery(
                            name=f"causal_pc_{name}",
                            description=f"Causal edge discovered: {direction}",
                            discovery_type=DiscoveryType.CAUSAL_RELATIONSHIP,
                            content={
                                "feature": name,
                                "direction": direction,
                                "correlation": float(corr),
                                "method": "pc_algorithm",
                            },
                            confidence=0.6,
                            evidence=[
                                f"PC algorithm discovered edge: {direction}",
                                f"Correlation: {corr:.3f}",
                            ],
                        )
                    )

        except Exception as e:
            logger.warning(f"PC algorithm failed: {e}")

        return discoveries

    async def _run_numpy_fallback(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str], programs: list[dict[str, Any]]
    ) -> list[Discovery]:
        """Fallback: Use heuristics to suggest causal relationships."""
        discoveries = []

        # Partial correlations (controlling for other variables)
        for i, name in enumerate(feature_names):
            x = X[:, i]

            # Full correlation
            corr_full = np.corrcoef(x, y)[0, 1]

            # Partial correlation controlling for other features
            other_indices = [j for j in range(X.shape[1]) if j != i]
            if other_indices:
                # Regress out other variables
                X_other = X[:, other_indices]
                X_other_aug = np.column_stack([np.ones(len(X_other)), X_other])

                try:
                    # Residualize x
                    coeffs_x, _, _, _ = np.linalg.lstsq(X_other_aug, x, rcond=None)
                    x_resid = x - X_other_aug @ coeffs_x

                    # Residualize y
                    coeffs_y, _, _, _ = np.linalg.lstsq(X_other_aug, y, rcond=None)
                    y_resid = y - X_other_aug @ coeffs_y

                    # Partial correlation
                    corr_partial = np.corrcoef(x_resid, y_resid)[0, 1]

                    if np.isfinite(corr_partial) and abs(corr_partial) > 0.2:
                        # This suggests a more direct relationship
                        discoveries.append(
                            Discovery(
                                name=f"causal_partial_{name}",
                                description=f"Partial correlation: {name} -> fitness (r={corr_partial:.3f})",
                                discovery_type=DiscoveryType.CAUSAL_RELATIONSHIP,
                                content={
                                    "feature": name,
                                    "full_correlation": float(corr_full),
                                    "partial_correlation": float(corr_partial),
                                    "controls": [feature_names[j] for j in other_indices],
                                    "method": "partial_correlation",
                                },
                                confidence=abs(corr_partial),
                                evidence=[
                                    f"Full correlation: {corr_full:.3f}",
                                    f"Partial correlation: {corr_partial:.3f}",
                                    f"Controlling for {len(other_indices)} other features",
                                ],
                                testable=True,
                            )
                        )
                except Exception:
                    continue

        # Natural experiment detection (look for quasi-random variation)
        for i, name in enumerate(feature_names):
            x = X[:, i]

            # Check if feature has discrete levels (natural experiment)
            unique_values = np.unique(x)
            if 2 <= len(unique_values) <= 5:
                # Compare fitness across levels
                level_means = []
                for val in unique_values:
                    mask = x == val
                    if mask.sum() > 3:
                        level_means.append((val, y[mask].mean(), mask.sum()))

                if len(level_means) >= 2:
                    # Check monotonic relationship
                    sorted_levels = sorted(level_means, key=lambda x: x[0])
                    fitness_values = [lm[1] for lm in sorted_levels]

                    is_monotonic = all(
                        a <= b for a, b in itertools.pairwise(fitness_values)
                    ) or all(a >= b for a, b in itertools.pairwise(fitness_values))

                    if is_monotonic:
                        direction = (
                            "increasing" if fitness_values[-1] > fitness_values[0] else "decreasing"
                        )
                        effect = fitness_values[-1] - fitness_values[0]

                        discoveries.append(
                            Discovery(
                                name=f"causal_levels_{name}",
                                description=f"Monotonic effect: {name} levels -> fitness ({direction})",
                                discovery_type=DiscoveryType.CAUSAL_RELATIONSHIP,
                                content={
                                    "feature": name,
                                    "levels": [
                                        (float(v), float(m), int(n)) for v, m, n in sorted_levels
                                    ],
                                    "direction": direction,
                                    "total_effect": float(effect),
                                    "method": "natural_experiment",
                                },
                                confidence=0.5,
                                evidence=[
                                    f"Effect: {effect:.3f}",
                                    f"Direction: {direction}",
                                    f"Levels: {len(unique_values)}",
                                ],
                            )
                        )

        return discoveries[:10]
