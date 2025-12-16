"""
GomJabbar - The test that separates true discoveries from illusions.

"I must not fear. Fear is the mind-killer...
I will face my fear. I will permit it to pass over me and through me."

The Gom Jabbar tests whether a hypothesized hidden variable is REAL:
1. Can we actually compute it from programs?
2. Does it correlate with fitness as predicted?
3. Does it provide information BEYOND existing metrics?
4. Is the correlation robust across different subsets?

Only variables that pass the Gom Jabbar are worthy of integration.
The test is painful but necessary.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of Gom Jabbar validation for a hidden variable."""

    variable_name: str
    passed: bool  # Did it pass the test?

    # Core metrics
    correlation: float  # Correlation with fitness
    p_value: float  # Statistical significance
    incremental_r2: float  # R² improvement over existing metrics

    # Robustness checks
    cross_validation_score: float  # Consistency across folds
    bootstrap_ci_lower: float  # 95% CI lower bound
    bootstrap_ci_upper: float  # 95% CI upper bound

    # Computation success
    computation_success_rate: float  # What fraction of programs could be computed?
    computation_errors: list[str] = field(default_factory=list)

    # Diagnostics
    failure_reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GomJabbarConfig:
    """Configuration for the Gom Jabbar validation."""

    # Statistical thresholds
    min_correlation: float = 0.15  # Minimum absolute correlation
    max_p_value: float = 0.05  # Maximum p-value for significance
    min_incremental_r2: float = 0.02  # Minimum improvement over existing

    # Robustness requirements
    min_cross_validation: float = 0.6  # Minimum CV consistency
    bootstrap_iterations: int = 100
    min_ci_bound: float = 0.05  # CI lower bound must exceed this

    # Computation requirements
    min_computation_success: float = 0.8  # At least 80% must compute

    # Cross-validation
    cv_folds: int = 5


class GomJabbar:
    """
    The Gom Jabbar - validates hidden variable discoveries.

    "What's in the box?"
    "Pain."

    Only true discoveries survive the test.
    """

    def __init__(self, config: GomJabbarConfig | None = None):
        self.config = config or GomJabbarConfig()
        self.validation_history: list[ValidationResult] = []

    def validate(
        self,
        variable: Any,  # HiddenVariable
        programs: list[dict[str, Any]],
        existing_metrics: list[str],
    ) -> ValidationResult:
        """
        Validate a hypothesized hidden variable.

        Args:
            variable: The HiddenVariable to test
            programs: Programs with 'code', 'fitness', 'metrics' keys
            existing_metrics: Names of currently tracked metrics

        Returns:
            ValidationResult with pass/fail and diagnostics
        """
        logger.info(f"Gom Jabbar testing: {variable.name}")

        failure_reasons = []
        details = {}

        # Step 1: Try to compute the variable for all programs
        computed_values, success_rate, errors = self._compute_variable(variable, programs)
        details["computation_success_rate"] = success_rate
        details["computation_errors"] = errors[:5]  # Keep first 5 errors

        if success_rate < self.config.min_computation_success:
            failure_reasons.append(f"Computation success rate too low: {success_rate:.2f}")

        if len(computed_values) < 10:
            return ValidationResult(
                variable_name=variable.name,
                passed=False,
                correlation=0.0,
                p_value=1.0,
                incremental_r2=0.0,
                cross_validation_score=0.0,
                bootstrap_ci_lower=0.0,
                bootstrap_ci_upper=0.0,
                computation_success_rate=success_rate,
                computation_errors=errors,
                failure_reasons=["Insufficient computable programs"],
                details=details,
            )

        # Step 2: Compute correlation with fitness
        fitnesses = [p["fitness"] for p in programs if p.get("_computed_value") is not None]

        # Filter to only programs where we could compute
        valid_programs = [p for p in programs if p.get("_computed_value") is not None]
        values = [p["_computed_value"] for p in valid_programs]
        fitnesses = [p["fitness"] for p in valid_programs]

        correlation, p_value = self._compute_correlation(values, fitnesses)
        details["raw_correlation"] = correlation
        details["raw_p_value"] = p_value

        if abs(correlation) < self.config.min_correlation:
            failure_reasons.append(f"Correlation too weak: {correlation:.3f}")

        if p_value > self.config.max_p_value:
            failure_reasons.append(f"Not statistically significant: p={p_value:.4f}")

        # Step 3: Compute incremental R² over existing metrics
        incremental_r2 = self._compute_incremental_r2(
            values, fitnesses, valid_programs, existing_metrics
        )
        details["incremental_r2"] = incremental_r2

        if incremental_r2 < self.config.min_incremental_r2:
            failure_reasons.append(f"Insufficient incremental R²: {incremental_r2:.4f}")

        # Step 4: Cross-validation robustness
        cv_score = self._cross_validation(values, fitnesses)
        details["cv_score"] = cv_score

        if cv_score < self.config.min_cross_validation:
            failure_reasons.append(f"Poor cross-validation: {cv_score:.3f}")

        # Step 5: Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(values, fitnesses)
        details["bootstrap_ci"] = (ci_lower, ci_upper)

        if abs(ci_lower) < self.config.min_ci_bound and abs(ci_upper) < self.config.min_ci_bound:
            failure_reasons.append(f"Bootstrap CI includes zero: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Determine pass/fail
        passed = len(failure_reasons) == 0

        result = ValidationResult(
            variable_name=variable.name,
            passed=passed,
            correlation=correlation,
            p_value=p_value,
            incremental_r2=incremental_r2,
            cross_validation_score=cv_score,
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper,
            computation_success_rate=success_rate,
            computation_errors=errors,
            failure_reasons=failure_reasons,
            details=details,
        )

        self.validation_history.append(result)

        if passed:
            logger.info(
                f"✓ {variable.name} PASSED Gom Jabbar (corr={correlation:.3f}, incr_r2={incremental_r2:.4f})"
            )
        else:
            logger.info(f"✗ {variable.name} FAILED Gom Jabbar: {', '.join(failure_reasons)}")

        # Clean up temporary data
        for p in programs:
            p.pop("_computed_value", None)

        return result

    def _compute_variable(
        self,
        variable: Any,
        programs: list[dict[str, Any]],
    ) -> tuple[list[float], float, list[str]]:
        """
        Compute the variable value for all programs.

        Returns (values, success_rate, errors).
        """
        values = []
        errors = []
        successes = 0

        # Compile the computation code
        try:
            # Create a namespace for the computation
            namespace = {
                "np": __import__("numpy"),
                "re": __import__("re"),
                "ast": __import__("ast"),
                "math": __import__("math"),
            }

            # Execute the computation code to define the function
            exec(variable.computation_code, namespace)

            # Find the compute function
            compute_func = None
            for name, obj in namespace.items():
                if callable(obj) and name.startswith("compute"):
                    compute_func = obj
                    break

            if compute_func is None:
                return [], 0.0, ["Could not find compute function in code"]

        except Exception as e:
            return [], 0.0, [f"Failed to compile computation code: {e}"]

        # Compute for each program
        for prog in programs:
            try:
                value = compute_func(prog.get("code", ""), prog.get("metrics", {}))
                if value is not None and not np.isnan(value) and not np.isinf(value):
                    values.append(float(value))
                    prog["_computed_value"] = float(value)
                    successes += 1
                else:
                    prog["_computed_value"] = None
                    errors.append(f"Invalid value: {value}")
            except Exception as e:
                prog["_computed_value"] = None
                errors.append(str(e)[:100])

        success_rate = successes / len(programs) if programs else 0.0
        return values, success_rate, errors

    def _compute_correlation(
        self,
        values: list[float],
        fitnesses: list[float],
    ) -> tuple[float, float]:
        """Compute Pearson correlation and p-value."""
        try:
            if len(values) < 3:
                return 0.0, 1.0

            correlation, p_value = stats.pearsonr(values, fitnesses)

            if np.isnan(correlation):
                return 0.0, 1.0

            return float(correlation), float(p_value)
        except Exception:
            return 0.0, 1.0

    def _compute_incremental_r2(
        self,
        new_values: list[float],
        fitnesses: list[float],
        programs: list[dict[str, Any]],
        existing_metrics: list[str],
    ) -> float:
        """
        Compute incremental R² - how much variance is explained
        by the new variable beyond existing metrics.
        """
        try:
            if len(new_values) < 10:
                return 0.0

            # Build feature matrix from existing metrics
            X_existing = []
            for prog in programs:
                metrics = prog.get("metrics", {})
                row = [metrics.get(m, 0.0) for m in existing_metrics]
                X_existing.append(row)

            X_existing = np.array(X_existing)
            y = np.array(fitnesses)

            # R² with existing metrics only
            if X_existing.shape[1] > 0:
                try:
                    X_with_intercept = np.column_stack([np.ones(len(X_existing)), X_existing])
                    coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
                    y_pred = X_with_intercept @ coeffs
                    ss_res_existing = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2_existing = 1 - ss_res_existing / ss_tot if ss_tot > 0 else 0
                except:
                    r2_existing = 0
            else:
                r2_existing = 0

            # R² with existing metrics + new variable
            new_values_arr = np.array(new_values).reshape(-1, 1)
            X_combined = (
                np.column_stack([X_existing, new_values_arr])
                if X_existing.shape[1] > 0
                else new_values_arr
            )

            try:
                X_with_intercept = np.column_stack([np.ones(len(X_combined)), X_combined])
                coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
                y_pred = X_with_intercept @ coeffs
                ss_res_combined = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_combined = 1 - ss_res_combined / ss_tot if ss_tot > 0 else 0
            except:
                r2_combined = r2_existing

            incremental = r2_combined - r2_existing
            return float(max(0, incremental))

        except Exception as e:
            logger.debug(f"Incremental R² calculation failed: {e}")
            return 0.0

    def _cross_validation(
        self,
        values: list[float],
        fitnesses: list[float],
    ) -> float:
        """
        Cross-validation to check robustness of correlation.

        Returns average correlation across folds.
        """
        try:
            n = len(values)
            if n < 10:
                return 0.0

            values = np.array(values)
            fitnesses = np.array(fitnesses)

            fold_size = n // self.config.cv_folds
            correlations = []

            for i in range(self.config.cv_folds):
                # Create test fold
                start = i * fold_size
                end = start + fold_size if i < self.config.cv_folds - 1 else n

                # Exclude test fold
                mask = np.ones(n, dtype=bool)
                mask[start:end] = False

                train_values = values[mask]
                train_fitnesses = fitnesses[mask]

                if len(train_values) >= 5:
                    corr, _ = stats.pearsonr(train_values, train_fitnesses)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                # Return consistency: mean correlation / std
                mean_corr = np.mean(correlations)
                std_corr = np.std(correlations) + 1e-10
                consistency = mean_corr / (std_corr + mean_corr)
                return float(consistency)

            return 0.0

        except Exception as e:
            logger.debug(f"Cross-validation failed: {e}")
            return 0.0

    def _bootstrap_ci(
        self,
        values: list[float],
        fitnesses: list[float],
    ) -> tuple[float, float]:
        """
        Bootstrap confidence interval for the correlation.

        Returns (lower_95, upper_95).
        """
        try:
            n = len(values)
            if n < 10:
                return 0.0, 0.0

            values = np.array(values)
            fitnesses = np.array(fitnesses)

            bootstrap_correlations = []

            for _ in range(self.config.bootstrap_iterations):
                # Sample with replacement
                indices = np.random.choice(n, size=n, replace=True)
                boot_values = values[indices]
                boot_fitnesses = fitnesses[indices]

                corr, _ = stats.pearsonr(boot_values, boot_fitnesses)
                if not np.isnan(corr):
                    bootstrap_correlations.append(corr)

            if bootstrap_correlations:
                lower = float(np.percentile(bootstrap_correlations, 2.5))
                upper = float(np.percentile(bootstrap_correlations, 97.5))
                return lower, upper

            return 0.0, 0.0

        except Exception as e:
            logger.debug(f"Bootstrap CI failed: {e}")
            return 0.0, 0.0

    def get_validated_variables(self) -> list[str]:
        """Get names of all variables that passed validation."""
        return [r.variable_name for r in self.validation_history if r.passed]

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary statistics of validation history."""
        if not self.validation_history:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = sum(1 for r in self.validation_history if r.passed)
        return {
            "total": len(self.validation_history),
            "passed": passed,
            "failed": len(self.validation_history) - passed,
            "pass_rate": passed / len(self.validation_history),
            "avg_correlation": np.mean([r.correlation for r in self.validation_history]),
            "avg_incremental_r2": np.mean([r.incremental_r2 for r in self.validation_history]),
        }
