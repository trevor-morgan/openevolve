"""
Symbolic Regression Tool - Discover mathematical formulas from data.

"The mystery of life isn't a problem to solve, but a reality to experience."

Uses PySR (or gplearn as fallback) to discover mathematical expressions
that predict fitness from program features. This finds relationships
that don't have names - TRUE ontological discovery.

Example discoveries:
- fitness ~ sqrt(coil_count) * log(mirror_ratio)
- stability ~ exp(-well_depth^2) * symmetry_index

These are formulas that emerge from the data, not from human intuition.
"""

import logging
import re
from typing import Any, ClassVar

import numpy as np

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class SymbolicRegressionTool(DiscoveryTool):
    """
    Discovers mathematical formulas using symbolic regression.

    Primary: PySR (state-of-the-art symbolic regression)
    Fallback: gplearn (simpler but no Julia dependency)
    """

    name = "symbolic_regression"
    description = "Discover mathematical formulas that predict fitness from program features"
    discovery_types: ClassVar[list[DiscoveryType]] = [
        DiscoveryType.MATHEMATICAL_FORMULA,
        DiscoveryType.LATENT_VARIABLE,
    ]
    dependencies: ClassVar[list[str]] = []  # Check dynamically

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._backend: str | None = None

    def _check_dependencies(self) -> bool:
        """Check for PySR or gplearn."""
        import importlib.util

        if importlib.util.find_spec("pysr"):
            self._backend = "pysr"
            return True

        if importlib.util.find_spec("gplearn"):
            self._backend = "gplearn"
            return True

        return False

    def _extract_features(self, programs: list[dict[str, Any]]) -> tuple:
        """Extract numerical features from programs for regression."""
        feature_names = []
        feature_matrix = []

        for prog in programs:
            features = {}

            # Extract from metrics
            for k, v in prog.get("metrics", {}).items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    features[f"metric_{k}"] = v

            # Extract from code
            code = prog.get("code", "")
            if code:
                # Count patterns
                features["n_functions"] = len(re.findall(r"def \w+\(", code))
                features["n_loops"] = len(re.findall(r"\b(for|while)\b", code))
                features["n_calls"] = len(re.findall(r"add_coil\(", code))
                features["code_length"] = len(code)
                features["n_numbers"] = len(re.findall(r"\b\d+\.?\d*\b", code))

                # Extract numeric values from code
                numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", code)]
                if numbers:
                    features["num_mean"] = np.mean(numbers)
                    features["num_std"] = np.std(numbers) if len(numbers) > 1 else 0
                    features["num_max"] = max(numbers)
                    features["num_min"] = min(numbers)
                    features["num_range"] = features["num_max"] - features["num_min"]

            feature_matrix.append(features)

        # Get common feature names
        if feature_matrix:
            all_keys = set()
            for f in feature_matrix:
                all_keys.update(f.keys())
            feature_names = sorted(all_keys)

            # Build matrix
            X = []
            for f in feature_matrix:
                row = [f.get(k, 0.0) for k in feature_names]
                X.append(row)
            X = np.array(X)

            # Remove constant columns
            mask = np.std(X, axis=0) > 1e-10
            X = X[:, mask]
            feature_names = [n for n, m in zip(feature_names, mask) if m]

            return X, feature_names

        return np.array([]), []

    async def discover(self, context: ToolContext) -> list[Discovery]:
        """Run symbolic regression to discover formulas."""
        logger.info(f"Running symbolic regression ({self._backend})...")

        programs = context.programs
        if len(programs) < 10:
            logger.warning("Not enough programs for symbolic regression")
            return []

        # Extract features
        X, feature_names = self._extract_features(programs)
        y = np.array([p.get("fitness", 0.0) for p in programs])

        if X.shape[1] == 0:
            logger.warning("No features extracted for symbolic regression")
            return []

        # Normalize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-10
        X_norm = (X - X_mean) / X_std

        y_mean = y.mean()
        y_std = y.std() + 1e-10
        y_norm = (y - y_mean) / y_std

        discoveries = []

        if self._backend == "pysr":
            discoveries = await self._run_pysr(
                X_norm, y_norm, feature_names, X_mean, X_std, y_mean, y_std
            )
        elif self._backend == "gplearn":
            discoveries = await self._run_gplearn(
                X_norm, y_norm, feature_names, X_mean, X_std, y_mean, y_std
            )
        else:
            discoveries = await self._run_numpy_fallback(
                X_norm, y_norm, feature_names, X_mean, X_std, y_mean, y_std
            )

        logger.info(f"Symbolic regression found {len(discoveries)} formulas")
        return discoveries

    async def _run_pysr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        X_mean: np.ndarray,
        X_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> list[Discovery]:
        """Run PySR symbolic regression."""
        import pysr

        model = pysr.PySRRegressor(
            niterations=self.config.get("niterations", 40),
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sqrt", "log", "exp", "sin", "cos", "abs"],
            populations=self.config.get("populations", 15),
            population_size=self.config.get("population_size", 33),
            maxsize=self.config.get("maxsize", 20),
            timeout_in_seconds=self.config.get("timeout", 300),
            temp_equation_file=True,
            verbosity=0,
        )

        model.fit(X, y, variable_names=feature_names)

        discoveries = []
        for i, eq in enumerate(model.equations_):
            if eq.loss < 0.5:  # Reasonable fit
                formula = str(eq.sympy_format)
                discoveries.append(
                    self._create_discovery(
                        name=f"sr_formula_{i}",
                        formula=formula,
                        feature_names=feature_names,
                        r2=1 - eq.loss,
                        complexity=eq.complexity,
                        X_mean=X_mean,
                        X_std=X_std,
                        y_mean=y_mean,
                        y_std=y_std,
                    )
                )

        return discoveries

    async def _run_gplearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        X_mean: np.ndarray,
        X_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> list[Discovery]:
        """Run gplearn symbolic regression."""
        from gplearn.genetic import SymbolicRegressor

        model = SymbolicRegressor(
            population_size=self.config.get("population_size", 1000),
            generations=self.config.get("generations", 20),
            tournament_size=20,
            stopping_criteria=0.01,
            const_range=(-1, 1),
            init_depth=(2, 6),
            init_method="half and half",
            function_set=["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "sin", "cos"],
            parsimony_coefficient=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            random_state=42,
        )

        model.fit(X, y)

        # Get best program
        formula = str(model._program)

        # Replace X0, X1, etc with feature names
        for i, name in enumerate(feature_names):
            formula = formula.replace(f"X{i}", name)

        y_pred = model.predict(X)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        discoveries = [
            self._create_discovery(
                name="sr_gplearn_best",
                formula=formula,
                feature_names=feature_names,
                r2=r2,
                complexity=model._program.length_,
                X_mean=X_mean,
                X_std=X_std,
                y_mean=y_mean,
                y_std=y_std,
            )
        ]

        return discoveries

    async def _run_numpy_fallback(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        X_mean: np.ndarray,
        X_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> list[Discovery]:
        """Fallback: find best linear combination and simple nonlinear terms."""
        discoveries = []

        # Linear regression
        try:
            coeffs, _residuals, _rank, _s = np.linalg.lstsq(
                np.column_stack([np.ones(len(X)), X]), y, rcond=None
            )

            # Build formula
            terms = []
            for i, (name, coef) in enumerate(zip(feature_names, coeffs[1:])):
                if abs(coef) > 0.01:
                    terms.append(f"{coef:.3f}*{name}")

            if terms:
                formula = f"{coeffs[0]:.3f} + " + " + ".join(terms)

                y_pred = np.column_stack([np.ones(len(X)), X]) @ coeffs
                r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

                discoveries.append(
                    self._create_discovery(
                        name="sr_linear",
                        formula=formula,
                        feature_names=feature_names,
                        r2=r2,
                        complexity=len(terms) + 1,
                        X_mean=X_mean,
                        X_std=X_std,
                        y_mean=y_mean,
                        y_std=y_std,
                    )
                )
        except Exception as e:
            logger.debug(f"Linear regression failed: {e}")

        # Try sqrt and log transforms of best features
        for i, name in enumerate(feature_names):
            x = X[:, i]
            if np.all(x > 0):
                # sqrt transform
                corr_sqrt = np.corrcoef(np.sqrt(x), y)[0, 1]
                if abs(corr_sqrt) > 0.3:
                    discoveries.append(
                        Discovery(
                            name=f"sr_sqrt_{name}",
                            description=f"Square root relationship: fitness ~ sqrt({name})",
                            discovery_type=DiscoveryType.MATHEMATICAL_FORMULA,
                            content={
                                "formula": f"sqrt({name})",
                                "correlation": float(corr_sqrt),
                                "feature": name,
                            },
                            confidence=abs(corr_sqrt),
                            evidence=[f"Correlation: {corr_sqrt:.3f}"],
                        )
                    )

                # log transform
                corr_log = np.corrcoef(np.log(x + 1e-10), y)[0, 1]
                if abs(corr_log) > 0.3:
                    discoveries.append(
                        Discovery(
                            name=f"sr_log_{name}",
                            description=f"Logarithmic relationship: fitness ~ log({name})",
                            discovery_type=DiscoveryType.MATHEMATICAL_FORMULA,
                            content={
                                "formula": f"log({name})",
                                "correlation": float(corr_log),
                                "feature": name,
                            },
                            confidence=abs(corr_log),
                            evidence=[f"Correlation: {corr_log:.3f}"],
                        )
                    )

        return discoveries

    def _create_discovery(
        self,
        name: str,
        formula: str,
        feature_names: list[str],
        r2: float,
        complexity: int,
        X_mean: np.ndarray,
        X_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> Discovery:
        """Create a Discovery object from a symbolic regression result."""

        # Generate computation code
        # This transforms the formula into executable Python
        computation_code = self._formula_to_code(name, formula, feature_names)

        return Discovery(
            name=name,
            description=f"Discovered formula: {formula} (R²={r2:.3f})",
            discovery_type=DiscoveryType.MATHEMATICAL_FORMULA,
            content={
                "formula": formula,
                "r2": float(r2),
                "complexity": complexity,
                "feature_names": feature_names,
                "normalization": {
                    "X_mean": X_mean.tolist(),
                    "X_std": X_std.tolist(),
                    "y_mean": float(y_mean),
                    "y_std": float(y_std),
                },
            },
            computation_code=computation_code,
            confidence=min(1.0, r2),
            evidence=[
                f"R² = {r2:.3f}",
                f"Complexity = {complexity}",
                f"Features used: {', '.join(feature_names)}",
            ],
            testable=True,
            validation_criteria=f"Must achieve R² > {r2 * 0.8:.3f} on held-out data",
        )

    def _formula_to_code(self, name: str, formula: str, feature_names: list[str]) -> str:
        """Convert a formula string to executable Python code."""

        # Sanitize name for function
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        code = f'''
def compute_{safe_name}(code: str, metrics: dict) -> float:
    """
    Discovered formula: {formula}

    Auto-generated by SymbolicRegressionTool.
    """
    import numpy as np
    import re

    # Extract features from code and metrics
    features = {{}}

    # From metrics
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            features[f"metric_{{k}}"] = v

    # From code structure
    features["n_functions"] = len(re.findall(r'def \\w+\\(', code))
    features["n_loops"] = len(re.findall(r'\\b(for|while)\\b', code))
    features["n_calls"] = len(re.findall(r'add_coil\\(', code))
    features["code_length"] = len(code)

    # Extract numbers from code
    numbers = [float(x) for x in re.findall(r'[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', code)]
    if numbers:
        features["num_mean"] = np.mean(numbers)
        features["num_std"] = np.std(numbers) if len(numbers) > 1 else 0
        features["num_max"] = max(numbers)
        features["num_min"] = min(numbers)
        features["num_range"] = features["num_max"] - features["num_min"]
    else:
        features["num_mean"] = 0
        features["num_std"] = 0
        features["num_max"] = 0
        features["num_min"] = 0
        features["num_range"] = 0

    # Compute the formula
    try:
        # Make feature variables available
        {self._generate_feature_assignments(feature_names)}

        # The discovered formula
        result = {self._sanitize_formula(formula)}
        return float(result) if np.isfinite(result) else 0.0
    except Exception:
        return 0.0
'''
        return code

    def _generate_feature_assignments(self, feature_names: list[str]) -> str:
        """Generate variable assignments for features."""
        lines = []
        for name in feature_names:
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            lines.append(f'{safe_name} = features.get("{name}", 0.0)')
        return "\n        ".join(lines)

    def _sanitize_formula(self, formula: str) -> str:
        """Sanitize formula for safe execution."""
        # Replace common symbolic notation
        formula = formula.replace("^", "**")
        formula = formula.replace("sqrt", "np.sqrt")
        formula = formula.replace("log", "np.log")
        formula = formula.replace("exp", "np.exp")
        formula = formula.replace("sin", "np.sin")
        formula = formula.replace("cos", "np.cos")
        formula = formula.replace("abs", "np.abs")

        # Sanitize variable names
        formula = re.sub(r"metric_(\w+)", r"metric_\1", formula)

        return formula
