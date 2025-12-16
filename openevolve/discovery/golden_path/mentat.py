"""
Mentat - Human computers who process vast amounts of data to find patterns.

"It is by will alone I set my mind in motion."

The Mentat mines evolved programs to discover hidden patterns that correlate
with success. Unlike simple metrics, Mentat finds EMERGENT features:
- Structural patterns in code
- Relationships between components
- Invariants that successful programs share
- Features the current ontology doesn't capture

"The first step in avoiding a trap is knowing of its existence."
"""

import ast
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPattern:
    """A pattern discovered by the Mentat."""

    name: str
    description: str
    pattern_type: str  # "structural", "numeric", "relational", "semantic"

    # Pattern definition
    extractor: Callable[[str], float] | None = None  # Function to extract feature value
    extraction_code: str | None = None  # Code representation of extractor

    # Statistics
    correlation_with_fitness: float = 0.0
    presence_in_successful: float = 0.0  # What fraction of successful programs have this
    presence_in_unsuccessful: float = 0.0
    discriminative_power: float = 0.0  # How well does this separate success from failure

    # Evidence
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MentatConfig:
    """Configuration for the Mentat."""

    # Analysis settings
    min_programs_for_analysis: int = 20
    top_n_patterns: int = 10

    # Correlation thresholds
    min_correlation_threshold: float = 0.3
    min_discriminative_power: float = 0.2

    # Pattern types to search for
    search_structural: bool = True
    search_numeric: bool = True
    search_relational: bool = True


class Mentat:
    """
    The Mentat - mines programs to discover hidden patterns.

    "The mind commands the body and it obeys. The mind orders itself and meets resistance."
    """

    def __init__(self, config: MentatConfig | None = None):
        self.config = config or MentatConfig()
        self.discovered_patterns: list[ExtractedPattern] = []

    def analyze_programs(
        self,
        programs: list[dict[str, Any]],
    ) -> list[ExtractedPattern]:
        """
        Analyze a set of programs to discover hidden patterns.

        Args:
            programs: List of dicts with 'code', 'fitness', 'metrics' keys

        Returns:
            List of discovered patterns, sorted by discriminative power
        """
        if len(programs) < self.config.min_programs_for_analysis:
            logger.warning(
                f"Mentat needs at least {self.config.min_programs_for_analysis} programs"
            )
            return []

        logger.info(f"Mentat analyzing {len(programs)} programs for hidden patterns...")

        all_patterns = []

        # Extract different types of patterns
        if self.config.search_structural:
            structural = self._extract_structural_patterns(programs)
            all_patterns.extend(structural)

        if self.config.search_numeric:
            numeric = self._extract_numeric_patterns(programs)
            all_patterns.extend(numeric)

        if self.config.search_relational:
            relational = self._extract_relational_patterns(programs)
            all_patterns.extend(relational)

        # Filter by discriminative power
        valid_patterns = [
            p
            for p in all_patterns
            if p.discriminative_power >= self.config.min_discriminative_power
            or abs(p.correlation_with_fitness) >= self.config.min_correlation_threshold
        ]

        # Sort by discriminative power
        valid_patterns.sort(key=lambda p: p.discriminative_power, reverse=True)

        # Keep top N
        top_patterns = valid_patterns[: self.config.top_n_patterns]

        logger.info(f"Mentat discovered {len(top_patterns)} significant patterns")
        for p in top_patterns[:5]:
            logger.info(
                f"  - {p.name}: corr={p.correlation_with_fitness:.3f}, disc={p.discriminative_power:.3f}"
            )

        self.discovered_patterns.extend(top_patterns)
        return top_patterns

    def _extract_structural_patterns(
        self,
        programs: list[dict[str, Any]],
    ) -> list[ExtractedPattern]:
        """Extract structural patterns from code."""
        patterns = []

        # Define structural features to extract
        structural_features = [
            ("num_functions", self._count_functions, "Number of function definitions"),
            ("num_loops", self._count_loops, "Number of loop structures"),
            ("num_conditionals", self._count_conditionals, "Number of conditional branches"),
            ("max_nesting_depth", self._compute_max_nesting, "Maximum nesting depth"),
            ("num_numeric_literals", self._count_numeric_literals, "Count of numeric literals"),
            ("code_complexity", self._compute_complexity, "Cyclomatic-like complexity"),
            (
                "num_list_comprehensions",
                self._count_comprehensions,
                "Number of list comprehensions",
            ),
            ("num_math_ops", self._count_math_operations, "Number of math operations"),
            ("has_numpy", self._has_numpy, "Uses numpy"),
            ("num_array_operations", self._count_array_ops, "Number of array operations"),
        ]

        for name, extractor, description in structural_features:
            pattern = self._evaluate_feature(
                name=name,
                description=description,
                pattern_type="structural",
                extractor=extractor,
                programs=programs,
            )
            if pattern:
                patterns.append(pattern)

        return patterns

    def _extract_numeric_patterns(
        self,
        programs: list[dict[str, Any]],
    ) -> list[ExtractedPattern]:
        """Extract patterns from numeric values in the code."""
        patterns = []

        # Collect all numeric literals from successful vs unsuccessful programs
        fitnesses = [p["fitness"] for p in programs]
        median_fitness = np.median(fitnesses)

        successful_nums = []
        unsuccessful_nums = []

        for prog in programs:
            nums = self._extract_numeric_values(prog["code"])
            if prog["fitness"] > median_fitness:
                successful_nums.extend(nums)
            else:
                unsuccessful_nums.extend(nums)

        # Look for numeric ranges that correlate with success
        numeric_features = [
            (
                "has_large_numbers",
                lambda c: 1.0 if any(n > 1000 for n in self._extract_numeric_values(c)) else 0.0,
                "Contains numbers > 1000",
            ),
            (
                "has_small_decimals",
                lambda c: 1.0
                if any(0 < abs(n) < 0.1 for n in self._extract_numeric_values(c))
                else 0.0,
                "Contains small decimals (0-0.1)",
            ),
            (
                "num_unique_values",
                lambda c: len(set(self._extract_numeric_values(c))),
                "Count of unique numeric values",
            ),
            (
                "mean_numeric_value",
                lambda c: np.mean(self._extract_numeric_values(c))
                if self._extract_numeric_values(c)
                else 0,
                "Mean of numeric values",
            ),
            (
                "numeric_range",
                lambda c: (
                    max(self._extract_numeric_values(c)) - min(self._extract_numeric_values(c))
                )
                if len(self._extract_numeric_values(c)) > 1
                else 0,
                "Range of numeric values",
            ),
        ]

        for name, extractor, description in numeric_features:
            pattern = self._evaluate_feature(
                name=name,
                description=description,
                pattern_type="numeric",
                extractor=extractor,
                programs=programs,
            )
            if pattern:
                patterns.append(pattern)

        return patterns

    def _extract_relational_patterns(
        self,
        programs: list[dict[str, Any]],
    ) -> list[ExtractedPattern]:
        """Extract relational patterns - relationships between code elements."""
        patterns = []

        relational_features = [
            ("function_to_loop_ratio", self._function_to_loop_ratio, "Ratio of functions to loops"),
            ("literal_density", self._literal_density, "Numeric literals per line"),
            ("variable_reuse_ratio", self._variable_reuse_ratio, "How much variables are reused"),
            ("symmetry_score", self._symmetry_score, "Code symmetry/repetition"),
        ]

        for name, extractor, description in relational_features:
            pattern = self._evaluate_feature(
                name=name,
                description=description,
                pattern_type="relational",
                extractor=extractor,
                programs=programs,
            )
            if pattern:
                patterns.append(pattern)

        return patterns

    def _evaluate_feature(
        self,
        name: str,
        description: str,
        pattern_type: str,
        extractor: Callable[[str], float],
        programs: list[dict[str, Any]],
    ) -> ExtractedPattern | None:
        """Evaluate a feature extractor against the program set."""
        try:
            # Extract feature values
            feature_values = []
            fitnesses = []

            for prog in programs:
                try:
                    value = extractor(prog["code"])
                    feature_values.append(float(value))
                    fitnesses.append(prog["fitness"])
                except Exception:
                    continue

            if len(feature_values) < 10:
                return None

            feature_values = np.array(feature_values)
            fitnesses = np.array(fitnesses)

            # Compute correlation
            if np.std(feature_values) > 0 and np.std(fitnesses) > 0:
                correlation = np.corrcoef(feature_values, fitnesses)[0, 1]
            else:
                correlation = 0.0

            # Compute discriminative power
            median_fitness = np.median(fitnesses)
            successful_mask = fitnesses > median_fitness

            successful_values = feature_values[successful_mask]
            unsuccessful_values = feature_values[~successful_mask]

            if len(successful_values) > 0 and len(unsuccessful_values) > 0:
                # Use effect size (Cohen's d) as discriminative power
                pooled_std = np.sqrt((np.var(successful_values) + np.var(unsuccessful_values)) / 2)
                if pooled_std > 0:
                    effect_size = (
                        abs(np.mean(successful_values) - np.mean(unsuccessful_values)) / pooled_std
                    )
                else:
                    effect_size = 0.0

                presence_successful = np.mean(successful_values > 0)
                presence_unsuccessful = np.mean(unsuccessful_values > 0)
            else:
                effect_size = 0.0
                presence_successful = 0.0
                presence_unsuccessful = 0.0

            # Generate extraction code
            extraction_code = self._generate_extraction_code(name, extractor)

            return ExtractedPattern(
                name=name,
                description=description,
                pattern_type=pattern_type,
                extractor=extractor,
                extraction_code=extraction_code,
                correlation_with_fitness=float(correlation) if not np.isnan(correlation) else 0.0,
                presence_in_successful=float(presence_successful),
                presence_in_unsuccessful=float(presence_unsuccessful),
                discriminative_power=float(effect_size),
            )

        except Exception as e:
            logger.debug(f"Error evaluating feature {name}: {e}")
            return None

    def _generate_extraction_code(self, name: str, extractor: Callable) -> str:
        """Generate code representation of the extractor."""
        # For now, return a placeholder - in production, would inspect the function
        return f"def extract_{name}(code: str) -> float:\n    # Auto-generated extractor\n    pass"

    # ==================== Feature Extractors ====================

    def _count_functions(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        except:
            return 0.0

    def _count_loops(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
        except:
            return 0.0

    def _count_conditionals(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
        except:
            return 0.0

    def _compute_max_nesting(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            max_depth = [0]

            def walk(node, depth):
                if isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                    depth += 1
                    max_depth[0] = max(max_depth[0], depth)
                for child in ast.iter_child_nodes(node):
                    walk(child, depth)

            walk(tree, 0)
            return float(max_depth[0])
        except:
            return 0.0

    def _count_numeric_literals(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Num))
        except:
            # Fallback: regex count
            return float(len(re.findall(r"\b\d+\.?\d*\b", code)))

    def _compute_complexity(self, code: str) -> float:
        """Simplified cyclomatic complexity."""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity

            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return float(complexity)
        except:
            return 1.0

    def _count_comprehensions(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
            )
        except:
            return 0.0

    def _count_math_operations(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            math_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
            return sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.BinOp) and isinstance(node.op, math_ops)
            )
        except:
            return 0.0

    def _has_numpy(self, code: str) -> float:
        return 1.0 if "numpy" in code or "np." in code else 0.0

    def _count_array_ops(self, code: str) -> float:
        """Count array-like operations."""
        patterns = [
            r"\[.*:.*\]",  # Slicing
            r"\.reshape\(",
            r"\.append\(",
            r"np\.\w+\(",
            r"\.sum\(",
            r"\.mean\(",
        ]
        count = sum(len(re.findall(p, code)) for p in patterns)
        return float(count)

    def _extract_numeric_values(self, code: str) -> list[float]:
        """Extract all numeric literals from code."""
        try:
            tree = ast.parse(code)
            nums = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Num):
                    nums.append(float(node.n))
                elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    nums.append(float(node.value))
            return nums
        except:
            # Fallback: regex
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", code)
            return [float(m) for m in matches if m]

    def _function_to_loop_ratio(self, code: str) -> float:
        funcs = self._count_functions(code)
        loops = self._count_loops(code)
        return funcs / (loops + 1)  # +1 to avoid division by zero

    def _literal_density(self, code: str) -> float:
        lines = code.count("\n") + 1
        literals = self._count_numeric_literals(code)
        return literals / lines

    def _variable_reuse_ratio(self, code: str) -> float:
        """Estimate variable reuse from identifier frequency."""
        try:
            tree = ast.parse(code)
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.append(node.id)

            if not names:
                return 0.0

            counter = Counter(names)
            avg_usage = np.mean(list(counter.values()))
            return float(avg_usage)
        except:
            return 0.0

    def _symmetry_score(self, code: str) -> float:
        """Measure code symmetry/repetition."""
        lines = code.split("\n")
        if len(lines) < 2:
            return 0.0

        # Count duplicate lines
        counter = Counter(lines)
        duplicates = sum(1 for count in counter.values() if count > 1)

        return duplicates / len(lines)

    def get_top_patterns(self, n: int = 5) -> list[ExtractedPattern]:
        """Get top N discovered patterns by discriminative power."""
        sorted_patterns = sorted(
            self.discovered_patterns, key=lambda p: p.discriminative_power, reverse=True
        )
        return sorted_patterns[:n]
