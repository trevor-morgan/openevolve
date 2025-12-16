"""
Code Analysis Tool - Structural pattern discovery via AST analysis.

"Deep in the human unconscious is a pervasive need for a logical universe
that makes sense. But the real universe is always one step beyond logic."

Uses Python's AST to discover structural patterns in code that correlate
with fitness. No external dependencies required.

Example discoveries:
- Programs with nested loops in specific patterns perform better
- Certain function call orderings correlate with success
- Numeric literal distributions predict fitness
"""

import ast
import logging
import re
from collections import Counter
from typing import ClassVar

import numpy as np
from scipy import stats

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class CodeAnalysisTool(DiscoveryTool):
    """
    Discovers structural patterns in code using AST analysis.

    No external dependencies - uses Python's built-in ast module.
    """

    name = "code_analysis"
    description = "Analyze code structure to find patterns that correlate with fitness"
    discovery_types: ClassVar[list[DiscoveryType]] = [
        DiscoveryType.CODE_PATTERN,
        DiscoveryType.LATENT_VARIABLE,
    ]
    dependencies: ClassVar[list[str]] = []  # No external dependencies

    def _check_dependencies(self) -> bool:
        return True  # Always available

    async def discover(self, context: ToolContext) -> list[Discovery]:
        """Analyze code structure to find patterns."""
        logger.info("Running code analysis...")

        programs = context.programs
        if len(programs) < 10:
            logger.warning("Not enough programs for code analysis")
            return []

        discoveries = []

        # Extract structural features
        features_list = []
        fitnesses = []
        for prog in programs:
            code = prog.get("code", "")
            if code:
                features = self._extract_ast_features(code)
                features_list.append(features)
                fitnesses.append(prog.get("fitness", 0.0))

        if not features_list:
            return []

        fitnesses = np.array(fitnesses)

        # Find features that correlate with fitness
        all_feature_names = set()
        for f in features_list:
            all_feature_names.update(f.keys())

        for feature_name in all_feature_names:
            values = np.array([f.get(feature_name, 0) for f in features_list])

            # Skip constant features
            if np.std(values) < 1e-10:
                continue

            # Compute correlation
            try:
                corr, p_value = stats.pearsonr(values, fitnesses)

                if abs(corr) > 0.2 and p_value < 0.1:
                    discoveries.append(
                        self._create_pattern_discovery(
                            feature_name, corr, p_value, values, fitnesses
                        )
                    )
            except Exception:
                continue

        # Look for interaction effects (feature combinations)
        interaction_discoveries = self._find_interactions(features_list, fitnesses)
        discoveries.extend(interaction_discoveries)

        # Look for threshold effects
        threshold_discoveries = self._find_thresholds(features_list, fitnesses)
        discoveries.extend(threshold_discoveries)

        logger.info(f"Code analysis found {len(discoveries)} patterns")
        return discoveries

    def _extract_ast_features(self, code: str) -> dict[str, float]:
        """Extract structural features from code using AST."""
        features = {}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to regex-based extraction
            return self._extract_regex_features(code)

        # Count node types
        node_counts = Counter(type(node).__name__ for node in ast.walk(tree))

        for node_type, count in node_counts.items():
            features[f"ast_{node_type.lower()}"] = count

        # Structural metrics
        features["ast_depth"] = self._tree_depth(tree)
        features["ast_total_nodes"] = sum(node_counts.values())
        features["ast_num_functions"] = node_counts.get("FunctionDef", 0)
        features["ast_num_classes"] = node_counts.get("ClassDef", 0)
        features["ast_num_loops"] = node_counts.get("For", 0) + node_counts.get("While", 0)
        features["ast_num_conditionals"] = node_counts.get("If", 0)
        features["ast_num_calls"] = node_counts.get("Call", 0)
        features["ast_num_binops"] = node_counts.get("BinOp", 0)
        features["ast_num_comparisons"] = node_counts.get("Compare", 0)

        # Extract numeric literals
        numbers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                numbers.append(node.value)

        if numbers:
            features["num_literals_count"] = len(numbers)
            features["num_literals_mean"] = np.mean(numbers)
            features["num_literals_std"] = np.std(numbers) if len(numbers) > 1 else 0
            features["num_literals_max"] = max(numbers)
            features["num_literals_min"] = min(numbers)
            features["num_literals_range"] = (
                features["num_literals_max"] - features["num_literals_min"]
            )
            features["num_positive_count"] = sum(1 for n in numbers if n > 0)
            features["num_negative_count"] = sum(1 for n in numbers if n < 0)

        # Function call analysis
        call_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_names.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    call_names.append(node.func.attr)

        call_counts = Counter(call_names)
        for name, count in call_counts.most_common(10):
            features[f"call_{name}"] = count

        features["unique_calls"] = len(set(call_names))
        features["total_calls"] = len(call_names)

        # Nesting analysis
        features["max_nesting"] = self._max_nesting_depth(tree)

        # Add regex features too
        features.update(self._extract_regex_features(code))

        return features

    def _extract_regex_features(self, code: str) -> dict[str, float]:
        """Extract features using regex (fallback for non-Python or broken code)."""
        features = {}

        features["code_length"] = len(code)
        features["num_lines"] = code.count("\n") + 1
        features["num_empty_lines"] = len([l for l in code.split("\n") if not l.strip()])

        # Pattern counts
        features["regex_functions"] = len(re.findall(r"def \w+\(", code))
        features["regex_classes"] = len(re.findall(r"class \w+", code))
        features["regex_for_loops"] = len(re.findall(r"\bfor\b", code))
        features["regex_while_loops"] = len(re.findall(r"\bwhile\b", code))
        features["regex_if_statements"] = len(re.findall(r"\bif\b", code))
        features["regex_imports"] = len(re.findall(r"\bimport\b", code))
        features["regex_comments"] = len(re.findall(r"#.*$", code, re.MULTILINE))

        # Domain-specific (adjust for your domain)
        features["regex_add_coil"] = len(re.findall(r"add_coil\(", code))
        features["regex_return"] = len(re.findall(r"\breturn\b", code))

        # Numeric extraction
        numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", code)]
        if numbers:
            features["regex_num_count"] = len(numbers)
            features["regex_num_mean"] = np.mean(numbers)
            features["regex_num_max"] = max(numbers)
            features["regex_num_min"] = min(numbers)

        return features

    def _tree_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum depth of AST."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            max_depth = max(max_depth, self._tree_depth(child, depth + 1))
        return max_depth

    def _max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth of control structures."""

        def get_nesting(node, depth=0):
            max_depth = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                    max_depth = max(max_depth, get_nesting(child, depth + 1))
                else:
                    max_depth = max(max_depth, get_nesting(child, depth))
            return max_depth

        return get_nesting(tree)

    def _create_pattern_discovery(
        self,
        feature_name: str,
        corr: float,
        p_value: float,
        values: np.ndarray,
        fitnesses: np.ndarray,
    ) -> Discovery:
        """Create a discovery from a correlating feature."""

        # Determine relationship direction
        direction = "higher" if corr > 0 else "lower"

        # Create computation code
        computation_code = self._generate_computation_code(feature_name)

        return Discovery(
            name=f"pattern_{feature_name}",
            description=f"Code pattern: {direction} {feature_name} correlates with better fitness (r={corr:.3f})",
            discovery_type=DiscoveryType.CODE_PATTERN,
            content={
                "feature_name": feature_name,
                "correlation": float(corr),
                "p_value": float(p_value),
                "direction": direction,
                "value_range": [float(values.min()), float(values.max())],
                "value_mean": float(values.mean()),
            },
            computation_code=computation_code,
            confidence=min(1.0, abs(corr)),
            evidence=[
                f"Correlation: r = {corr:.3f}",
                f"P-value: {p_value:.4f}",
                f"Value range: [{values.min():.2f}, {values.max():.2f}]",
            ],
            testable=True,
            validation_criteria=f"Correlation should be {'>0.15' if corr > 0 else '<-0.15'} on new data",
        )

    def _find_interactions(
        self, features_list: list[dict[str, float]], fitnesses: np.ndarray
    ) -> list[Discovery]:
        """Find interaction effects between features."""
        discoveries = []

        # Get common features
        all_keys = set()
        for f in features_list:
            all_keys.update(f.keys())
        common_keys = [k for k in all_keys if all(k in f for f in features_list)]

        # Test pairwise interactions (limit to prevent explosion)
        tested = 0
        for i, k1 in enumerate(common_keys[:20]):
            for k2 in common_keys[i + 1 : 20]:
                if tested > 100:
                    break

                v1 = np.array([f[k1] for f in features_list])
                v2 = np.array([f[k2] for f in features_list])

                # Skip if either is constant
                if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
                    continue

                # Test ratio
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = v1 / (v2 + 1e-10)
                    if np.all(np.isfinite(ratio)):
                        corr, p = stats.pearsonr(ratio, fitnesses)
                        if abs(corr) > 0.3 and p < 0.05:
                            discoveries.append(
                                Discovery(
                                    name=f"interaction_{k1}_div_{k2}",
                                    description=f"Ratio {k1}/{k2} correlates with fitness (r={corr:.3f})",
                                    discovery_type=DiscoveryType.CODE_PATTERN,
                                    content={
                                        "type": "ratio",
                                        "features": [k1, k2],
                                        "correlation": float(corr),
                                    },
                                    confidence=abs(corr),
                                    evidence=[f"Ratio correlation: {corr:.3f}"],
                                )
                            )

                # Test product
                product = v1 * v2
                if np.std(product) > 1e-10:
                    corr, p = stats.pearsonr(product, fitnesses)
                    if abs(corr) > 0.3 and p < 0.05:
                        discoveries.append(
                            Discovery(
                                name=f"interaction_{k1}_mul_{k2}",
                                description=f"Product {k1}*{k2} correlates with fitness (r={corr:.3f})",
                                discovery_type=DiscoveryType.CODE_PATTERN,
                                content={
                                    "type": "product",
                                    "features": [k1, k2],
                                    "correlation": float(corr),
                                },
                                confidence=abs(corr),
                                evidence=[f"Product correlation: {corr:.3f}"],
                            )
                        )

                tested += 1

        return discoveries[:5]  # Limit to top 5

    def _find_thresholds(
        self, features_list: list[dict[str, float]], fitnesses: np.ndarray
    ) -> list[Discovery]:
        """Find threshold effects (binary patterns)."""
        discoveries = []

        all_keys = set()
        for f in features_list:
            all_keys.update(f.keys())

        for key in list(all_keys)[:30]:
            values = np.array([f.get(key, 0) for f in features_list])

            if np.std(values) < 1e-10:
                continue

            # Try different thresholds
            for percentile in [25, 50, 75]:
                threshold = np.percentile(values, percentile)

                above = fitnesses[values > threshold]
                below = fitnesses[values <= threshold]

                if len(above) > 5 and len(below) > 5:
                    try:
                        t_stat, p_value = stats.ttest_ind(above, below)

                        if p_value < 0.05 and abs(np.mean(above) - np.mean(below)) > 0.05:
                            better = "above" if np.mean(above) > np.mean(below) else "below"
                            discoveries.append(
                                Discovery(
                                    name=f"threshold_{key}_p{percentile}",
                                    description=f"Programs with {key} {better} {threshold:.2f} perform better",
                                    discovery_type=DiscoveryType.CODE_PATTERN,
                                    content={
                                        "type": "threshold",
                                        "feature": key,
                                        "threshold": float(threshold),
                                        "percentile": percentile,
                                        "better_side": better,
                                        "mean_above": float(np.mean(above)),
                                        "mean_below": float(np.mean(below)),
                                        "t_statistic": float(t_stat),
                                        "p_value": float(p_value),
                                    },
                                    confidence=min(1.0, abs(t_stat) / 3),
                                    evidence=[
                                        f"Mean fitness above threshold: {np.mean(above):.3f}",
                                        f"Mean fitness below threshold: {np.mean(below):.3f}",
                                        f"P-value: {p_value:.4f}",
                                    ],
                                )
                            )
                    except Exception:
                        continue

        return discoveries[:5]

    def _generate_computation_code(self, feature_name: str) -> str:
        """Generate code to compute a discovered feature."""

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", feature_name)

        # Determine extraction method based on feature name
        if feature_name.startswith("ast_"):
            return self._generate_ast_code(safe_name, feature_name)
        elif feature_name.startswith("regex_"):
            return self._generate_regex_code(safe_name, feature_name)
        elif feature_name.startswith("call_"):
            return self._generate_call_code(safe_name, feature_name)
        else:
            return self._generate_generic_code(safe_name, feature_name)

    def _generate_ast_code(self, safe_name: str, feature_name: str) -> str:
        """Generate AST-based extraction code."""
        return f'''
def compute_{safe_name}(code: str, metrics: dict) -> float:
    """Compute {feature_name} from code structure."""
    import ast
    from collections import Counter

    try:
        tree = ast.parse(code)
        node_counts = Counter(type(node).__name__ for node in ast.walk(tree))

        # Extract the specific feature
        feature_key = "{feature_name}".replace("ast_", "")
        if feature_key in node_counts:
            return float(node_counts[feature_key])

        # Handle derived metrics
        if feature_key == "depth":
            def tree_depth(node, d=0):
                return max([d] + [tree_depth(c, d+1) for c in ast.iter_child_nodes(node)])
            return float(tree_depth(tree))

        if feature_key == "total_nodes":
            return float(sum(node_counts.values()))

        return 0.0
    except:
        return 0.0
'''

    def _generate_regex_code(self, safe_name: str, feature_name: str) -> str:
        """Generate regex-based extraction code."""
        # Map feature names to patterns
        patterns = {
            "regex_functions": r"def \w+\(",
            "regex_for_loops": r"\bfor\b",
            "regex_while_loops": r"\bwhile\b",
            "regex_if_statements": r"\bif\b",
            "regex_add_coil": r"add_coil\(",
            "regex_return": r"\breturn\b",
        }

        pattern = patterns.get(feature_name, "")

        # Escape quotes for embedding in generated code
        escaped_pattern = pattern.replace("'", "\\'")

        return f'''
def compute_{safe_name}(code: str, metrics: dict) -> float:
    """Compute {feature_name} from code."""
    import re
    pattern = r'{escaped_pattern}'
    if pattern:
        return float(len(re.findall(pattern, code)))
    return 0.0
'''

    def _generate_call_code(self, safe_name: str, feature_name: str) -> str:
        """Generate function call counting code."""
        call_name = feature_name.replace("call_", "")

        return f'''
def compute_{safe_name}(code: str, metrics: dict) -> float:
    """Count calls to {call_name} in code."""
    import re
    pattern = r'\\b{call_name}\\s*\\('
    return float(len(re.findall(pattern, code)))
'''

    def _generate_generic_code(self, safe_name: str, feature_name: str) -> str:
        """Generate generic feature extraction code."""
        return f'''
def compute_{safe_name}(code: str, metrics: dict) -> float:
    """Compute {feature_name} from code."""
    import re
    import numpy as np

    # Try to extract from metrics first
    if "{feature_name}" in metrics:
        return float(metrics["{feature_name}"])

    # Extract numbers and compute statistics
    numbers = [float(x) for x in re.findall(r'[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', code)]

    if "mean" in "{feature_name}" and numbers:
        return float(np.mean(numbers))
    if "max" in "{feature_name}" and numbers:
        return float(max(numbers))
    if "min" in "{feature_name}" and numbers:
        return float(min(numbers))
    if "count" in "{feature_name}" and numbers:
        return float(len(numbers))

    return 0.0
'''
