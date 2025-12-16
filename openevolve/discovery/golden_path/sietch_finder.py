"""
SietchFinder - Discovers hidden variables (the sietches - hidden places).

"The mystery of life isn't a problem to solve, but a reality to experience.
But there are hidden places in that reality that must be found." - Fremen wisdom

A sietch is a hidden Fremen community in the desert. In ontological discovery,
sietches are hidden variables - dimensions of the problem space that aren't
captured by the current representation but influence success.

SietchFinder uses:
1. Pattern mining results from Mentat
2. LLM reasoning about domain physics
3. Cross-program analysis of what makes solutions work
4. Hypothesis generation about unmeasured quantities

"He who controls the Spice controls the Universe."
But first you must find where the Spice comes from.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HiddenVariable:
    """A hypothesized hidden variable - a sietch waiting to be found."""

    name: str
    description: str
    hypothesis: str  # Why we think this variable matters

    # How to compute it
    computation_code: str  # Python code to compute from program/metrics
    computation_signature: str  # e.g., "def compute_X(metrics: dict, code: str) -> float"

    # Validation requirements
    expected_correlation: str  # "positive", "negative", "nonlinear"
    validation_criteria: str

    # Metadata
    source: str  # "pattern_mining", "llm_hypothesis", "domain_analysis"
    confidence: float  # 0-1, how confident are we this is real?

    # After validation
    is_validated: bool = False
    actual_correlation: float = 0.0


@dataclass
class SietchFinderConfig:
    """Configuration for the SietchFinder."""

    # Hypothesis generation
    max_hypotheses_per_round: int = 5

    # Sources to use
    use_pattern_mining: bool = True
    use_llm_hypothesis: bool = True
    use_domain_templates: bool = True

    # Confidence thresholds
    min_confidence_to_test: float = 0.3


class SietchFinder:
    """
    The SietchFinder - discovers hidden variables in the problem space.

    "Deep in the human unconscious is a pervasive need for a logical universe
    that makes sense. But the real universe is always one step beyond logic."
    """

    def __init__(
        self,
        config: SietchFinderConfig | None = None,
        llm_ensemble: Any | None = None,
        domain_context: str = "",
    ):
        self.config = config or SietchFinderConfig()
        self.llm_ensemble = llm_ensemble
        self.domain_context = domain_context

        self.discovered_sietches: list[HiddenVariable] = []

    async def find_hidden_variables(
        self,
        patterns: list[Any],  # ExtractedPattern from Mentat
        programs: list[dict[str, Any]],
        current_metrics: list[str],
    ) -> list[HiddenVariable]:
        """
        Find hidden variables based on pattern mining results and domain knowledge.

        Args:
            patterns: Patterns discovered by Mentat
            programs: Program data for analysis
            current_metrics: Names of currently tracked metrics

        Returns:
            List of hypothesized hidden variables
        """
        logger.info("SietchFinder searching for hidden variables...")

        hypotheses = []

        # Method 1: Convert patterns to variables
        if self.config.use_pattern_mining and patterns:
            pattern_vars = self._patterns_to_variables(patterns, current_metrics)
            hypotheses.extend(pattern_vars)

        # Method 2: LLM hypothesis generation
        if self.config.use_llm_hypothesis and self.llm_ensemble:
            llm_vars = await self._llm_generate_hypotheses(patterns, programs, current_metrics)
            hypotheses.extend(llm_vars)

        # Method 3: Domain-specific templates
        if self.config.use_domain_templates:
            template_vars = self._apply_domain_templates(current_metrics)
            hypotheses.extend(template_vars)

        # Deduplicate and rank
        unique_hypotheses = self._deduplicate_hypotheses(hypotheses)

        # Keep top N by confidence
        unique_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        top_hypotheses = unique_hypotheses[: self.config.max_hypotheses_per_round]

        logger.info(f"SietchFinder generated {len(top_hypotheses)} hidden variable hypotheses")
        for h in top_hypotheses:
            logger.info(f"  - {h.name}: {h.description[:50]}... (conf={h.confidence:.2f})")

        self.discovered_sietches.extend(top_hypotheses)
        return top_hypotheses

    def _patterns_to_variables(
        self,
        patterns: list[Any],
        current_metrics: list[str],
    ) -> list[HiddenVariable]:
        """Convert discovered patterns into hidden variable hypotheses."""
        variables = []

        for pattern in patterns:
            # Skip if already in metrics
            if pattern.name in current_metrics:
                continue

            # Skip if low discriminative power
            if pattern.discriminative_power < 0.2:
                continue

            # Generate computation code
            computation_code = self._generate_computation_from_pattern(pattern)

            variable = HiddenVariable(
                name=f"pattern_{pattern.name}",
                description=f"Hidden variable derived from pattern: {pattern.description}",
                hypothesis=f"Programs with higher {pattern.name} tend to have {'higher' if pattern.correlation_with_fitness > 0 else 'lower'} fitness (corr={pattern.correlation_with_fitness:.3f})",
                computation_code=computation_code,
                computation_signature=f"def compute_pattern_{pattern.name}(code: str, metrics: dict) -> float",
                expected_correlation="positive"
                if pattern.correlation_with_fitness > 0
                else "negative",
                validation_criteria=f"Correlation with fitness should be > {abs(pattern.correlation_with_fitness) * 0.5:.2f}",
                source="pattern_mining",
                confidence=min(0.9, pattern.discriminative_power + 0.3),
            )
            variables.append(variable)

        return variables

    def _generate_computation_from_pattern(self, pattern: Any) -> str:
        """Generate Python code to compute a pattern-based variable."""
        # Use the pattern's extraction code if available
        if hasattr(pattern, "extraction_code") and pattern.extraction_code:
            return pattern.extraction_code

        # Generate generic code based on pattern type
        if pattern.pattern_type == "structural":
            return f'''
def compute_pattern_{pattern.name}(code: str, metrics: dict) -> float:
    """Compute {pattern.name} from program code."""
    import ast
    try:
        tree = ast.parse(code)
        # Pattern-specific extraction logic
        # This is a placeholder - actual logic depends on pattern
        return 0.0
    except:
        return 0.0
'''
        elif pattern.pattern_type == "numeric":
            return f'''
def compute_pattern_{pattern.name}(code: str, metrics: dict) -> float:
    """Compute {pattern.name} from numeric patterns in code."""
    import re
    nums = re.findall(r'[-+]?\\d*\\.?\\d+', code)
    nums = [float(n) for n in nums if n]
    if not nums:
        return 0.0
    # Pattern-specific computation
    return 0.0
'''
        else:
            return f'''
def compute_pattern_{pattern.name}(code: str, metrics: dict) -> float:
    """Compute {pattern.name}."""
    # Auto-generated stub
    return 0.0
'''

    async def _llm_generate_hypotheses(
        self,
        patterns: list[Any],
        programs: list[dict[str, Any]],
        current_metrics: list[str],
    ) -> list[HiddenVariable]:
        """Use LLM to generate hypotheses about hidden variables."""
        if not self.llm_ensemble:
            return []

        # Prepare context for LLM
        pattern_summary = (
            "\n".join(
                [
                    f"- {p.name}: correlation={p.correlation_with_fitness:.3f}, discrimination={p.discriminative_power:.3f}"
                    for p in patterns[:10]
                ]
            )
            if patterns
            else "No patterns discovered yet."
        )

        metrics_list = ", ".join(current_metrics[:20])

        # Sample successful and unsuccessful programs
        if programs:
            programs_sorted = sorted(programs, key=lambda p: p["fitness"], reverse=True)
            successful = programs_sorted[:3]
            unsuccessful = programs_sorted[-3:]

            success_summary = "\n".join(
                [f"[fitness={p['fitness']:.3f}] Length: {len(p['code'])} chars" for p in successful]
            )
            fail_summary = "\n".join(
                [
                    f"[fitness={p['fitness']:.3f}] Length: {len(p['code'])} chars"
                    for p in unsuccessful
                ]
            )
        else:
            success_summary = "No data"
            fail_summary = "No data"

        prompt = f"""You are a scientific discovery system analyzing an evolutionary optimization process.

DOMAIN CONTEXT:
{self.domain_context}

CURRENT METRICS BEING TRACKED:
{metrics_list}

DISCOVERED PATTERNS (from code analysis):
{pattern_summary}

SUCCESSFUL PROGRAMS:
{success_summary}

UNSUCCESSFUL PROGRAMS:
{fail_summary}

TASK: Hypothesize 3-5 HIDDEN VARIABLES that might explain why some programs succeed and others fail.

These should be variables that:
1. Are NOT already in the current metrics
2. Could explain the patterns we're seeing
3. Are computable from the program code or execution results
4. Represent genuine domain-relevant quantities

For each hidden variable, provide:
1. NAME: A snake_case variable name
2. HYPOTHESIS: Why this variable might matter
3. COMPUTATION: How to compute it (Python pseudocode)
4. EXPECTED_CORRELATION: positive/negative/nonlinear

Format your response as:
---
NAME: variable_name
HYPOTHESIS: Why this matters...
COMPUTATION:
```python
def compute_variable_name(code: str, metrics: dict) -> float:
    # computation logic
    return value
```
EXPECTED_CORRELATION: positive
---
"""

        try:
            response = await self.llm_ensemble.generate(prompt)
            variables = self._parse_llm_hypotheses(response)
            return variables
        except Exception as e:
            logger.error(f"LLM hypothesis generation failed: {e}")
            return []

    def _parse_llm_hypotheses(self, response: str) -> list[HiddenVariable]:
        """Parse LLM response into HiddenVariable objects."""
        variables = []

        # Split by delimiter
        sections = response.split("---")

        for section in sections:
            if not section.strip():
                continue

            try:
                # Extract fields
                name_match = re.search(r"NAME:\s*(\w+)", section)
                hypothesis_match = re.search(
                    r"HYPOTHESIS:\s*(.+?)(?=COMPUTATION:|$)", section, re.DOTALL
                )
                code_match = re.search(r"```python\s*(.+?)\s*```", section, re.DOTALL)
                corr_match = re.search(r"EXPECTED_CORRELATION:\s*(\w+)", section)

                if name_match and hypothesis_match:
                    name = name_match.group(1)
                    hypothesis = hypothesis_match.group(1).strip()
                    computation = code_match.group(1).strip() if code_match else ""
                    correlation = corr_match.group(1).lower() if corr_match else "positive"

                    variable = HiddenVariable(
                        name=name,
                        description=hypothesis[:100],
                        hypothesis=hypothesis,
                        computation_code=computation,
                        computation_signature=f"def compute_{name}(code: str, metrics: dict) -> float",
                        expected_correlation=correlation,
                        validation_criteria="Significant correlation with fitness",
                        source="llm_hypothesis",
                        confidence=0.5,  # Medium confidence for LLM hypotheses
                    )
                    variables.append(variable)
            except Exception as e:
                logger.debug(f"Failed to parse hypothesis section: {e}")
                continue

        return variables

    def _apply_domain_templates(
        self,
        current_metrics: list[str],
    ) -> list[HiddenVariable]:
        """Apply domain-specific templates for common hidden variables."""

        # Generic templates that often reveal hidden structure
        templates = [
            HiddenVariable(
                name="parameter_entropy",
                description="Entropy of numeric parameters in the solution",
                hypothesis="Solutions with well-distributed parameters may be more robust",
                computation_code="""
def compute_parameter_entropy(code: str, metrics: dict) -> float:
    import re
    import numpy as np
    nums = re.findall(r'[-+]?\\d*\\.?\\d+', code)
    nums = [float(n) for n in nums if n]
    if len(nums) < 2:
        return 0.0
    # Normalize to probabilities
    nums = np.abs(nums)
    nums = nums / (np.sum(nums) + 1e-10)
    # Compute entropy
    entropy = -np.sum(nums * np.log(nums + 1e-10))
    return float(entropy)
""",
                computation_signature="def compute_parameter_entropy(code: str, metrics: dict) -> float",
                expected_correlation="nonlinear",
                validation_criteria="Should show inverted-U relationship with fitness",
                source="domain_template",
                confidence=0.4,
            ),
            HiddenVariable(
                name="structural_balance",
                description="Balance between different code structures",
                hypothesis="Well-balanced code structure may indicate better design",
                computation_code="""
def compute_structural_balance(code: str, metrics: dict) -> float:
    import ast
    try:
        tree = ast.parse(code)
        funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        loops = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
        conds = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
        total = funcs + loops + conds + 1
        # Compute balance (lower variance = more balanced)
        props = [funcs/total, loops/total, conds/total]
        variance = sum((p - 1/3)**2 for p in props)
        return 1.0 / (1.0 + variance * 10)
    except:
        return 0.5
""",
                computation_signature="def compute_structural_balance(code: str, metrics: dict) -> float",
                expected_correlation="positive",
                validation_criteria="Correlation > 0.2",
                source="domain_template",
                confidence=0.35,
            ),
            HiddenVariable(
                name="golden_ratio_proximity",
                description="How close numeric ratios are to golden ratio",
                hypothesis="Natural/optimal structures often exhibit golden ratio relationships",
                computation_code="""
def compute_golden_ratio_proximity(code: str, metrics: dict) -> float:
    import re
    import numpy as np
    PHI = 1.618033988749895
    nums = re.findall(r'[-+]?\\d*\\.?\\d+', code)
    nums = [abs(float(n)) for n in nums if n and float(n) != 0]
    if len(nums) < 2:
        return 0.0
    # Check ratios between consecutive values
    ratios = []
    for i in range(len(nums)-1):
        if nums[i+1] != 0:
            r = nums[i] / nums[i+1]
            if r > 1:
                r = 1/r
            ratios.append(r)
    if not ratios:
        return 0.0
    # How close are ratios to 1/PHI?
    target = 1/PHI
    distances = [abs(r - target) for r in ratios]
    avg_dist = np.mean(distances)
    return float(1.0 / (1.0 + avg_dist * 5))
""",
                computation_signature="def compute_golden_ratio_proximity(code: str, metrics: dict) -> float",
                expected_correlation="positive",
                validation_criteria="Correlation > 0.15",
                source="domain_template",
                confidence=0.3,
            ),
        ]

        # Filter out templates that overlap with current metrics
        return [t for t in templates if t.name not in current_metrics]

    def _deduplicate_hypotheses(
        self,
        hypotheses: list[HiddenVariable],
    ) -> list[HiddenVariable]:
        """Remove duplicate hypotheses based on name similarity."""
        seen_names = set()
        unique = []

        for h in hypotheses:
            # Normalize name
            normalized = h.name.lower().replace("_", "")
            if normalized not in seen_names:
                seen_names.add(normalized)
                unique.append(h)

        return unique
