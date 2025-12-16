"""
Wolfram Alpha Tool - Query for analytical solutions and mathematical insights.

"Mathematics is the language in which God has written the universe." - Galileo

Uses Wolfram Alpha to:
- Find closed-form solutions to optimization problems
- Get mathematical identities that might help
- Query for physics formulas relevant to the domain
"""

import logging
import re
from typing import Any, ClassVar
from urllib.parse import quote_plus

import numpy as np

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class WolframTool(DiscoveryTool):
    """
    Queries Wolfram Alpha for mathematical insights.

    Requires a Wolfram Alpha API key (set in config).
    Falls back to simple mathematical analysis if no API key.
    """

    name = "wolfram"
    description = "Query Wolfram Alpha for mathematical solutions and physics formulas"
    discovery_types: ClassVar[list[DiscoveryType]] = [
        DiscoveryType.ANALYTICAL_SOLUTION,
        DiscoveryType.HYPOTHESIS,
    ]
    dependencies: ClassVar[list[str]] = ["aiohttp"]

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.api_key = config.get("wolfram_api_key") if config else None

    def _check_dependencies(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("aiohttp") is not None

    async def discover(self, context: ToolContext) -> list[Discovery]:
        """Query Wolfram Alpha for insights."""
        logger.info("Running Wolfram Alpha analysis...")

        discoveries = []

        # Extract mathematical questions from domain context
        queries = self._generate_queries(context.domain_context, context.current_metrics)

        if self.api_key:
            # Use Wolfram Alpha API
            for query in queries[:5]:
                result = await self._query_wolfram(query)
                if result:
                    discoveries.append(result)
        else:
            # Fallback: Generate mathematical hypotheses without API
            discoveries = self._generate_math_hypotheses(context)

        logger.info(f"Wolfram tool generated {len(discoveries)} insights")
        return discoveries

    def _generate_queries(self, domain_context: str, metrics: list[str]) -> list[str]:
        """Generate queries for Wolfram Alpha."""
        queries = []

        # Domain-specific queries
        context_lower = domain_context.lower()

        if "magnetic" in context_lower:
            queries.extend(
                [
                    "magnetic field of circular current loop",
                    "mirror ratio plasma confinement formula",
                    "minimum-B stability criterion",
                ]
            )

        if "optimization" in context_lower:
            queries.extend(
                [
                    "gradient descent convergence conditions",
                    "convex optimization optimal solution",
                ]
            )

        # Metric-based queries
        for metric in metrics[:3]:
            queries.append(f"optimize {metric}")

        # General mathematical queries
        queries.extend(
            [
                "maximize function with constraints",
                "saddle point conditions",
            ]
        )

        return queries

    async def _query_wolfram(self, query: str) -> Discovery | None:
        """Query Wolfram Alpha API."""
        import aiohttp

        url = f"http://api.wolframalpha.com/v2/query?input={quote_plus(query)}&appid={self.api_key}&output=json"

        try:
            async with aiohttp.ClientSession() as session, session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_wolfram_response(query, data)
        except Exception as e:
            logger.warning(f"Wolfram query failed: {e}")

        return None

    def _parse_wolfram_response(self, query: str, data: dict[str, Any]) -> Discovery | None:
        """Parse Wolfram Alpha API response."""
        try:
            pods = data.get("queryresult", {}).get("pods", [])

            results = []
            for pod in pods:
                title = pod.get("title", "")
                subpods = pod.get("subpods", [])

                for subpod in subpods:
                    plaintext = subpod.get("plaintext", "")
                    if plaintext:
                        results.append(f"{title}: {plaintext}")

            if results:
                return Discovery(
                    name=f"wolfram_{self._slugify(query)}",
                    description=f"Wolfram Alpha result for: {query}",
                    discovery_type=DiscoveryType.ANALYTICAL_SOLUTION,
                    content={
                        "query": query,
                        "results": results[:5],
                        "source": "wolfram_alpha",
                    },
                    confidence=0.7,
                    evidence=results[:3],
                )
        except Exception as e:
            logger.debug(f"Failed to parse Wolfram response: {e}")

        return None

    def _generate_math_hypotheses(self, context: ToolContext) -> list[Discovery]:
        """Generate mathematical hypotheses without API."""
        import numpy as np

        discoveries = []

        programs = context.programs
        if len(programs) < 10:
            return []

        fitnesses = np.array([p.get("fitness", 0.0) for p in programs])

        # Analyze fitness distribution
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        skew = self._compute_skew(fitnesses)

        # Hypothesis based on distribution
        if skew > 0.5:
            discoveries.append(
                Discovery(
                    name="math_skewed_distribution",
                    description=f"Fitness is right-skewed (skew={skew:.2f}): most programs are low fitness, few high",
                    discovery_type=DiscoveryType.HYPOTHESIS,
                    content={
                        "statistic": "skewness",
                        "value": float(skew),
                        "interpretation": "The fitness landscape has rare high-performance regions",
                        "suggestion": "Focus exploration on high-fitness neighbors",
                    },
                    confidence=0.6,
                    evidence=[
                        f"Skewness: {skew:.3f}",
                        f"Mean: {mean_fitness:.3f}, Std: {std_fitness:.3f}",
                    ],
                )
            )
        elif skew < -0.5:
            discoveries.append(
                Discovery(
                    name="math_ceiling_effect",
                    description=f"Fitness shows ceiling effect (skew={skew:.2f}): approaching maximum",
                    discovery_type=DiscoveryType.HYPOTHESIS,
                    content={
                        "statistic": "skewness",
                        "value": float(skew),
                        "interpretation": "Evolution is hitting a ceiling - need new dimensions",
                        "suggestion": "Search for orthogonal improvements",
                    },
                    confidence=0.6,
                    evidence=[
                        f"Skewness: {skew:.3f}",
                        f"Mean: {mean_fitness:.3f}",
                    ],
                )
            )

        # Check for multimodality (multiple local optima)
        hist, _bin_edges = np.histogram(fitnesses, bins=10)
        peaks = self._find_peaks(hist)

        if len(peaks) > 1:
            discoveries.append(
                Discovery(
                    name="math_multimodal",
                    description=f"Fitness is multimodal ({len(peaks)} modes): multiple local optima exist",
                    discovery_type=DiscoveryType.HYPOTHESIS,
                    content={
                        "statistic": "modality",
                        "num_modes": len(peaks),
                        "interpretation": "The landscape has multiple basins of attraction",
                        "suggestion": "Maintain population diversity across modes",
                    },
                    confidence=0.5,
                    evidence=[
                        f"Number of modes: {len(peaks)}",
                        f"Peak bins: {peaks}",
                    ],
                )
            )

        # Mathematical relationships to explore
        discoveries.append(
            Discovery(
                name="math_sqrt_transform",
                description="Try sqrt transformation of fitness for better optimization landscape",
                discovery_type=DiscoveryType.HYPOTHESIS,
                content={
                    "transform": "sqrt(fitness)",
                    "rationale": "Square root often stabilizes variance and normalizes skewed distributions",
                    "suggestion": "Consider optimizing sqrt(fitness) instead of raw fitness",
                },
                computation_code='''
def compute_sqrt_fitness(code: str, metrics: dict) -> float:
    """Apply sqrt transformation to fitness."""
    import numpy as np
    fitness = metrics.get("fitness", 0.0)
    return np.sqrt(max(0, fitness))
''',
                confidence=0.4,
                evidence=["Mathematical heuristic"],
            )
        )

        discoveries.append(
            Discovery(
                name="math_log_transform",
                description="Try log transformation for multiplicative relationships",
                discovery_type=DiscoveryType.HYPOTHESIS,
                content={
                    "transform": "log(1 + fitness)",
                    "rationale": "Log transform reveals multiplicative relationships",
                    "suggestion": "Look for features that multiply rather than add",
                },
                computation_code='''
def compute_log_fitness(code: str, metrics: dict) -> float:
    """Apply log transformation to fitness."""
    import numpy as np
    fitness = metrics.get("fitness", 0.0)
    return np.log1p(max(0, fitness))
''',
                confidence=0.4,
                evidence=["Mathematical heuristic"],
            )
        )

        return discoveries

    def _compute_skew(self, data: "np.ndarray") -> float:
        """Compute skewness of distribution."""
        import numpy as np

        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return np.sum(((data - mean) / std) ** 3) / n

    def _find_peaks(self, hist: "np.ndarray") -> list[int]:
        """Find peaks in histogram."""
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append(i)
        return peaks

    def _slugify(self, text: str) -> str:
        """Convert text to slug."""
        return re.sub(r"[^a-z0-9]+", "_", text.lower())[:30]
