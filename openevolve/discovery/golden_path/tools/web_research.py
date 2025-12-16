"""
Web Research Tool - Search scientific literature for insights.

"The beginning of knowledge is the discovery of something we do not understand."

Searches arXiv, Semantic Scholar, and other sources to find:
- Relevant research papers
- Known techniques for the problem domain
- Hidden variables discovered by others
- Novel approaches from related fields

This helps discover things humans HAVE named but we didn't know about.
"""

import asyncio
import logging
import re
from typing import Any, ClassVar
from urllib.parse import quote_plus

from ..toolkit import Discovery, DiscoveryTool, DiscoveryType, ToolContext

logger = logging.getLogger(__name__)


class WebResearchTool(DiscoveryTool):
    """
    Searches scientific literature for relevant insights.

    Uses:
    - arXiv API (free, no auth)
    - Semantic Scholar API (free tier, no auth)
    - Optional: Google Scholar (via serpapi if available)
    """

    name = "web_research"
    description = "Search scientific literature (arXiv, Semantic Scholar) for relevant research and techniques"
    discovery_types: ClassVar[list[DiscoveryType]] = [
        DiscoveryType.RESEARCH_INSIGHT,
        DiscoveryType.HYPOTHESIS,
    ]
    dependencies: ClassVar[list[str]] = ["aiohttp"]

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._aiohttp = None

    def _check_dependencies(self) -> bool:
        try:
            import aiohttp

            self._aiohttp = aiohttp
            return True
        except ImportError:
            return False

    async def discover(self, context: ToolContext) -> list[Discovery]:
        """Search literature based on domain context."""
        logger.info("Running web research...")

        discoveries = []

        # Extract search terms from domain context
        search_terms = self._extract_search_terms(context.domain_context)

        if not search_terms:
            search_terms = ["optimization", "hidden variables", "machine learning"]

        logger.info(f"Searching with terms: {search_terms[:5]}")

        # Search multiple sources in parallel
        tasks = [
            self._search_arxiv(search_terms),
            self._search_semantic_scholar(search_terms),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                discoveries.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Search failed: {result}")

        # Deduplicate by title similarity
        discoveries = self._deduplicate(discoveries)

        logger.info(f"Web research found {len(discoveries)} relevant papers/insights")
        return discoveries[:10]  # Limit results

    def _extract_search_terms(self, domain_context: str) -> list[str]:
        """Extract meaningful search terms from domain context."""
        terms = []

        # Common domain-specific keywords
        keywords = [
            "plasma",
            "fusion",
            "magnetic",
            "confinement",
            "mirror",
            "optimization",
            "neural network",
            "genetic algorithm",
            "hidden variable",
            "latent factor",
            "causal",
            "symbolic regression",
            "evolutionary",
        ]

        context_lower = domain_context.lower()

        for kw in keywords:
            if kw in context_lower:
                terms.append(kw)

        # Extract capitalized phrases (likely important terms)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", domain_context)
        terms.extend(capitalized[:5])

        # Add domain-specific combinations
        if "plasma" in context_lower or "fusion" in context_lower:
            terms.extend(["plasma physics optimization", "magnetic mirror machine learning"])
        if "coil" in context_lower:
            terms.extend(["coil optimization", "magnetic field design"])

        return list(set(terms))[:10]

    async def _search_arxiv(self, terms: list[str]) -> list[Discovery]:
        """Search arXiv API."""
        import aiohttp

        discoveries = []

        # Build query
        query = " OR ".join(f'all:"{t}"' for t in terms[:5])
        url = f"http://export.arxiv.org/api/query?search_query={quote_plus(query)}&max_results=10&sortBy=relevance"

        try:
            async with aiohttp.ClientSession() as session, session.get(url, timeout=30) as response:
                if response.status == 200:
                    text = await response.text()
                    discoveries = self._parse_arxiv_response(text)
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")

        return discoveries

    def _parse_arxiv_response(self, xml_text: str) -> list[Discovery]:
        """Parse arXiv API XML response."""
        discoveries = []

        # Simple XML parsing without external deps
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for entry in entries[:5]:
            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link_match = re.search(r"<id>(.*?)</id>", entry)
            authors_match = re.findall(r"<name>(.*?)</name>", entry)

            if title_match and summary_match:
                title = title_match.group(1).strip().replace("\n", " ")
                summary = summary_match.group(1).strip().replace("\n", " ")
                link = link_match.group(1) if link_match else ""
                authors = authors_match[:3]  # First 3 authors

                # Extract potential insights from summary
                insights = self._extract_insights(summary)

                discoveries.append(
                    Discovery(
                        name=f"arxiv_{self._slugify(title[:30])}",
                        description=f"Paper: {title[:100]}...",
                        discovery_type=DiscoveryType.RESEARCH_INSIGHT,
                        content={
                            "source": "arxiv",
                            "title": title,
                            "authors": authors,
                            "summary": summary[:500],
                            "link": link,
                            "insights": insights,
                        },
                        confidence=0.5,
                        evidence=[
                            f"Title: {title[:80]}",
                            f"Authors: {', '.join(authors[:2])}",
                        ],
                        testable=False,  # Research insights need human interpretation
                    )
                )

        return discoveries

    async def _search_semantic_scholar(self, terms: list[str]) -> list[Discovery]:
        """Search Semantic Scholar API."""
        import aiohttp

        discoveries = []

        query = " ".join(terms[:3])
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(query)}&limit=5&fields=title,abstract,authors,citationCount,year,url"

        try:
            async with aiohttp.ClientSession() as session, session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    discoveries = self._parse_semantic_scholar_response(data)
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")

        return discoveries

    def _parse_semantic_scholar_response(self, data: dict[str, Any]) -> list[Discovery]:
        """Parse Semantic Scholar API response."""
        discoveries = []

        papers = data.get("data", [])

        for paper in papers[:5]:
            title = paper.get("title", "Unknown")
            abstract = paper.get("abstract", "")
            authors = [a.get("name", "") for a in paper.get("authors", [])[:3]]
            citations = paper.get("citationCount", 0)
            year = paper.get("year", 0)
            url = paper.get("url", "")

            if abstract:
                insights = self._extract_insights(abstract)

                # Higher confidence for highly cited recent papers
                confidence = min(0.8, 0.3 + (citations / 1000) + (0.1 if year >= 2020 else 0))

                discoveries.append(
                    Discovery(
                        name=f"ss_{self._slugify(title[:30])}",
                        description=f"Paper ({year}, {citations} citations): {title[:80]}",
                        discovery_type=DiscoveryType.RESEARCH_INSIGHT,
                        content={
                            "source": "semantic_scholar",
                            "title": title,
                            "authors": authors,
                            "abstract": abstract[:500],
                            "citations": citations,
                            "year": year,
                            "url": url,
                            "insights": insights,
                        },
                        confidence=confidence,
                        evidence=[
                            f"Citations: {citations}",
                            f"Year: {year}",
                            f"Authors: {', '.join(authors[:2])}",
                        ],
                        testable=False,
                    )
                )

        return discoveries

    def _extract_insights(self, text: str) -> list[str]:
        """Extract potential insights from paper abstract."""
        insights = []

        # Look for methodology mentions
        methods = re.findall(
            r"(?:we propose|we introduce|we develop|novel|new approach|method|technique)\s+([^.]+)",
            text.lower(),
        )
        insights.extend([f"Method: {m.strip()[:100]}" for m in methods[:2]])

        # Look for findings
        findings = re.findall(
            r"(?:we show|we demonstrate|results show|we find|our results)\s+([^.]+)", text.lower()
        )
        insights.extend([f"Finding: {f.strip()[:100]}" for f in findings[:2]])

        # Look for improvements
        improvements = re.findall(
            r"(?:improve|outperform|better than|superior to|achieve)\s+([^.]+)", text.lower()
        )
        insights.extend([f"Improvement: {i.strip()[:100]}" for i in improvements[:2]])

        return insights[:5]

    def _slugify(self, text: str) -> str:
        """Convert text to slug for naming."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())
        return slug.strip("_")[:30]

    def _deduplicate(self, discoveries: list[Discovery]) -> list[Discovery]:
        """Remove duplicate discoveries based on title similarity."""
        seen_titles = set()
        unique = []

        for d in discoveries:
            title = d.content.get("title", "").lower()[:50]
            if title not in seen_titles:
                seen_titles.add(title)
                unique.append(d)

        return unique
