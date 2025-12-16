"""
DiscoveryToolkit - Orchestrator for External Discovery Tools

"He who controls the spice controls the universe."

The toolkit allows the Golden Path to leverage external tools and frameworks
for TRUE ontological discovery - finding patterns and relationships that
don't exist in any textbook.

Instead of implementing everything ourselves, we orchestrate:
- Symbolic regression (PySR, gplearn) - discover mathematical formulas
- Causal discovery (DoWhy, causal-learn) - find causal structures
- Research tools (arXiv, Semantic Scholar) - find relevant work
- Computation (Wolfram Alpha, SymPy) - analytical solutions
- Code analysis (tree-sitter, AST) - structural patterns

The LLM decides which tools to invoke based on the discovery context.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class DiscoveryType(Enum):
    """Types of discoveries that tools can produce."""

    MATHEMATICAL_FORMULA = "mathematical_formula"  # y = f(x) relationship
    CAUSAL_RELATIONSHIP = "causal_relationship"  # X causes Y
    LATENT_VARIABLE = "latent_variable"  # Hidden dimension
    RESEARCH_INSIGHT = "research_insight"  # From literature
    CODE_PATTERN = "code_pattern"  # Structural pattern
    ANALYTICAL_SOLUTION = "analytical_solution"  # Closed-form solution
    HYPOTHESIS = "hypothesis"  # Testable idea


@dataclass
class Discovery:
    """A discovery produced by a tool."""

    name: str
    description: str
    discovery_type: DiscoveryType

    # The actual discovery content
    content: dict[str, Any]  # Tool-specific content

    # If this discovery can be computed at runtime
    computation_code: str | None = None

    # Confidence and evidence
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)

    # Source tool
    source_tool: str = ""

    # For validation
    testable: bool = True
    validation_criteria: str | None = None


@dataclass
class ToolContext:
    """Context passed to discovery tools."""

    # Programs to analyze
    programs: list[dict[str, Any]]

    # Current metrics being tracked
    current_metrics: list[str]

    # Domain description
    domain_context: str

    # Crisis information (why are we discovering?)
    crisis_type: str = ""
    crisis_details: dict[str, Any] = field(default_factory=dict)

    # Best program info
    best_fitness: float = 0.0
    best_program_code: str = ""

    # What we're looking for
    discovery_goal: str = "Find hidden variables that explain fitness variance"


class DiscoveryTool(ABC):
    """
    Base interface for external discovery tools.

    Subclass this to integrate any discovery tool/framework.
    """

    name: str = "base_tool"
    description: str = "Base discovery tool"
    discovery_types: ClassVar[list[DiscoveryType]] = []

    # Dependencies - what packages need to be installed
    dependencies: ClassVar[list[str]] = []

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if tool dependencies are installed."""
        if self._available is not None:
            return self._available

        self._available = self._check_dependencies()
        return self._available

    def _check_dependencies(self) -> bool:
        """Check if all dependencies are available."""
        for dep in self.dependencies:
            try:
                __import__(dep)
            except ImportError:
                logger.debug(f"{self.name}: missing dependency {dep}")
                return False
        return True

    @abstractmethod
    async def discover(self, context: ToolContext) -> list[Discovery]:
        """
        Run discovery and return findings.

        Args:
            context: ToolContext with programs, metrics, domain info

        Returns:
            List of Discovery objects
        """
        raise NotImplementedError

    def get_info(self) -> dict[str, Any]:
        """Get tool information for LLM selection."""
        return {
            "name": self.name,
            "description": self.description,
            "discovery_types": [dt.value for dt in self.discovery_types],
            "available": self.is_available(),
            "dependencies": self.dependencies,
        }


class DiscoveryToolkit:
    """
    Orchestrates multiple discovery tools.

    The toolkit:
    1. Registers available tools
    2. Lets LLM select which tools to use
    3. Runs tools in parallel
    4. Aggregates and deduplicates results
    """

    def __init__(self, llm_ensemble: Any | None = None):
        self.tools: dict[str, DiscoveryTool] = {}
        self.llm_ensemble = llm_ensemble
        self._discovery_history: list[Discovery] = []

    def register_tool(self, tool: DiscoveryTool) -> None:
        """Register a discovery tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered discovery tool: {tool.name} (available={tool.is_available()})")

    def register_tools(self, tools: list[DiscoveryTool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register_tool(tool)

    def get_available_tools(self) -> list[DiscoveryTool]:
        """Get list of available (installed) tools."""
        return [t for t in self.tools.values() if t.is_available()]

    def get_tools_info(self) -> list[dict[str, Any]]:
        """Get info about all tools for LLM selection."""
        return [t.get_info() for t in self.tools.values()]

    async def run_tool(self, tool_name: str, context: ToolContext) -> list[Discovery]:
        """Run a specific tool."""
        if tool_name not in self.tools:
            logger.warning(f"Unknown tool: {tool_name}")
            return []

        tool = self.tools[tool_name]
        if not tool.is_available():
            logger.warning(f"Tool not available: {tool_name}")
            return []

        try:
            discoveries = await tool.discover(context)
            for d in discoveries:
                d.source_tool = tool_name
            self._discovery_history.extend(discoveries)
            return discoveries
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return []

    async def run_tools(
        self,
        tool_names: list[str],
        context: ToolContext,
        parallel: bool = True,
    ) -> list[Discovery]:
        """Run multiple tools, optionally in parallel."""
        if parallel:
            tasks = [self.run_tool(name, context) for name in tool_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            discoveries = []
            for r in results:
                if isinstance(r, list):
                    discoveries.extend(r)
                elif isinstance(r, Exception):
                    logger.error(f"Tool error: {r}")
            return discoveries
        else:
            discoveries = []
            for name in tool_names:
                discoveries.extend(await self.run_tool(name, context))
            return discoveries

    async def run_all_available(self, context: ToolContext) -> list[Discovery]:
        """Run all available tools."""
        available = self.get_available_tools()
        return await self.run_tools([t.name for t in available], context)

    async def select_and_run(self, context: ToolContext) -> list[Discovery]:
        """
        Use LLM to select appropriate tools, then run them.

        This is the smart orchestration mode - the LLM decides
        which tools are most likely to help given the context.
        """
        if not self.llm_ensemble:
            # Fall back to running all available
            logger.info("No LLM for tool selection, running all available tools")
            return await self.run_all_available(context)

        # Build prompt for tool selection
        tools_info = self.get_tools_info()
        available_tools = [t for t in tools_info if t["available"]]

        if not available_tools:
            logger.warning("No discovery tools available")
            return []

        selection_prompt = f"""You are selecting discovery tools for scientific exploration.

CONTEXT:
- Domain: {context.domain_context[:500]}...
- Crisis: {context.crisis_type}
- Current metrics: {context.current_metrics}
- Goal: {context.discovery_goal}
- Number of programs to analyze: {len(context.programs)}

AVAILABLE TOOLS:
{self._format_tools_for_llm(available_tools)}

Select the most appropriate tools for this discovery task.
Return a JSON list of tool names, e.g.: ["symbolic_regression", "causal_discovery"]

Consider:
1. What type of discovery would help most?
2. Which tools can find patterns humans haven't named?
3. Which tools complement each other?

Selected tools (JSON list):"""

        try:
            response = await self.llm_ensemble.generate(
                selection_prompt,
                system_message="You are a scientific discovery coordinator. Select tools wisely.",
            )

            # Parse tool names from response
            import json
            import re

            # Try to find JSON list in response
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                selected = json.loads(match.group())
                logger.info(f"LLM selected tools: {selected}")
                return await self.run_tools(selected, context)
            else:
                logger.warning(f"Could not parse tool selection: {response}")
                return await self.run_all_available(context)

        except Exception as e:
            logger.error(f"Tool selection failed: {e}, running all available")
            return await self.run_all_available(context)

    def _format_tools_for_llm(self, tools: list[dict[str, Any]]) -> str:
        """Format tool info for LLM prompt."""
        lines = []
        for t in tools:
            types = ", ".join(t["discovery_types"])
            lines.append(f"- {t['name']}: {t['description']}")
            lines.append(f"  Discovery types: {types}")
        return "\n".join(lines)

    def get_discovery_history(self) -> list[Discovery]:
        """Get all discoveries made so far."""
        return self._discovery_history.copy()

    def clear_history(self) -> None:
        """Clear discovery history."""
        self._discovery_history = []


# =============================================================================
# Tool Registration Helper
# =============================================================================


def create_default_toolkit(llm_ensemble: Any | None = None) -> DiscoveryToolkit:
    """
    Create a toolkit with all available default tools.

    This imports and registers all built-in tools that have
    their dependencies installed.
    """
    toolkit = DiscoveryToolkit(llm_ensemble=llm_ensemble)

    # Import and register built-in tools
    # Each tool handles its own dependency checking
    try:
        from .tools.symbolic_regression import SymbolicRegressionTool

        toolkit.register_tool(SymbolicRegressionTool())
    except ImportError as e:
        logger.debug(f"Symbolic regression tool not available: {e}")

    try:
        from .tools.causal_discovery import CausalDiscoveryTool

        toolkit.register_tool(CausalDiscoveryTool())
    except ImportError as e:
        logger.debug(f"Causal discovery tool not available: {e}")

    try:
        from .tools.web_research import WebResearchTool

        toolkit.register_tool(WebResearchTool())
    except ImportError as e:
        logger.debug(f"Web research tool not available: {e}")

    try:
        from .tools.code_analysis import CodeAnalysisTool

        toolkit.register_tool(CodeAnalysisTool())
    except ImportError as e:
        logger.debug(f"Code analysis tool not available: {e}")

    try:
        from .tools.wolfram import WolframTool

        toolkit.register_tool(WolframTool())
    except ImportError as e:
        logger.debug(f"Wolfram tool not available: {e}")

    try:
        from .tools.topology import TopologyAnalysisTool

        toolkit.register_tool(TopologyAnalysisTool())
    except ImportError as e:
        logger.debug(f"Topology analysis tool not available: {e}")

    try:
        from .tools.physics_ml import PhysicsMLTool

        toolkit.register_tool(PhysicsMLTool())
    except ImportError as e:
        logger.debug(f"Physics ML tool not available: {e}")

    logger.info(f"Created toolkit with {len(toolkit.get_available_tools())} available tools")

    return toolkit
