"""
The Golden Path - Autonomous Ontological Discovery Framework

"The vision of time is broad, but when you pass through it,
time becomes a narrow door." - Leto II

This framework enables true ontological discovery - finding hidden variables
and patterns that don't exist in the current representation. Unlike parameter
optimization, the Golden Path discovers NEW dimensions of the problem space.

Architecture (Dune-themed):

    GoldenPath (Orchestrator)
        │
        ├── Prescience - Crisis detection, sees when evolution is truly stuck
        │
        ├── Mentat - Program mining, extracts patterns from successful programs
        │
        ├── SietchFinder - Discovers hidden variables (the "sietches" - hidden places)
        │
        ├── GomJabbar - Validates discoveries (the test that separates true from false)
        │
        ├── SpiceAgony - Transforms the ontology (the painful but necessary change)
        │
        └── DiscoveryToolkit - Orchestrates external discovery tools
            │
            ├── SymbolicRegressionTool (PySR/gplearn) - Discover mathematical formulas
            ├── CausalDiscoveryTool (DoWhy) - Find causal relationships
            ├── WebResearchTool (arXiv, Semantic Scholar) - Search literature
            ├── CodeAnalysisTool (AST) - Structural pattern analysis
            └── WolframTool - Mathematical insights

The Golden Path operates autonomously:
1. Prescience detects true plateaus (not just slow progress)
2. Mentat mines programs for patterns not in current ontology
3. DiscoveryToolkit runs external tools (symbolic regression, causal discovery, etc.)
4. SietchFinder proposes hidden variables from patterns and tool outputs
5. GomJabbar tests if these variables actually predict fitness
6. SpiceAgony injects validated variables into the evaluator

"The spice must flow" - but first, it must be discovered.
"""

from .golden_path import DiscoveryRound, GoldenPath, GoldenPathConfig
from .gom_jabbar import GomJabbar, GomJabbarConfig, ValidationResult
from .mentat import ExtractedPattern, Mentat, MentatConfig
from .prescience import CrisisType, Prescience, PrescienceConfig, PrescienceReading
from .sietch_finder import HiddenVariable, SietchFinder, SietchFinderConfig
from .spice_agony import OntologyTransformation, SpiceAgony, SpiceAgonyConfig
from .toolkit import (
    Discovery,
    DiscoveryTool,
    DiscoveryToolkit,
    DiscoveryType,
    ToolContext,
    create_default_toolkit,
)

__all__ = [
    # Core components
    "GoldenPath",
    "GoldenPathConfig",
    "DiscoveryRound",
    # Prescience
    "Prescience",
    "PrescienceConfig",
    "PrescienceReading",
    "CrisisType",
    # Mentat
    "Mentat",
    "MentatConfig",
    "ExtractedPattern",
    # SietchFinder
    "SietchFinder",
    "SietchFinderConfig",
    "HiddenVariable",
    # GomJabbar
    "GomJabbar",
    "GomJabbarConfig",
    "ValidationResult",
    # SpiceAgony
    "SpiceAgony",
    "SpiceAgonyConfig",
    "OntologyTransformation",
    # Toolkit
    "DiscoveryToolkit",
    "DiscoveryTool",
    "Discovery",
    "DiscoveryType",
    "ToolContext",
    "create_default_toolkit",
]
