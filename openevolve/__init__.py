"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

from openevolve._version import __version__
from openevolve.controller import OpenEvolve
from openevolve.api import (
    run_evolution,
    run_discovery,
    evolve_function,
    evolve_algorithm,
    evolve_code,
    EvolutionResult
)
from openevolve.config import Config, DiscoveryConfig, SkepticConfig

# Discovery Mode imports (optional - may not be installed)
try:
    from openevolve.discovery import (
        DiscoveryEngine,
        ProblemSpace,
        ProblemEvolver,
        AdversarialSkeptic,
        EpistemicArchive,
    )
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False

__all__ = [
    # Core
    "OpenEvolve",
    "__version__",
    # High-level API
    "run_evolution",
    "run_discovery",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
    # Configuration
    "Config",
    "DiscoveryConfig",
    "SkepticConfig",
    # Discovery Mode
    "DiscoveryEngine",
    "ProblemSpace",
    "ProblemEvolver",
    "AdversarialSkeptic",
    "EpistemicArchive",
    "DISCOVERY_AVAILABLE",
]
