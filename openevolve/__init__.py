"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

from openevolve._version import __version__
from openevolve.api import (
    EvolutionResult,
    evolve_algorithm,
    evolve_code,
    evolve_function,
    run_discovery,
    run_evolution,
)
from openevolve.config import Config, DiscoveryConfig, SkepticConfig
from openevolve.controller import OpenEvolve

# Discovery Mode imports (optional - may not be installed)
try:
    from openevolve.discovery import (
        AdversarialSkeptic,
        DiscoveryEngine,
        EpistemicArchive,
        ProblemEvolver,
        ProblemSpace,
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
