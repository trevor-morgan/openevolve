"""
Discovery Engine for OpenEvolve

This module implements the Open-Ended Scientific Discovery architecture:
- ProblemEvolver: Evolves the problem space (questions, not just answers)
- AdversarialSkeptic: Falsification-based evaluation (not LLM-as-Judge)
- EpistemicArchive: Behavioral diversity (MAP-Elites with phenotype tracking)
- HeisenbergEngine: Ontological expansion (discover new variables)

The Heisenberg Engine enables automatic discovery of hidden variables when
optimization is stuck due to missing state space dimensions.
"""

from openevolve.discovery.code_instrumenter import (
    CodeInstrumenter,
    InstrumentationResult,
)

# Collaborative Discovery (Multi-Agent Novel Physics)
from openevolve.discovery.collaborative_discovery import (
    AgentRole,
    CollaborativeDiscovery,
    CollaborativeDiscoveryConfig,
    Critique,
    DiscoveryAgent,
    Idea,
    Synthesis,
)
from openevolve.discovery.crisis_detector import (
    CrisisDetector,
    CrisisDetectorConfig,
    EpistemicCrisis,
)
from openevolve.discovery.engine import DiscoveryEngine
from openevolve.discovery.epistemic_archive import EpistemicArchive, Phenotype
from openevolve.discovery.instrument_synthesizer import (
    InstrumentSynthesizer,
    InstrumentSynthesizerConfig,
    Probe,
    ProbeResult,
)

# Heisenberg Engine components (Ontological Expansion)
from openevolve.discovery.ontology import (
    Ontology,
    OntologyManager,
    Variable,
)
from openevolve.discovery.problem_archive import ProblemArchive, ProblemArchiveConfig
from openevolve.discovery.problem_space import ProblemEvolver, ProblemSpace
from openevolve.discovery.skeptic import AdversarialSkeptic

__all__ = [
    # Core Discovery Engine
    "ProblemSpace",
    "ProblemEvolver",
    "ProblemArchive",
    "ProblemArchiveConfig",
    "AdversarialSkeptic",
    "EpistemicArchive",
    "Phenotype",
    "DiscoveryEngine",
    # Heisenberg Engine (Ontological Expansion)
    "Variable",
    "Ontology",
    "OntologyManager",
    "EpistemicCrisis",
    "CrisisDetector",
    "CrisisDetectorConfig",
    "Probe",
    "ProbeResult",
    "InstrumentSynthesizer",
    "InstrumentSynthesizerConfig",
    "CodeInstrumenter",
    "InstrumentationResult",
    # Collaborative Discovery (Multi-Agent)
    "CollaborativeDiscovery",
    "CollaborativeDiscoveryConfig",
    "DiscoveryAgent",
    "AgentRole",
    "Idea",
    "Critique",
    "Synthesis",
]
