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

from openevolve.discovery.problem_space import ProblemSpace, ProblemEvolver
from openevolve.discovery.skeptic import AdversarialSkeptic
from openevolve.discovery.epistemic_archive import EpistemicArchive, Phenotype
from openevolve.discovery.engine import DiscoveryEngine

# Heisenberg Engine components (Ontological Expansion)
from openevolve.discovery.ontology import (
    Variable,
    Ontology,
    OntologyManager,
)
from openevolve.discovery.crisis_detector import (
    EpistemicCrisis,
    CrisisDetector,
    CrisisDetectorConfig,
)
from openevolve.discovery.instrument_synthesizer import (
    Probe,
    ProbeResult,
    InstrumentSynthesizer,
    InstrumentSynthesizerConfig,
)
from openevolve.discovery.code_instrumenter import (
    CodeInstrumenter,
    InstrumentationResult,
)

__all__ = [
    # Core Discovery Engine
    "ProblemSpace",
    "ProblemEvolver",
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
]
