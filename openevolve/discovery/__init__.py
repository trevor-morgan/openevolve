"""
Discovery Engine for OpenEvolve

This module implements the Open-Ended Scientific Discovery architecture:
- ProblemEvolver: Evolves the problem space (questions, not just answers)
- AdversarialSkeptic: Falsification-based evaluation (not LLM-as-Judge)
- EpistemicArchive: Behavioral diversity (MAP-Elites with phenotype tracking)
"""

from openevolve.discovery.problem_space import ProblemSpace, ProblemEvolver
from openevolve.discovery.skeptic import AdversarialSkeptic
from openevolve.discovery.epistemic_archive import EpistemicArchive
from openevolve.discovery.engine import DiscoveryEngine

__all__ = [
    "ProblemSpace",
    "ProblemEvolver",
    "AdversarialSkeptic",
    "EpistemicArchive",
    "DiscoveryEngine",
]
