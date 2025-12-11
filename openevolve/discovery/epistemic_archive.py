"""
Epistemic Archive - Behavioral Diversity Storage

This module enhances OpenEvolve's existing MAP-Elites implementation with:
1. Behavioral phenotype extraction (not just code metrics)
2. Surprise-based novelty tracking
3. Cross-problem knowledge transfer

Key insight: "Simple but Slow" and "Complex but Fast" solutions are BOTH valuable.
The simple solution might be the stepping stone for a harder problem.
"""

import ast
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openevolve.database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class Phenotype:
    """
    Behavioral characteristics of a solution.

    Unlike simple metrics (accuracy, speed), phenotypes describe
    HOW a solution behaves - its approach, trade-offs, and characteristics.
    """
    # Core behavioral dimensions
    complexity: int = 0  # AST complexity (structural)
    efficiency: int = 0  # Runtime efficiency (behavioral)
    robustness: int = 0  # Error handling quality
    generality: int = 0  # How general vs specialized

    # Derived characteristics
    approach_signature: str = ""  # Hash of structural approach
    uses_recursion: bool = False
    uses_iteration: bool = False
    uses_vectorization: bool = False
    uses_memoization: bool = False

    # Performance profile
    best_case_complexity: str = "O(?)"
    worst_case_complexity: str = "O(?)"
    space_complexity: str = "O(?)"

    def to_grid_coords(self, num_bins: int = 10) -> Tuple[int, int]:
        """Convert phenotype to 2D grid coordinates for MAP-Elites"""
        x = min(self.complexity, num_bins - 1)
        y = min(self.efficiency, num_bins - 1)
        return (x, y)

    def to_nd_coords(self, dimensions: List[str], num_bins: int = 10) -> Tuple[int, ...]:
        """Convert phenotype to N-dimensional coordinates"""
        coords = []
        for dim in dimensions:
            value = getattr(self, dim, 0)
            if isinstance(value, bool):
                value = 1 if value else 0
            elif isinstance(value, str):
                value = hash(value) % num_bins
            coords.append(min(int(value), num_bins - 1))
        return tuple(coords)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity,
            "efficiency": self.efficiency,
            "robustness": self.robustness,
            "generality": self.generality,
            "approach_signature": self.approach_signature,
            "uses_recursion": self.uses_recursion,
            "uses_iteration": self.uses_iteration,
            "uses_vectorization": self.uses_vectorization,
            "uses_memoization": self.uses_memoization,
            "best_case_complexity": self.best_case_complexity,
            "worst_case_complexity": self.worst_case_complexity,
            "space_complexity": self.space_complexity,
        }


class PhenotypeExtractor:
    """
    Extracts behavioral phenotypes from code.

    This goes beyond simple metrics to understand HOW code works,
    not just how well it performs on a benchmark.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def extract(self, code: str, metrics: Dict[str, float] = None) -> Phenotype:
        """Extract phenotype from code and optional metrics"""
        metrics = metrics or {}

        phenotype = Phenotype()

        # Structural analysis
        try:
            tree = ast.parse(code)
            phenotype = self._analyze_ast(tree, phenotype)
        except SyntaxError:
            # Non-Python or invalid code
            phenotype.complexity = self.num_bins - 1  # Max complexity

        # Incorporate performance metrics if available
        if "combined_score" in metrics:
            # Map score to efficiency (higher score = higher efficiency)
            phenotype.efficiency = int(metrics["combined_score"] * self.num_bins)
            phenotype.efficiency = min(phenotype.efficiency, self.num_bins - 1)

        if "robustness" in metrics:
            phenotype.robustness = int(metrics["robustness"] * self.num_bins)

        # Generate approach signature
        phenotype.approach_signature = self._generate_approach_signature(code)

        return phenotype

    def _analyze_ast(self, tree: ast.AST, phenotype: Phenotype) -> Phenotype:
        """Analyze AST for structural characteristics"""

        # Count nodes for complexity
        node_count = len(list(ast.walk(tree)))
        phenotype.complexity = min(node_count // 10, self.num_bins - 1)

        # Analyze control flow and patterns
        for node in ast.walk(tree):
            # Recursion detection (function calling itself)
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == func_name:
                            phenotype.uses_recursion = True

            # Iteration detection
            if isinstance(node, (ast.For, ast.While)):
                phenotype.uses_iteration = True

            # Vectorization detection (numpy-style operations)
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                module = getattr(node, 'module', '') or ''
                names = [alias.name for alias in getattr(node, 'names', [])]
                if 'numpy' in module or 'np' in names or 'numpy' in names:
                    phenotype.uses_vectorization = True

            # Memoization detection (lru_cache, cache decorators)
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id in ('lru_cache', 'cache', 'memoize'):
                            phenotype.uses_memoization = True
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            if decorator.func.id in ('lru_cache', 'cache', 'memoize'):
                                phenotype.uses_memoization = True

        # Robustness analysis (error handling)
        try_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))
        assert_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))
        phenotype.robustness = min((try_count * 2 + assert_count), self.num_bins - 1)

        return phenotype

    def _generate_approach_signature(self, code: str) -> str:
        """
        Generate a signature that captures the structural approach.

        Two solutions with the same signature likely use similar algorithms,
        even if the specific implementation differs.
        """
        features = []

        try:
            tree = ast.parse(code)

            # Capture structural features
            has_class = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
            has_recursion = False
            has_loop = False
            has_comprehension = False
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    has_loop = True
                if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    has_comprehension = True
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split('.')[0])

            features.append("class" if has_class else "func")
            features.append("loop" if has_loop else "noloop")
            features.append("comp" if has_comprehension else "nocomp")
            features.extend(sorted(imports)[:3])  # Top 3 imports

        except SyntaxError:
            features.append("invalid")

        signature = "_".join(features)
        return hashlib.md5(signature.encode()).hexdigest()[:8]


@dataclass
class SurpriseMetric:
    """
    Tracks how surprising a result was.

    Surprise = |Predicted - Actual|

    High surprise indicates the system encountered something unexpected,
    which is exactly where scientific discoveries happen.
    """
    predicted_fitness: float = 0.0
    actual_fitness: float = 0.0
    surprise_score: float = 0.0
    is_positive_surprise: bool = False  # Better than expected
    timestamp: float = field(default_factory=time.time)

    def calculate(self) -> float:
        """Calculate surprise score"""
        self.surprise_score = abs(self.actual_fitness - self.predicted_fitness)
        self.is_positive_surprise = self.actual_fitness > self.predicted_fitness
        return self.surprise_score


class EpistemicArchive:
    """
    Enhanced MAP-Elites archive with epistemic (knowledge) tracking.

    Key enhancements over standard MAP-Elites:
    1. Behavioral phenotypes instead of simple metrics
    2. Surprise-based curiosity reward
    3. Cross-problem knowledge transfer
    4. Approach diversity (not just performance diversity)
    """

    def __init__(
        self,
        database: "ProgramDatabase",
        phenotype_dimensions: List[str] = None,
        num_bins: int = 10,
    ):
        self.database = database
        self.num_bins = num_bins
        self.phenotype_dimensions = phenotype_dimensions or ["complexity", "efficiency"]
        self.phenotype_extractor = PhenotypeExtractor(num_bins)

        # Surprise tracking
        self.surprise_history: List[SurpriseMetric] = []
        self.prediction_model: Dict[str, float] = {}  # Simple prediction model

        # Cross-problem knowledge
        self.approach_library: Dict[str, Dict[str, Any]] = {}  # signature -> approach info

        logger.info(
            f"Initialized EpistemicArchive with dimensions: {self.phenotype_dimensions}"
        )

    def add_with_phenotype(
        self,
        program: "Program",
        predicted_fitness: float = None,
    ) -> Tuple[bool, Optional[SurpriseMetric]]:
        """
        Add a program to the archive with phenotype extraction and surprise tracking.

        Args:
            program: The program to add
            predicted_fitness: What we expected the fitness to be

        Returns:
            Tuple of (was_added, surprise_metric)
        """
        # Extract phenotype
        phenotype = self.phenotype_extractor.extract(program.code, program.metrics)

        # Store phenotype in program metadata
        if program.metadata is None:
            program.metadata = {}
        program.metadata["phenotype"] = phenotype.to_dict()

        # Calculate surprise if prediction was provided
        surprise = None
        if predicted_fitness is not None:
            actual_fitness = program.metrics.get("combined_score", 0.0)
            surprise = SurpriseMetric(
                predicted_fitness=predicted_fitness,
                actual_fitness=actual_fitness,
            )
            surprise.calculate()
            self.surprise_history.append(surprise)

            # Log surprising results
            if surprise.surprise_score > 0.2:
                logger.info(
                    f"SURPRISE: Program {program.id} "
                    f"(predicted: {predicted_fitness:.3f}, actual: {actual_fitness:.3f}, "
                    f"surprise: {surprise.surprise_score:.3f}, "
                    f"{'positive' if surprise.is_positive_surprise else 'negative'})"
                )

        # Store approach in library for cross-problem transfer
        if phenotype.approach_signature not in self.approach_library:
            self.approach_library[phenotype.approach_signature] = {
                "first_seen": time.time(),
                "example_code": program.code[:500],
                "phenotype": phenotype.to_dict(),
                "success_count": 0,
                "failure_count": 0,
            }

        # Update approach statistics
        actual_fitness = program.metrics.get("combined_score", 0.0)
        if actual_fitness > 0.5:
            self.approach_library[phenotype.approach_signature]["success_count"] += 1
        else:
            self.approach_library[phenotype.approach_signature]["failure_count"] += 1

        # Add to underlying database (which handles MAP-Elites)
        # The database will use its own feature dimensions
        self.database.add(program)

        # Check if this was a novel niche occupation
        was_novel = self._check_novelty(program, phenotype)

        return was_novel, surprise

    def _check_novelty(self, program: "Program", phenotype: Phenotype) -> bool:
        """Check if this program occupies a novel niche"""
        # Get current island
        island_idx = program.metadata.get("island", 0)

        # Check the island's feature map
        if island_idx < len(self.database.island_feature_maps):
            feature_map = self.database.island_feature_maps[island_idx]

            # Calculate feature key using database's method
            coords = self.database._calculate_feature_coords(program)
            key = self.database._feature_coords_to_key(coords)

            # Novel if this key is new OR if we just replaced a worse program
            return key not in feature_map or feature_map[key] == program.id

        return False

    def predict_fitness(self, code: str) -> float:
        """
        Predict expected fitness for code before evaluation.

        This enables surprise-based curiosity: we're more interested
        in results that differ from our predictions.
        """
        phenotype = self.phenotype_extractor.extract(code)

        # Use approach signature for prediction if we've seen it before
        if phenotype.approach_signature in self.approach_library:
            approach = self.approach_library[phenotype.approach_signature]
            total = approach["success_count"] + approach["failure_count"]
            if total > 0:
                return approach["success_count"] / total

        # Default prediction based on complexity heuristic
        # Very simple or very complex code tends to be worse
        optimal_complexity = self.num_bins // 2
        complexity_penalty = abs(phenotype.complexity - optimal_complexity) / self.num_bins
        return max(0.0, 0.5 - complexity_penalty)

    def get_surprise_statistics(self) -> Dict[str, Any]:
        """Get statistics about surprises encountered"""
        if not self.surprise_history:
            return {"total_surprises": 0}

        surprises = [s.surprise_score for s in self.surprise_history]
        positive = [s for s in self.surprise_history if s.is_positive_surprise]

        return {
            "total_surprises": len(self.surprise_history),
            "mean_surprise": np.mean(surprises),
            "max_surprise": np.max(surprises),
            "positive_surprises": len(positive),
            "positive_ratio": len(positive) / len(self.surprise_history),
            "high_surprise_count": sum(1 for s in surprises if s > 0.2),
        }

    def get_approach_diversity(self) -> Dict[str, Any]:
        """Get statistics about approach diversity"""
        return {
            "unique_approaches": len(self.approach_library),
            "approaches": {
                sig: {
                    "success_rate": info["success_count"] / max(
                        1, info["success_count"] + info["failure_count"]
                    ),
                    "total_uses": info["success_count"] + info["failure_count"],
                }
                for sig, info in self.approach_library.items()
            },
        }

    def get_promising_approaches(self, min_success_rate: float = 0.6) -> List[str]:
        """Get code examples of promising approaches"""
        promising = []
        for sig, info in self.approach_library.items():
            total = info["success_count"] + info["failure_count"]
            if total >= 3:  # Enough data points
                success_rate = info["success_count"] / total
                if success_rate >= min_success_rate:
                    promising.append(info["example_code"])
        return promising

    def sample_for_curiosity(self, n: int = 3) -> List["Program"]:
        """
        Sample programs that maximize expected information gain.

        We want to sample from:
        1. Regions with high surprise history (unexpected results)
        2. Under-explored regions of the phenotype space
        3. Approaches we haven't tried much yet
        """
        candidates = []

        # Get all programs
        all_programs = list(self.database.programs.values())
        if not all_programs:
            return []

        # Score each program by "curiosity value"
        scored = []
        for program in all_programs:
            score = 0.0

            # Factor 1: Phenotype uniqueness
            phenotype = program.metadata.get("phenotype", {})
            approach_sig = phenotype.get("approach_signature", "")
            if approach_sig in self.approach_library:
                uses = (
                    self.approach_library[approach_sig]["success_count"] +
                    self.approach_library[approach_sig]["failure_count"]
                )
                # Rare approaches are more interesting
                score += 1.0 / max(uses, 1)

            # Factor 2: Associated surprise
            program_surprises = [
                s for s in self.surprise_history
                if abs(s.actual_fitness - program.metrics.get("combined_score", 0)) < 0.01
            ]
            if program_surprises:
                score += max(s.surprise_score for s in program_surprises)

            # Factor 3: Frontier programs (high in one dimension, low in another)
            complexity = phenotype.get("complexity", 0)
            efficiency = phenotype.get("efficiency", 0)
            if (complexity > self.num_bins * 0.7 and efficiency < self.num_bins * 0.3) or \
               (complexity < self.num_bins * 0.3 and efficiency > self.num_bins * 0.7):
                score += 0.5  # Bonus for frontier programs

            scored.append((program, score))

        # Sort by score and take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:n]]
