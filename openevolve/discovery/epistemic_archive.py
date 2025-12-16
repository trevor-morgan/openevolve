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
from typing import TYPE_CHECKING, Any

import numpy as np

from openevolve.config import EpistemicArchiveConfig

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

    # Ontology tracking - for Heisenberg Engine
    ontology_generation: int = 0  # Which ontology generation this was evaluated under
    extracted_variables: dict[str, Any] = field(
        default_factory=dict
    )  # Values of ontology variables

    def to_grid_coords(self, num_bins: int = 10) -> tuple[int, int]:
        """Convert phenotype to 2D grid coordinates for MAP-Elites"""
        x = min(self.complexity, num_bins - 1)
        y = min(self.efficiency, num_bins - 1)
        return (x, y)

    def to_nd_coords(self, dimensions: list[str], num_bins: int = 10) -> tuple[int, ...]:
        """Convert phenotype to N-dimensional coordinates"""
        coords = []
        for dim in dimensions:
            value = getattr(self, dim, None)
            if value is None and dim.startswith("var_"):
                value = self.extracted_variables.get(dim[len("var_") :])
            if value is None:
                value = 0
            if isinstance(value, bool):
                value = 1 if value else 0
            elif isinstance(value, str):
                value = hash(value) % num_bins
            coords.append(min(int(value), num_bins - 1))
        return tuple(coords)

    def to_dict(self) -> dict[str, Any]:
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
            "ontology_generation": self.ontology_generation,
            "extracted_variables": self.extracted_variables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Phenotype":
        """Deserialize from dictionary"""
        return cls(
            complexity=data.get("complexity", 0),
            efficiency=data.get("efficiency", 0),
            robustness=data.get("robustness", 0),
            generality=data.get("generality", 0),
            approach_signature=data.get("approach_signature", ""),
            uses_recursion=data.get("uses_recursion", False),
            uses_iteration=data.get("uses_iteration", False),
            uses_vectorization=data.get("uses_vectorization", False),
            uses_memoization=data.get("uses_memoization", False),
            best_case_complexity=data.get("best_case_complexity", "O(?)"),
            worst_case_complexity=data.get("worst_case_complexity", "O(?)"),
            space_complexity=data.get("space_complexity", "O(?)"),
            ontology_generation=data.get("ontology_generation", 0),
            extracted_variables=data.get("extracted_variables", {}),
        )


class PhenotypeExtractor:
    """
    Extracts behavioral phenotypes from code.

    This goes beyond simple metrics to understand HOW code works,
    not just how well it performs on a benchmark.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def extract(self, code: str, metrics: dict[str, float] = None) -> Phenotype:
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
                module = getattr(node, "module", "") or ""
                names = [alias.name for alias in getattr(node, "names", [])]
                if "numpy" in module or "np" in names or "numpy" in names:
                    phenotype.uses_vectorization = True

            # Memoization detection (lru_cache, cache decorators)
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id in ("lru_cache", "cache", "memoize"):
                            phenotype.uses_memoization = True
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            if decorator.func.id in ("lru_cache", "cache", "memoize"):
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
                    imports.add(node.module.split(".")[0])

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
        phenotype_dimensions: list[str] = None,
        custom_phenotype_extractor: Any | None = None,
        mirror_dimensions: list[str] | None = None,
        num_bins: int = 10,
        config: EpistemicArchiveConfig = None,
    ):
        self.database = database
        self.num_bins = num_bins
        self.config = config or EpistemicArchiveConfig()
        self.phenotype_dimensions = phenotype_dimensions or ["complexity", "efficiency"]
        self.phenotype_extractor = PhenotypeExtractor(num_bins)
        self.custom_phenotype_extractor = custom_phenotype_extractor
        self.mirror_dimensions = mirror_dimensions or []

        # Surprise tracking
        self.surprise_history: list[SurpriseMetric] = []
        self.prediction_model: dict[str, float] = {}  # Simple prediction model

        # Cross-problem knowledge
        self.approach_library: dict[str, dict[str, Any]] = {}  # signature -> approach info

        # Ontology tracking (for Heisenberg Engine)
        self.current_ontology_generation: int = 0
        self._ontology_history: list[dict[str, Any]] = []

        logger.info(f"Initialized EpistemicArchive with dimensions: {self.phenotype_dimensions}")

    def add_with_phenotype(
        self,
        program: "Program",
        predicted_fitness: float = None,
    ) -> tuple[bool, SurpriseMetric | None]:
        """
        Add a program to the archive with phenotype extraction and surprise tracking.

        Args:
            program: The program to add
            predicted_fitness: What we expected the fitness to be

        Returns:
            Tuple of (was_added, surprise_metric)
        """
        # Extract phenotype (default structural extractor)
        phenotype = self.phenotype_extractor.extract(program.code, program.metrics)
        phenotype.ontology_generation = self.current_ontology_generation

        artifacts: dict[str, Any] | None = None
        if getattr(self.database, "get_artifacts", None) is not None:
            try:
                artifacts = self.database.get_artifacts(program.id)
            except Exception:
                artifacts = None
        if artifacts is None and program.metadata:
            artifacts = program.metadata.get("artifacts")

        # If ontology variable dimensions are present, try to extract numeric values from
        # artifacts/metrics safely (no arbitrary code execution).
        for dim in self.phenotype_dimensions:
            if not dim.startswith("var_"):
                continue
            var_name = dim[len("var_") :]
            val = None
            try:
                if program.metrics and var_name in program.metrics:
                    val = program.metrics.get(var_name)
            except Exception:
                val = None
            if val is None and artifacts is not None:
                val = self._find_numeric_value(artifacts, var_name)
            numeric = self._coerce_numeric(val)
            if numeric is not None:
                phenotype.extracted_variables[var_name] = numeric

        # Optional task-specific phenotype augmentation/override.
        if self.custom_phenotype_extractor is not None:
            try:
                extra = self.custom_phenotype_extractor(program.code, program.metrics, artifacts)
                if isinstance(extra, Phenotype):
                    phenotype = extra
                elif isinstance(extra, dict):
                    for k, v in extra.items():
                        setattr(phenotype, k, v)
                        if (
                            k not in self.phenotype_dimensions
                            and len(self.phenotype_dimensions) < 10
                        ):
                            self.phenotype_dimensions.append(k)
            except Exception as e:
                logger.debug(f"Custom phenotype extractor failed: {e}")

        # Optionally mirror selected phenotype dimensions into program metrics
        # so MAP-Elites can use them as feature dimensions.
        if self.mirror_dimensions:
            if program.metrics is None:
                program.metrics = {}
            for dim in self.mirror_dimensions:
                if dim in program.metrics:
                    continue
                val = getattr(phenotype, dim, None)
                if val is None and dim.startswith("var_"):
                    val = phenotype.extracted_variables.get(dim[len("var_") :])
                if isinstance(val, bool):
                    program.metrics[dim] = 1.0 if val else 0.0
                elif isinstance(val, (int, float)):
                    program.metrics[dim] = float(val)

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
                "fitness_sum": 0.0,
                "fitness_count": 0,
            }

        # Update approach statistics
        actual_fitness = program.metrics.get("combined_score", 0.0)
        approach_entry = self.approach_library[phenotype.approach_signature]
        approach_entry["fitness_sum"] = float(approach_entry.get("fitness_sum", 0.0)) + float(
            actual_fitness
        )
        approach_entry["fitness_count"] = int(approach_entry.get("fitness_count", 0)) + 1
        if actual_fitness > 0.5:
            approach_entry["success_count"] += 1
        else:
            approach_entry["failure_count"] += 1

        # Add to underlying database (which handles MAP-Elites) only if not already present.
        # In Discovery Mode integrated into OpenEvolve, programs are added by the main loop.
        if (
            getattr(self.database, "programs", None) is None
            or program.id not in self.database.programs
        ):
            # Preserve iteration_found/last_iteration when discovery admits programs
            iter_idx = getattr(program, "iteration_found", None)
            self.database.add(program, iteration=iter_idx)

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

    @staticmethod
    def _coerce_numeric(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            try:
                return float(text)
            except Exception:
                return None
        return None

    @classmethod
    def _find_numeric_value(cls, obj: Any, key: str, depth: int = 0) -> float | None:
        """Find the first numeric value for `key` in nested dict/list structures."""
        if depth > 6 or obj is None:
            return None

        def _norm(s: str) -> str:
            return "".join(ch.lower() for ch in s if ch.isalnum() or ch in ("_",))

        target = _norm(key)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and _norm(k) == target:
                    coerced = cls._coerce_numeric(v)
                    if coerced is not None:
                        return coerced
                found = cls._find_numeric_value(v, key, depth=depth + 1)
                if found is not None:
                    return found
            return None

        if isinstance(obj, (list, tuple)):
            for item in obj:
                found = cls._find_numeric_value(item, key, depth=depth + 1)
                if found is not None:
                    return found
            return None

        if isinstance(obj, str):
            text = obj.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                except Exception:
                    return None
                return cls._find_numeric_value(parsed, key, depth=depth + 1)
            return cls._coerce_numeric(text)

        return cls._coerce_numeric(obj)

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
            count = int(approach.get("fitness_count", 0))
            fitness_sum = float(approach.get("fitness_sum", 0.0))
            if count > 0:
                # Shrink mean toward a neutral prior to avoid early saturation.
                prior_mean = self.config.prediction_prior_mean
                prior_weight = self.config.prediction_prior_weight
                mean = (fitness_sum + prior_mean * prior_weight) / (count + prior_weight)
                return float(np.clip(mean, 0.0, 1.0))

            total = approach.get("success_count", 0) + approach.get("failure_count", 0)
            if total > 0:
                prior_alpha = 1.0
                prior_beta = 1.0
                smoothed = (approach.get("success_count", 0) + prior_alpha) / (
                    total + prior_alpha + prior_beta
                )
                return float(np.clip(smoothed, 0.0, 1.0))

        # Default prediction based on complexity heuristic
        # Very simple or very complex code tends to be worse
        optimal_complexity = self.num_bins // 2
        complexity_penalty = abs(phenotype.complexity - optimal_complexity) / self.num_bins
        return max(0.0, 0.5 - complexity_penalty)

    def get_surprise_statistics(self) -> dict[str, Any]:
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

    def get_approach_diversity(self) -> dict[str, Any]:
        """Get statistics about approach diversity"""
        return {
            "unique_approaches": len(self.approach_library),
            "approaches": {
                sig: {
                    "success_rate": info["success_count"]
                    / max(1, info["success_count"] + info["failure_count"]),
                    "total_uses": info["success_count"] + info["failure_count"],
                }
                for sig, info in self.approach_library.items()
            },
        }

    def get_promising_approaches(self, min_success_rate: float = 0.6) -> list[str]:
        """Get code examples of promising approaches"""
        promising = []
        for sig, info in self.approach_library.items():
            total = info["success_count"] + info["failure_count"]
            if total >= 3:  # Enough data points
                success_rate = info["success_count"] / total
                if success_rate >= min_success_rate:
                    promising.append(info["example_code"])
        return promising

    def sample_for_curiosity(self, n: int = 3) -> list["Program"]:
        """
        Sample programs that maximize expected information gain.

        We want to sample from:
        1. Regions with high surprise history (unexpected results)
        2. Under-explored regions of the phenotype space
        3. Approaches we haven't tried much yet
        """
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
                    self.approach_library[approach_sig]["success_count"]
                    + self.approach_library[approach_sig]["failure_count"]
                )
                # Rare approaches are more interesting
                score += 1.0 / max(uses, 1)

            # Factor 2: Associated surprise
            program_surprises = [
                s
                for s in self.surprise_history
                if abs(s.actual_fitness - program.metrics.get("combined_score", 0)) < 0.01
            ]
            if program_surprises:
                score += max(s.surprise_score for s in program_surprises)

            # Factor 3: Frontier programs (high in one dimension, low in another)
            complexity = phenotype.get("complexity", 0)
            efficiency = phenotype.get("efficiency", 0)
            if (
                complexity > self.num_bins * self.config.frontier_threshold_high
                and efficiency < self.num_bins * self.config.frontier_threshold_low
            ) or (
                complexity < self.num_bins * self.config.frontier_threshold_low
                and efficiency > self.num_bins * self.config.frontier_threshold_high
            ):
                score += self.config.frontier_bonus  # Bonus for frontier programs

            scored.append((program, score))

        # Sort by score and take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:n]]

    def update_for_ontology(
        self,
        ontology_generation: int,
        new_variables: list[str],
    ) -> None:
        """
        Update the archive for a new ontology generation.

        Called when the Heisenberg Engine expands the ontology with new variables.
        This adds new dimensions to the phenotype tracking and can optionally
        add the new variables as additional diversity dimensions.

        Args:
            ontology_generation: The new ontology generation number
            new_variables: Names of newly discovered variables
        """
        self.current_ontology_generation = ontology_generation

        # Optionally add new variables as phenotype dimensions
        # This allows the MAP-Elites grid to diversify along discovered variables
        for var_name in new_variables:
            dimension_name = f"var_{var_name}"
            if dimension_name not in self.phenotype_dimensions:
                # Only add if we haven't exceeded max dimensions
                if len(self.phenotype_dimensions) < 10:  # Cap at 10 dimensions
                    self.phenotype_dimensions.append(dimension_name)
                    logger.info(
                        f"Added new phenotype dimension: {dimension_name} "
                        f"(ontology gen {ontology_generation})"
                    )

        # Track the ontology expansion event
        self._ontology_history.append(
            {
                "generation": ontology_generation,
                "new_variables": new_variables,
                "timestamp": time.time(),
                "phenotype_dimensions": self.phenotype_dimensions.copy(),
            }
        )

        # Update the underlying database to use the new dimensions
        # This dynamically re-bins the MAP-Elites grid to explore the new dimensions
        if hasattr(self.database, "update_feature_dimensions"):
            # Combine config dimensions with our phenotype dimensions
            # We want to preserve built-ins but add our discovered vars
            base_dims = [
                d for d in self.database.config.feature_dimensions if not d.startswith("var_")
            ]
            # Filter phenotype dims to only include variables
            var_dims = [d for d in self.phenotype_dimensions if d.startswith("var_")]

            new_db_dims = base_dims + var_dims
            self.database.update_feature_dimensions(new_db_dims)

        logger.info(
            f"Updated archive for ontology generation {ontology_generation}, "
            f"new variables: {new_variables}"
        )

    def get_programs_for_reevaluation(
        self,
        n: int = 10,
        ontology_generation: int = None,
    ) -> list["Program"]:
        """
        Get top programs that should be re-evaluated after ontology expansion.

        Programs evaluated under older ontology generations may perform
        differently when considering newly discovered variables.

        Args:
            n: Number of programs to return
            ontology_generation: If provided, get programs from this generation or earlier

        Returns:
            List of top programs for re-evaluation
        """
        programs = []
        for program in self.database.programs.values():
            phenotype_data = program.metadata.get("phenotype", {})
            prog_ontology_gen = phenotype_data.get("ontology_generation", 0)

            # Include if from older ontology generation
            if ontology_generation is None or prog_ontology_gen < ontology_generation:
                programs.append(program)

        # Sort by combined_score descending
        programs.sort(key=lambda p: p.metrics.get("combined_score", 0.0), reverse=True)

        return programs[:n]

    def get_ontology_statistics(self) -> dict[str, Any]:
        """Get statistics about ontology evolution impact on archive"""
        if not hasattr(self, "_ontology_history") or not self._ontology_history:
            return {
                "current_ontology_generation": getattr(self, "current_ontology_generation", 0),
                "expansion_count": 0,
            }

        return {
            "current_ontology_generation": self.current_ontology_generation,
            "expansion_count": len(self._ontology_history),
            "total_discovered_variables": sum(
                len(h["new_variables"]) for h in self._ontology_history
            ),
            "current_dimensions": len(self.phenotype_dimensions),
            "history": self._ontology_history,
        }
