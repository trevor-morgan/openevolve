"""
State feature extraction for RL-based adaptive selection

This module provides rich state representation for the RL policy learner
by extracting relevant features from the evolution state.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from openevolve.database import ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolutionState:
    """Rich state representation for RL policy

    All values are normalized to [0, 1] or [-1, 1] for stable learning.
    """

    # Fitness Statistics (global)
    best_fitness: float = 0.0  # Best fitness so far (normalized)
    mean_fitness: float = 0.0  # Population mean fitness
    fitness_std: float = 0.0  # Population standard deviation
    fitness_improvement_rate: float = 0.0  # Recent improvement rate

    # Progress Indicators
    iteration: int = 0  # Current iteration
    normalized_iteration: float = 0.0  # iteration / max_iterations
    generations_without_improvement: int = 0  # Stagnation counter

    # Diversity Metrics
    population_diversity: float = 0.0  # Edit distance diversity
    archive_coverage: float = 0.0  # MAP-Elites grid coverage fraction
    unique_solutions: int = 0  # Number of unique programs

    # Island-Specific (if using islands)
    island_idx: int = 0
    island_best_fitness: float = 0.0
    island_mean_fitness: float = 0.0
    island_diversity: float = 0.0
    inter_island_variance: float = 0.0  # How different are islands?

    # Selection History
    recent_exploration_success: float = 0.5  # Success rate of exploration
    recent_exploitation_success: float = 0.5  # Success rate of exploitation
    recent_weighted_success: float = 0.5  # Success rate of weighted sampling
    recent_novelty_success: float = 0.5  # Success rate of novelty sampling
    recent_curiosity_success: float = 0.5  # Success rate of curiosity sampling

    # Meta-Prompting Integration (if enabled)
    current_strategy_success_rate: float = 0.5  # Meta-prompting strategy success

    def to_array(self, feature_names: list[str] | None = None) -> np.ndarray:
        """Convert state to numpy array for policy input

        Args:
            feature_names: List of feature names to include. If None, uses default set.

        Returns:
            Normalized feature vector
        """
        if feature_names is None:
            feature_names = DEFAULT_FEATURES

        features = []
        for name in feature_names:
            value = getattr(self, name, 0.0)
            features.append(float(value))

        return np.array(features, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "fitness_std": self.fitness_std,
            "fitness_improvement_rate": self.fitness_improvement_rate,
            "iteration": self.iteration,
            "normalized_iteration": self.normalized_iteration,
            "generations_without_improvement": self.generations_without_improvement,
            "population_diversity": self.population_diversity,
            "archive_coverage": self.archive_coverage,
            "unique_solutions": self.unique_solutions,
            "island_idx": self.island_idx,
            "island_best_fitness": self.island_best_fitness,
            "island_mean_fitness": self.island_mean_fitness,
            "island_diversity": self.island_diversity,
            "inter_island_variance": self.inter_island_variance,
            "recent_exploration_success": self.recent_exploration_success,
            "recent_exploitation_success": self.recent_exploitation_success,
            "recent_weighted_success": self.recent_weighted_success,
            "recent_novelty_success": self.recent_novelty_success,
            "recent_curiosity_success": self.recent_curiosity_success,
            "current_strategy_success_rate": self.current_strategy_success_rate,
        }


# Default features used if not specified in config
DEFAULT_FEATURES = [
    "normalized_iteration",
    "best_fitness",
    "fitness_improvement_rate",
    "population_diversity",
    "archive_coverage",
    "generations_without_improvement",
    "recent_exploration_success",
    "recent_exploitation_success",
]


@dataclass
class ActionOutcome:
    """Record of an action and its outcome"""

    action: int
    parent_fitness: float
    child_fitness: float
    improved: bool
    timestamp: float = 0.0


class StateFeatureExtractor:
    """Extracts rich state features from the evolution state

    This class computes and tracks all features needed by the RL policy
    to make informed selection decisions.
    """

    def __init__(
        self,
        max_iterations: int = 10000,
        feature_names: list[str] | None = None,
        history_window: int = 100,
    ):
        """Initialize the state feature extractor

        Args:
            max_iterations: Maximum iterations for normalization
            feature_names: List of features to extract
            history_window: Window size for computing rolling statistics
        """
        self.max_iterations = max_iterations
        self.feature_names = feature_names or DEFAULT_FEATURES
        self.history_window = history_window

        # Tracking variables
        self.iteration = 0
        self.best_fitness_seen = float("-inf")
        self.last_improvement_iteration = 0

        # Rolling history for improvement rate
        self.fitness_history: deque[float] = deque(maxlen=history_window)

        # Action outcome tracking per action type
        self.action_outcomes: dict[int, deque[ActionOutcome]] = {
            i: deque(maxlen=history_window)
            for i in range(5)  # 5 action types
        }

        # Island-specific tracking
        self.island_best_fitness: dict[int, float] = {}

    def extract(
        self,
        database: "ProgramDatabase",
        island_idx: int | None = None,
        meta_prompting_success: float | None = None,
    ) -> EvolutionState:
        """Extract current state from database

        Args:
            database: The program database
            island_idx: Current island index (optional)
            meta_prompting_success: Current meta-prompting strategy success rate

        Returns:
            EvolutionState with all features populated
        """
        state = EvolutionState()

        # Get basic iteration info
        state.iteration = self.iteration
        state.normalized_iteration = min(1.0, self.iteration / max(1, self.max_iterations))

        # Extract fitness statistics
        self._extract_fitness_stats(state, database)

        # Extract diversity metrics
        self._extract_diversity_stats(state, database)

        # Extract island-specific stats
        if island_idx is not None:
            self._extract_island_stats(state, database, island_idx)

        # Compute selection history success rates
        self._compute_action_success_rates(state)

        # Add meta-prompting integration
        if meta_prompting_success is not None:
            state.current_strategy_success_rate = meta_prompting_success

        return state

    def _extract_fitness_stats(self, state: EvolutionState, database: "ProgramDatabase"):
        """Extract fitness-related statistics"""
        from openevolve.utils.metrics_utils import get_fitness_score

        programs = list(database.programs.values())
        if not programs:
            return

        # Get fitness scores
        feature_dims = database.config.feature_dimensions
        fitness_scores = [get_fitness_score(p.metrics, feature_dims) for p in programs]

        if not fitness_scores:
            return

        # Basic statistics
        best_fitness = max(fitness_scores)
        mean_fitness = sum(fitness_scores) / len(fitness_scores)

        # Normalize fitness to [0, 1] using sigmoid for unbounded values
        state.best_fitness = self._sigmoid_normalize(best_fitness)
        state.mean_fitness = self._sigmoid_normalize(mean_fitness)

        # Standard deviation (normalized by mean to get coefficient of variation)
        if len(fitness_scores) > 1:
            variance = sum((f - mean_fitness) ** 2 for f in fitness_scores) / len(fitness_scores)
            std = math.sqrt(variance)
            state.fitness_std = min(1.0, std / (abs(mean_fitness) + 1e-8))
        else:
            state.fitness_std = 0.0

        # Track improvement
        if best_fitness > self.best_fitness_seen:
            self.best_fitness_seen = best_fitness
            self.last_improvement_iteration = self.iteration

        # Compute improvement rate from history
        self.fitness_history.append(best_fitness)
        if len(self.fitness_history) >= 2:
            recent = list(self.fitness_history)[-20:]  # Last 20
            if len(recent) >= 2:
                improvement = recent[-1] - recent[0]
                # Normalize improvement rate
                state.fitness_improvement_rate = self._tanh_normalize(improvement * 10)

        # Stagnation counter
        state.generations_without_improvement = self.iteration - self.last_improvement_iteration

    def _extract_diversity_stats(self, state: EvolutionState, database: "ProgramDatabase"):
        """Extract diversity-related statistics"""
        # Population diversity (use database's diversity metric if available)
        programs = list(database.programs.values())
        state.unique_solutions = len(programs)

        # Estimate diversity from code length variance (simple proxy)
        if len(programs) > 1:
            lengths = [len(p.code) for p in programs]
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            # Normalize: high variance = high diversity
            state.population_diversity = min(1.0, math.sqrt(variance) / (mean_len + 1e-8))
        else:
            state.population_diversity = 0.0

        # Archive coverage (MAP-Elites grid coverage)
        if hasattr(database, "feature_map") and database.feature_map:
            total_cells = 1
            for dim in database.config.feature_dimensions:
                bins = database.config.feature_bins
                if isinstance(bins, dict):
                    total_cells *= bins.get(dim, 10)
                else:
                    total_cells *= bins

            occupied_cells = sum(1 for cell in database.feature_map.values() if cell is not None)
            state.archive_coverage = occupied_cells / max(1, total_cells)
        else:
            state.archive_coverage = 0.0

    def _extract_island_stats(
        self, state: EvolutionState, database: "ProgramDatabase", island_idx: int
    ):
        """Extract island-specific statistics"""
        from openevolve.utils.metrics_utils import get_fitness_score

        state.island_idx = island_idx
        feature_dims = database.config.feature_dimensions

        # Get programs from this island
        if hasattr(database, "islands") and island_idx < len(database.islands):
            island_program_ids = database.islands[island_idx]
            island_programs = [
                database.programs[pid] for pid in island_program_ids if pid in database.programs
            ]

            if island_programs:
                island_fitness = [
                    get_fitness_score(p.metrics, feature_dims) for p in island_programs
                ]
                state.island_best_fitness = self._sigmoid_normalize(max(island_fitness))
                state.island_mean_fitness = self._sigmoid_normalize(
                    sum(island_fitness) / len(island_fitness)
                )

                # Island diversity
                if len(island_programs) > 1:
                    lengths = [len(p.code) for p in island_programs]
                    mean_len = sum(lengths) / len(lengths)
                    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
                    state.island_diversity = min(1.0, math.sqrt(variance) / (mean_len + 1e-8))

        # Inter-island variance
        if hasattr(database, "islands") and len(database.islands) > 1:
            island_means = []
            for i, island_ids in enumerate(database.islands):
                programs = [
                    database.programs[pid] for pid in island_ids if pid in database.programs
                ]
                if programs:
                    mean_fitness = sum(
                        get_fitness_score(p.metrics, feature_dims) for p in programs
                    ) / len(programs)
                    island_means.append(mean_fitness)

            if len(island_means) > 1:
                mean_of_means = sum(island_means) / len(island_means)
                variance = sum((m - mean_of_means) ** 2 for m in island_means) / len(island_means)
                state.inter_island_variance = min(
                    1.0, math.sqrt(variance) / (abs(mean_of_means) + 1e-8)
                )

    def _compute_action_success_rates(self, state: EvolutionState):
        """Compute success rates for each action type from history"""
        action_success = {
            0: "recent_exploration_success",
            1: "recent_exploitation_success",
            2: "recent_weighted_success",
            3: "recent_novelty_success",
            4: "recent_curiosity_success",
        }

        for action_id, attr_name in action_success.items():
            outcomes = self.action_outcomes.get(action_id, [])
            if len(outcomes) >= 3:  # Need minimum samples
                success_count = sum(1 for o in outcomes if o.improved)
                success_rate = success_count / len(outcomes)
                setattr(state, attr_name, success_rate)
            # else: keep default 0.5

    def record_outcome(self, action: int, parent_fitness: float, child_fitness: float):
        """Record the outcome of an action for success rate tracking

        Args:
            action: The action that was taken
            parent_fitness: Parent program fitness
            child_fitness: Child program fitness
        """
        import time

        outcome = ActionOutcome(
            action=action,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            improved=child_fitness > parent_fitness,
            timestamp=time.time(),
        )

        if action in self.action_outcomes:
            self.action_outcomes[action].append(outcome)

    def increment_iteration(self):
        """Increment the iteration counter"""
        self.iteration += 1

    def reset(self):
        """Reset all tracking state"""
        self.iteration = 0
        self.best_fitness_seen = float("-inf")
        self.last_improvement_iteration = 0
        self.fitness_history.clear()
        for outcomes in self.action_outcomes.values():
            outcomes.clear()
        self.island_best_fitness.clear()

    def save_state(self) -> dict[str, Any]:
        """Save extractor state for checkpointing"""
        return {
            "iteration": self.iteration,
            "best_fitness_seen": self.best_fitness_seen,
            "last_improvement_iteration": self.last_improvement_iteration,
            "fitness_history": list(self.fitness_history),
            "action_outcomes": {
                action: [
                    {
                        "action": o.action,
                        "parent_fitness": o.parent_fitness,
                        "child_fitness": o.child_fitness,
                        "improved": o.improved,
                        "timestamp": o.timestamp,
                    }
                    for o in outcomes
                ]
                for action, outcomes in self.action_outcomes.items()
            },
            "island_best_fitness": self.island_best_fitness,
        }

    def load_state(self, state: dict[str, Any]):
        """Load extractor state from checkpoint"""
        self.iteration = state.get("iteration", 0)
        self.best_fitness_seen = state.get("best_fitness_seen", float("-inf"))
        self.last_improvement_iteration = state.get("last_improvement_iteration", 0)

        self.fitness_history.clear()
        for f in state.get("fitness_history", []):
            self.fitness_history.append(f)

        self.action_outcomes = {i: deque(maxlen=self.history_window) for i in range(5)}
        for action_str, outcomes in state.get("action_outcomes", {}).items():
            action = int(action_str)
            if action in self.action_outcomes:
                for o in outcomes:
                    self.action_outcomes[action].append(
                        ActionOutcome(
                            action=o["action"],
                            parent_fitness=o["parent_fitness"],
                            child_fitness=o["child_fitness"],
                            improved=o["improved"],
                            timestamp=o.get("timestamp", 0.0),
                        )
                    )

        self.island_best_fitness = state.get("island_best_fitness", {})

    @staticmethod
    def _sigmoid_normalize(x: float, scale: float = 1.0) -> float:
        """Normalize value to [0, 1] using sigmoid"""
        return 1.0 / (1.0 + math.exp(-x * scale))

    @staticmethod
    def _tanh_normalize(x: float) -> float:
        """Normalize value to [-1, 1] using tanh"""
        return math.tanh(x)
