"""
Reward calculation for RL-based adaptive selection

This module computes multi-objective rewards for the RL policy learner,
balancing fitness improvement, diversity, novelty, and efficiency.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

from openevolve.config import RLRewardConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis"""

    fitness_reward: float = 0.0
    diversity_reward: float = 0.0
    novelty_reward: float = 0.0
    efficiency_reward: float = 0.0
    shaped_bonus: float = 0.0
    plateau_penalty: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary"""
        return {
            "fitness_reward": self.fitness_reward,
            "diversity_reward": self.diversity_reward,
            "novelty_reward": self.novelty_reward,
            "efficiency_reward": self.efficiency_reward,
            "shaped_bonus": self.shaped_bonus,
            "plateau_penalty": self.plateau_penalty,
            "total_reward": self.total_reward,
        }


class RewardCalculator:
    """Calculates multi-objective rewards for RL policy learning

    The reward function balances:
    - Fitness improvement (primary objective)
    - Diversity maintenance (avoid premature convergence)
    - Novelty seeking (encourage exploration of new behaviors)
    - Efficiency (reward quick improvements)

    Supports reward shaping to guide early vs late evolution behavior.
    """

    def __init__(self, config: RLRewardConfig):
        """Initialize the reward calculator

        Args:
            config: Reward configuration with weights and thresholds
        """
        self.config = config

        # Running statistics for normalization
        self.fitness_improvements: list[float] = []
        self.max_improvement_seen = 0.01  # Avoid division by zero
        self.min_improvement_seen = -0.01

    def compute(
        self,
        parent_fitness: float,
        child_fitness: float,
        diversity_delta: float = 0.0,
        novelty_score: float = 0.0,
        iterations_since_improvement: int = 0,
        normalized_iteration: float = 0.0,
        detailed: bool = False,
    ) -> float | RewardComponents:
        """Compute the reward for an evolution step

        Args:
            parent_fitness: Fitness of the parent program
            child_fitness: Fitness of the child program
            diversity_delta: Change in population diversity
            novelty_score: Novelty score of the child (0-1)
            iterations_since_improvement: Stagnation counter
            normalized_iteration: Current iteration / max iterations
            detailed: If True, return RewardComponents with breakdown

        Returns:
            Total reward (float) or RewardComponents if detailed=True
        """
        components = RewardComponents()

        # 1. Fitness improvement reward
        fitness_delta = child_fitness - parent_fitness
        self._update_fitness_stats(fitness_delta)

        # Normalize improvement to roughly [-1, 1]
        normalized_delta = self._normalize_improvement(fitness_delta)

        # Apply threshold
        if fitness_delta < self.config.improvement_threshold:
            components.fitness_reward = normalized_delta * 0.5  # Penalize less severely
        else:
            components.fitness_reward = normalized_delta

        components.fitness_reward *= self.config.fitness_weight

        # 2. Diversity reward
        # Positive diversity_delta means population became more diverse
        components.diversity_reward = math.tanh(diversity_delta * 5) * self.config.diversity_weight

        # 3. Novelty reward
        # novelty_score is expected to be in [0, 1]
        components.novelty_reward = novelty_score * self.config.novelty_weight

        # 4. Efficiency reward (bonus for improving without many failures)
        if fitness_delta > 0:
            # Efficiency bonus: inversely proportional to attempts
            # Quick improvements get higher reward
            efficiency = 1.0 / (1.0 + math.log1p(iterations_since_improvement))
            components.efficiency_reward = efficiency * self.config.efficiency_weight

        # 5. Reward shaping based on evolution phase
        components.shaped_bonus = self._compute_shaped_bonus(
            fitness_delta, diversity_delta, normalized_iteration
        )

        # 6. Plateau penalty
        if iterations_since_improvement > self.config.plateau_window:
            # Gradual penalty that increases with stagnation
            penalty_factor = (
                iterations_since_improvement - self.config.plateau_window
            ) / self.config.plateau_window
            components.plateau_penalty = -self.config.plateau_penalty * min(1.0, penalty_factor)

        # Total reward
        components.total_reward = (
            components.fitness_reward
            + components.diversity_reward
            + components.novelty_reward
            + components.efficiency_reward
            + components.shaped_bonus
            + components.plateau_penalty
        )

        if detailed:
            return components
        return components.total_reward

    def compute_from_outcome(
        self,
        parent_fitness: float,
        child_fitness: float,
        state: Any,  # EvolutionState
    ) -> float:
        """Compute reward using an EvolutionState for context

        Args:
            parent_fitness: Parent program fitness
            child_fitness: Child program fitness
            state: EvolutionState with context

        Returns:
            Computed reward
        """
        # Extract relevant state features
        diversity_delta = 0.0  # Would need to track this separately
        novelty_score = 0.0  # Would need novelty archive
        iterations_since_improvement = getattr(state, "generations_without_improvement", 0)
        normalized_iteration = getattr(state, "normalized_iteration", 0.0)

        return self.compute(
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            diversity_delta=diversity_delta,
            novelty_score=novelty_score,
            iterations_since_improvement=iterations_since_improvement,
            normalized_iteration=normalized_iteration,
        )

    def _update_fitness_stats(self, delta: float):
        """Update running statistics for normalization"""
        self.fitness_improvements.append(delta)

        # Keep a bounded history
        if len(self.fitness_improvements) > 1000:
            self.fitness_improvements = self.fitness_improvements[-500:]

        self.max_improvement_seen = max(self.max_improvement_seen, delta)
        self.min_improvement_seen = min(self.min_improvement_seen, delta)

    def _normalize_improvement(self, delta: float) -> float:
        """Normalize fitness improvement to roughly [-1, 1]

        Uses adaptive normalization based on observed improvements.
        """
        if delta >= 0:
            # Positive improvements normalized by max seen
            return min(1.0, delta / max(0.01, self.max_improvement_seen))
        else:
            # Negative changes normalized by min seen
            return max(-1.0, delta / max(0.01, abs(self.min_improvement_seen)))

    def _compute_shaped_bonus(
        self,
        fitness_delta: float,
        diversity_delta: float,
        normalized_iteration: float,
    ) -> float:
        """Compute reward shaping bonus based on evolution phase

        Early phase: Encourage diversity
        Late phase: Encourage fitness improvement
        """
        # Early iterations (< 30%): bonus for diversity
        if normalized_iteration < 0.3:
            diversity_bonus = (0.3 - normalized_iteration) * diversity_delta * 0.3
        else:
            diversity_bonus = 0.0

        # Late iterations (> 70%): bonus for fitness
        if normalized_iteration > 0.7 and fitness_delta > 0:
            fitness_bonus = (normalized_iteration - 0.7) * fitness_delta * 0.3
        else:
            fitness_bonus = 0.0

        return diversity_bonus + fitness_bonus

    def reset(self):
        """Reset running statistics"""
        self.fitness_improvements = []
        self.max_improvement_seen = 0.01
        self.min_improvement_seen = -0.01

    def save_state(self) -> dict[str, Any]:
        """Save calculator state for checkpointing"""
        return {
            "fitness_improvements": self.fitness_improvements[-100:],  # Keep recent
            "max_improvement_seen": self.max_improvement_seen,
            "min_improvement_seen": self.min_improvement_seen,
        }

    def load_state(self, state: dict[str, Any]):
        """Load calculator state from checkpoint"""
        self.fitness_improvements = state.get("fitness_improvements", [])
        self.max_improvement_seen = state.get("max_improvement_seen", 0.01)
        self.min_improvement_seen = state.get("min_improvement_seen", -0.01)

    def get_statistics(self) -> dict[str, Any]:
        """Get reward statistics for logging"""
        if not self.fitness_improvements:
            return {"count": 0}

        improvements = self.fitness_improvements
        positive = [x for x in improvements if x > 0]
        negative = [x for x in improvements if x < 0]

        return {
            "count": len(improvements),
            "mean_improvement": sum(improvements) / len(improvements),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "positive_rate": len(positive) / len(improvements),
            "max_improvement": self.max_improvement_seen,
            "min_improvement": self.min_improvement_seen,
        }
