"""
Policy learning for RL-based adaptive selection

This module implements contextual bandit algorithms for learning
optimal selection policies during evolution.
"""

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openevolve.config import RLConfig
from openevolve.rl.actions import NUM_ACTIONS, SelectionAction
from openevolve.rl.reward import RewardCalculator
from openevolve.rl.state_features import EvolutionState, StateFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class ActionStats:
    """Statistics for a single action"""

    uses: int = 0
    total_reward: float = 0.0
    successes: int = 0  # Positive reward count
    recent_rewards: list[float] = field(default_factory=list)

    # For Thompson Sampling
    alpha: float = 1.0  # Beta prior successes
    beta: float = 1.0  # Beta prior failures

    def mean_reward(self) -> float:
        """Get mean reward"""
        if self.uses == 0:
            return 0.0
        return self.total_reward / self.uses

    def success_rate(self) -> float:
        """Get success rate"""
        if self.uses == 0:
            return 0.5
        return self.successes / self.uses

    def ucb_score(self, total_uses: int, c: float = 2.0) -> float:
        """Compute UCB score"""
        if self.uses == 0:
            return float("inf")
        exploitation = self.mean_reward()
        exploration = c * math.sqrt(math.log(total_uses + 1) / self.uses)
        return exploitation + exploration

    def thompson_sample(self) -> float:
        """Sample from beta posterior"""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float, max_history: int = 100):
        """Update statistics with new reward"""
        self.uses += 1
        self.total_reward += reward

        # Track success
        if reward > 0:
            self.successes += 1
            self.alpha += 1
        else:
            self.beta += 1

        # Keep recent rewards bounded
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > max_history:
            self.recent_rewards = self.recent_rewards[-max_history:]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "uses": self.uses,
            "total_reward": self.total_reward,
            "successes": self.successes,
            "alpha": self.alpha,
            "beta": self.beta,
            "recent_rewards": self.recent_rewards[-20:],  # Keep last 20
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionStats":
        """Create from dictionary"""
        stats = cls()
        stats.uses = data.get("uses", 0)
        stats.total_reward = data.get("total_reward", 0.0)
        stats.successes = data.get("successes", 0)
        stats.alpha = data.get("alpha", 1.0)
        stats.beta = data.get("beta", 1.0)
        stats.recent_rewards = data.get("recent_rewards", [])
        return stats


class BasePolicyLearner(ABC):
    """Abstract base class for policy learners"""

    @abstractmethod
    def select_action(self, state: EvolutionState) -> int:
        """Select an action given the current state"""
        pass

    @abstractmethod
    def update(self, state: EvolutionState, action: int, reward: float):
        """Update the policy with observed reward"""
        pass

    @abstractmethod
    def save_state(self) -> dict[str, Any]:
        """Save policy state"""
        pass

    @abstractmethod
    def load_state(self, state: dict[str, Any]):
        """Load policy state"""
        pass


class ContextualThompsonSampling(BasePolicyLearner):
    """Contextual Thompson Sampling with Bayesian linear regression

    Uses Bayesian linear regression to model expected reward as a
    function of state features for each action.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = NUM_ACTIONS,
        prior_variance: float = 1.0,
        noise_variance: float = 1.0,
    ):
        """Initialize contextual Thompson Sampling

        Args:
            state_dim: Dimension of state feature vector
            n_actions: Number of actions
            prior_variance: Prior variance for weight distribution
            noise_variance: Observation noise variance
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance

        # Bayesian linear regression parameters per action
        # Prior: w ~ N(0, prior_variance * I)
        self.mean = [np.zeros(state_dim) for _ in range(n_actions)]
        self.precision = [np.eye(state_dim) / prior_variance for _ in range(n_actions)]

        # Track updates for logging
        self.update_count = [0] * n_actions

    def select_action(self, state: EvolutionState) -> int:
        """Select action using Thompson Sampling

        Samples weights from posterior and selects action with
        highest predicted reward.
        """
        state_vec = state.to_array()

        sampled_rewards = []
        for a in range(self.n_actions):
            try:
                # Sample weights from posterior
                cov = np.linalg.inv(self.precision[a])
                weights = np.random.multivariate_normal(self.mean[a], cov)
                # Predict reward
                predicted = float(state_vec @ weights)
                sampled_rewards.append(predicted)
            except np.linalg.LinAlgError:
                # If matrix is singular, use mean prediction
                predicted = float(state_vec @ self.mean[a])
                sampled_rewards.append(predicted)

        return int(np.argmax(sampled_rewards))

    def update(self, state: EvolutionState, action: int, reward: float):
        """Bayesian update after observing reward

        Updates the posterior distribution for the chosen action.
        """
        state_vec = state.to_array()

        # Bayesian linear regression update
        # Precision update: Σ^(-1)_new = Σ^(-1)_old + x x^T / σ²
        outer_product = np.outer(state_vec, state_vec) / self.noise_variance
        self.precision[action] = self.precision[action] + outer_product

        # Mean update: μ_new = Σ_new (Σ^(-1)_old μ_old + y x / σ²)
        try:
            cov = np.linalg.inv(self.precision[action])
            prior_term = self.precision[action] @ self.mean[action]
            obs_term = reward * state_vec / self.noise_variance
            self.mean[action] = cov @ (prior_term + obs_term)
        except np.linalg.LinAlgError:
            # Fallback: simple gradient-like update
            lr = 0.01
            prediction = state_vec @ self.mean[action]
            error = reward - prediction
            self.mean[action] = self.mean[action] + lr * error * state_vec

        self.update_count[action] += 1

    def save_state(self) -> dict[str, Any]:
        """Save policy state"""
        return {
            "algorithm": "contextual_thompson",
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "mean": [m.tolist() for m in self.mean],
            "precision": [p.tolist() for p in self.precision],
            "update_count": self.update_count,
        }

    def load_state(self, state: dict[str, Any]):
        """Load policy state"""
        self.state_dim = state.get("state_dim", self.state_dim)
        self.n_actions = state.get("n_actions", self.n_actions)
        self.mean = [np.array(m) for m in state.get("mean", [])]
        self.precision = [np.array(p) for p in state.get("precision", [])]
        self.update_count = state.get("update_count", [0] * self.n_actions)

        # Ensure correct dimensions
        while len(self.mean) < self.n_actions:
            self.mean.append(np.zeros(self.state_dim))
        while len(self.precision) < self.n_actions:
            self.precision.append(np.eye(self.state_dim) / self.prior_variance)


class ContextualUCB(BasePolicyLearner):
    """Contextual UCB (Upper Confidence Bound) algorithm

    Uses linear regression with UCB exploration bonus.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = NUM_ACTIONS,
        exploration_constant: float = 2.0,
        regularization: float = 1.0,
    ):
        """Initialize contextual UCB

        Args:
            state_dim: Dimension of state feature vector
            n_actions: Number of actions
            exploration_constant: UCB exploration parameter
            regularization: L2 regularization for linear regression
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.exploration_constant = exploration_constant
        self.regularization = regularization

        # Linear regression parameters per action
        self.A = [np.eye(state_dim) * regularization for _ in range(n_actions)]  # Design matrix
        self.b = [np.zeros(state_dim) for _ in range(n_actions)]  # Target vector
        self.theta = [np.zeros(state_dim) for _ in range(n_actions)]  # Weights

        self.total_updates = 0

    def select_action(self, state: EvolutionState) -> int:
        """Select action using UCB criterion"""
        state_vec = state.to_array()

        ucb_values = []
        for a in range(self.n_actions):
            try:
                A_inv = np.linalg.inv(self.A[a])
                # Exploitation: predicted reward
                exploitation = float(state_vec @ self.theta[a])
                # Exploration: uncertainty bonus
                exploration = self.exploration_constant * math.sqrt(
                    float(state_vec @ A_inv @ state_vec)
                )
                ucb_values.append(exploitation + exploration)
            except np.linalg.LinAlgError:
                ucb_values.append(float("inf"))

        return int(np.argmax(ucb_values))

    def update(self, state: EvolutionState, action: int, reward: float):
        """Update model with observed reward"""
        state_vec = state.to_array()

        # Update design matrix and target vector
        self.A[action] = self.A[action] + np.outer(state_vec, state_vec)
        self.b[action] = self.b[action] + reward * state_vec

        # Update weights
        try:
            self.theta[action] = np.linalg.solve(self.A[action], self.b[action])
        except np.linalg.LinAlgError:
            pass  # Keep previous weights

        self.total_updates += 1

    def save_state(self) -> dict[str, Any]:
        """Save policy state"""
        return {
            "algorithm": "contextual_ucb",
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "A": [a.tolist() for a in self.A],
            "b": [b.tolist() for b in self.b],
            "theta": [t.tolist() for t in self.theta],
            "total_updates": self.total_updates,
        }

    def load_state(self, state: dict[str, Any]):
        """Load policy state"""
        self.state_dim = state.get("state_dim", self.state_dim)
        self.n_actions = state.get("n_actions", self.n_actions)
        self.A = [np.array(a) for a in state.get("A", [])]
        self.b = [np.array(b) for b in state.get("b", [])]
        self.theta = [np.array(t) for t in state.get("theta", [])]
        self.total_updates = state.get("total_updates", 0)


class EpsilonGreedy(BasePolicyLearner):
    """Simple epsilon-greedy baseline

    Non-contextual baseline that tracks per-action statistics.
    """

    def __init__(
        self,
        n_actions: int = NUM_ACTIONS,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
    ):
        """Initialize epsilon-greedy policy

        Args:
            n_actions: Number of actions
            epsilon: Initial exploration probability
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.action_stats = [ActionStats() for _ in range(n_actions)]
        self.iterations = 0

    def select_action(self, state: EvolutionState) -> int:
        """Select action using epsilon-greedy"""
        # Epsilon decay
        current_epsilon = max(
            self.min_epsilon,
            self.epsilon * (self.epsilon_decay**self.iterations),
        )

        if random.random() < current_epsilon:
            # Explore: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: best mean reward
            best_action = max(
                range(self.n_actions),
                key=lambda a: self.action_stats[a].mean_reward(),
            )
            return best_action

    def update(self, state: EvolutionState, action: int, reward: float):
        """Update action statistics"""
        self.action_stats[action].update(reward)
        self.iterations += 1

    def save_state(self) -> dict[str, Any]:
        """Save policy state"""
        return {
            "algorithm": "epsilon_greedy",
            "n_actions": self.n_actions,
            "epsilon": self.epsilon,
            "action_stats": [s.to_dict() for s in self.action_stats],
            "iterations": self.iterations,
        }

    def load_state(self, state: dict[str, Any]):
        """Load policy state"""
        self.n_actions = state.get("n_actions", self.n_actions)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.iterations = state.get("iterations", 0)
        stats_data = state.get("action_stats", [])
        self.action_stats = [ActionStats.from_dict(s) for s in stats_data]
        while len(self.action_stats) < self.n_actions:
            self.action_stats.append(ActionStats())


class PolicyLearner:
    """Main policy learner that wraps different algorithms

    This is the main interface used by the rest of OpenEvolve.
    It manages the underlying algorithm, state extraction, and
    reward calculation.
    """

    def __init__(self, config: RLConfig):
        """Initialize the policy learner

        Args:
            config: RL configuration
        """
        self.config = config
        self.enabled = config.enabled

        if not self.enabled:
            return

        # Determine state dimension from features
        self.feature_names = config.state_features
        self.state_dim = len(self.feature_names)

        # Warmup and exploration (must be set before creating algorithm)
        self.warmup_iterations = config.warmup_iterations
        self.exploration_rate = 1.0
        self.exploration_decay = config.exploration_decay
        self.min_exploration = config.min_exploration

        # Initialize the underlying algorithm
        self.algorithm = self._create_algorithm(config.algorithm)

        # State feature extractor
        self.state_extractor = StateFeatureExtractor(
            feature_names=self.feature_names,
            history_window=config.action_history_size,
        )

        # Reward calculator
        self.reward_calculator = RewardCalculator(config.reward)

        # Per-island policies (optional)
        self.per_island = config.per_island_policies
        self.island_algorithms: dict[int, BasePolicyLearner] = {}

        # Tracking
        self.iterations = 0
        self.last_action: int | None = None
        self.last_state: EvolutionState | None = None

        # Action statistics for logging
        self.global_action_stats = [ActionStats() for _ in range(NUM_ACTIONS)]

    def _create_algorithm(self, algorithm_name: str) -> BasePolicyLearner:
        """Create the underlying algorithm"""
        if algorithm_name == "contextual_thompson":
            return ContextualThompsonSampling(
                state_dim=self.state_dim,
                n_actions=NUM_ACTIONS,
            )
        elif algorithm_name == "contextual_ucb":
            return ContextualUCB(
                state_dim=self.state_dim,
                n_actions=NUM_ACTIONS,
                exploration_constant=self.config.exploration_bonus,
            )
        elif algorithm_name == "epsilon_greedy":
            return EpsilonGreedy(
                n_actions=NUM_ACTIONS,
                epsilon=self.min_exploration + (1 - self.min_exploration),
                epsilon_decay=self.exploration_decay,
                min_epsilon=self.min_exploration,
            )
        else:
            logger.warning(f"Unknown algorithm '{algorithm_name}', using contextual_thompson")
            return ContextualThompsonSampling(
                state_dim=self.state_dim,
                n_actions=NUM_ACTIONS,
            )

    def select_action(
        self,
        database: Any,  # ProgramDatabase
        island_idx: int | None = None,
        meta_prompting_success: float | None = None,
    ) -> SelectionAction:
        """Select an action given the current database state

        Args:
            database: The program database
            island_idx: Current island index
            meta_prompting_success: Meta-prompting strategy success rate

        Returns:
            Selected action
        """
        if not self.enabled:
            return SelectionAction.WEIGHTED

        # Extract state
        state = self.state_extractor.extract(
            database=database,
            island_idx=island_idx,
            meta_prompting_success=meta_prompting_success,
        )

        # During warmup, use random selection
        if self.iterations < self.warmup_iterations:
            action = random.randint(0, NUM_ACTIONS - 1)
            logger.debug(f"Warmup iteration {self.iterations}: random action {action}")
        else:
            # Get algorithm for this island (or global)
            algorithm = self._get_algorithm(island_idx)

            # Epsilon exploration
            if random.random() < self.exploration_rate:
                action = random.randint(0, NUM_ACTIONS - 1)
                logger.debug(f"Exploration: random action {action}")
            else:
                action = algorithm.select_action(state)
                logger.debug(f"Policy selected action {action}")

        # Track for later update
        self.last_action = action
        self.last_state = state

        return SelectionAction(action)

    def report_outcome(
        self,
        parent_fitness: float,
        child_fitness: float,
        island_idx: int | None = None,
    ):
        """Report the outcome of the last action

        Args:
            parent_fitness: Parent program fitness
            child_fitness: Child program fitness
            island_idx: Island index
        """
        if not self.enabled or self.last_action is None:
            return

        # Compute reward
        state = self.last_state or EvolutionState()
        reward = self.reward_calculator.compute_from_outcome(
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            state=state,
        )

        # Update algorithm
        algorithm = self._get_algorithm(island_idx)
        algorithm.update(state, self.last_action, reward)

        # Update global stats
        self.global_action_stats[self.last_action].update(reward)

        # Update state extractor
        self.state_extractor.record_outcome(
            action=self.last_action,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
        )

        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay,
        )

        # Increment iteration
        self.iterations += 1
        self.state_extractor.increment_iteration()

        logger.debug(
            f"RL update: action={self.last_action}, reward={reward:.4f}, "
            f"exploration={self.exploration_rate:.4f}"
        )

        # Clear last action
        self.last_action = None
        self.last_state = None

    def _get_algorithm(self, island_idx: int | None) -> BasePolicyLearner:
        """Get algorithm for island (or global)"""
        if not self.per_island or island_idx is None:
            return self.algorithm

        if island_idx not in self.island_algorithms:
            # Create per-island algorithm
            self.island_algorithms[island_idx] = self._create_algorithm(self.config.algorithm)

        return self.island_algorithms[island_idx]

    def set_max_iterations(self, max_iterations: int):
        """Set max iterations for state normalization"""
        self.state_extractor.max_iterations = max_iterations

    def get_action_probabilities(self, state: EvolutionState) -> dict[str, float]:
        """Get estimated action probabilities for analysis

        This is an approximation based on action values.
        """
        if not self.enabled:
            return {a.to_string(): 1.0 / NUM_ACTIONS for a in SelectionAction}

        # Get values from algorithm
        state_vec = state.to_array(self.feature_names)

        if isinstance(self.algorithm, ContextualThompsonSampling):
            # Use mean predictions
            values = [float(state_vec @ self.algorithm.mean[a]) for a in range(NUM_ACTIONS)]
        elif isinstance(self.algorithm, ContextualUCB):
            values = [float(state_vec @ self.algorithm.theta[a]) for a in range(NUM_ACTIONS)]
        else:
            values = [self.global_action_stats[a].mean_reward() for a in range(NUM_ACTIONS)]

        # Softmax to get probabilities
        max_val = max(values)
        exp_values = [math.exp(v - max_val) for v in values]
        total = sum(exp_values)
        probs = [e / total for e in exp_values]

        return {SelectionAction(a).to_string(): probs[a] for a in range(NUM_ACTIONS)}

    def get_statistics(self) -> dict[str, Any]:
        """Get policy statistics for logging"""
        return {
            "iterations": self.iterations,
            "exploration_rate": self.exploration_rate,
            "action_stats": {
                SelectionAction(a).to_string(): {
                    "uses": self.global_action_stats[a].uses,
                    "mean_reward": self.global_action_stats[a].mean_reward(),
                    "success_rate": self.global_action_stats[a].success_rate(),
                }
                for a in range(NUM_ACTIONS)
            },
            "reward_stats": self.reward_calculator.get_statistics(),
        }

    def save_state(self) -> dict[str, Any]:
        """Save policy state for checkpointing"""
        return {
            "enabled": self.enabled,
            "iterations": self.iterations,
            "exploration_rate": self.exploration_rate,
            "algorithm_state": self.algorithm.save_state() if self.enabled else {},
            "island_algorithms": {
                str(idx): alg.save_state() for idx, alg in self.island_algorithms.items()
            }
            if self.enabled
            else {},
            "state_extractor": self.state_extractor.save_state() if self.enabled else {},
            "reward_calculator": self.reward_calculator.save_state() if self.enabled else {},
            "global_action_stats": [s.to_dict() for s in self.global_action_stats],
        }

    def load_state(self, state: dict[str, Any]):
        """Load policy state from checkpoint"""
        if not self.enabled:
            return

        self.iterations = state.get("iterations", 0)
        self.exploration_rate = state.get("exploration_rate", 1.0)

        if state.get("algorithm_state"):
            self.algorithm.load_state(state["algorithm_state"])

        if "island_algorithms" in state:
            for idx_str, alg_state in state["island_algorithms"].items():
                idx = int(idx_str)
                if idx not in self.island_algorithms:
                    self.island_algorithms[idx] = self._create_algorithm(self.config.algorithm)
                self.island_algorithms[idx].load_state(alg_state)

        if "state_extractor" in state:
            self.state_extractor.load_state(state["state_extractor"])

        if "reward_calculator" in state:
            self.reward_calculator.load_state(state["reward_calculator"])

        if "global_action_stats" in state:
            stats_data = state["global_action_stats"]
            self.global_action_stats = [ActionStats.from_dict(s) for s in stats_data]
            while len(self.global_action_stats) < NUM_ACTIONS:
                self.global_action_stats.append(ActionStats())


def create_policy_learner(config: RLConfig) -> PolicyLearner:
    """Factory function to create a policy learner

    Args:
        config: RL configuration

    Returns:
        Configured PolicyLearner instance
    """
    return PolicyLearner(config)
