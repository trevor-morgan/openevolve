"""
Meta-prompting system for OpenEvolve

Enables evolving prompt strategies alongside code optimization using
multi-armed bandit algorithms (Thompson Sampling, UCB, Epsilon-Greedy).
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openevolve.config import MetaPromptConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Statistics for a single meta-prompt strategy

    Tracks performance metrics for a strategy including:
    - Usage counts and total reward
    - Beta distribution parameters for Thompson Sampling
    - Sum of squared rewards for UCB variance calculation
    - Recent history for trend analysis
    """

    name: str

    # Core statistics
    total_uses: int = 0
    total_reward: float = 0.0

    # For Thompson Sampling (Beta distribution)
    # Prior starts at (1, 1) = uniform distribution
    successes: float = 1.0  # Alpha parameter
    failures: float = 1.0  # Beta parameter

    # For UCB variance calculation
    sum_squared_reward: float = 0.0

    # Detailed history (bounded to prevent memory bloat)
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_contexts: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def mean_reward(self) -> float:
        """Average reward across all uses"""
        if self.total_uses == 0:
            return 0.0
        return self.total_reward / self.total_uses

    @property
    def variance(self) -> float:
        """Variance of rewards (for UCB)"""
        if self.total_uses < 2:
            return float("inf")
        mean = self.mean_reward
        return (self.sum_squared_reward / self.total_uses) - (mean**2)

    @property
    def success_rate(self) -> float:
        """Proportion of positive outcomes"""
        total = self.successes + self.failures - 2  # Subtract prior
        if total <= 0:
            return 0.5  # Prior mean
        return (self.successes - 1) / total

    def ucb_score(self, total_iterations: int, c: float = 2.0) -> float:
        """Upper Confidence Bound score

        UCB = mean + c * sqrt(log(n) / n_i)

        Args:
            total_iterations: Total iterations across all strategies
            c: Exploration constant (higher = more exploration)

        Returns:
            UCB score (higher = should be selected)
        """
        if self.total_uses == 0:
            return float("inf")  # Unexplored strategy has infinite potential

        exploitation = self.mean_reward
        exploration = c * math.sqrt(math.log(max(1, total_iterations)) / self.total_uses)
        return exploitation + exploration

    def thompson_sample(self) -> float:
        """Sample from Beta posterior for Thompson Sampling

        Returns a sample from Beta(successes, failures) distribution.
        Higher successes relative to failures = samples closer to 1.
        """
        return random.betavariate(max(0.1, self.successes), max(0.1, self.failures))

    def update(self, reward: float, context: dict | None = None) -> None:
        """Update statistics with new observation

        Args:
            reward: Observed reward (can be negative for regressions)
            context: Optional context dict for analysis
        """
        self.total_uses += 1
        self.total_reward += reward
        self.sum_squared_reward += reward**2
        self.recent_rewards.append(reward)

        # Update Beta distribution for Thompson Sampling
        # Interpret reward as probability of success
        if reward > 0:
            # Positive reward increases successes
            self.successes += min(reward, 1.0)  # Cap at 1 to prevent explosion
        else:
            # Negative or zero reward increases failures
            self.failures += min(abs(reward), 1.0) if reward < 0 else 0.1

        if context:
            self.recent_contexts.append(context)

    def get_recent_trend(self, window: int = 10) -> float:
        """Get recent performance trend

        Args:
            window: Number of recent observations to consider

        Returns:
            Average of recent rewards (0 if no history)
        """
        if not self.recent_rewards:
            return 0.0
        recent = list(self.recent_rewards)[-window:]
        return sum(recent) / len(recent)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint"""
        return {
            "name": self.name,
            "total_uses": self.total_uses,
            "total_reward": self.total_reward,
            "successes": self.successes,
            "failures": self.failures,
            "sum_squared_reward": self.sum_squared_reward,
            "recent_rewards": list(self.recent_rewards),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyStats:
        """Deserialize from checkpoint"""
        stats = cls(name=data["name"])
        stats.total_uses = data.get("total_uses", 0)
        stats.total_reward = data.get("total_reward", 0.0)
        stats.successes = data.get("successes", 1.0)
        stats.failures = data.get("failures", 1.0)
        stats.sum_squared_reward = data.get("sum_squared_reward", 0.0)
        stats.recent_rewards = deque(data.get("recent_rewards", []), maxlen=100)
        return stats


# Default meta-prompt strategies
DEFAULT_STRATEGIES = {
    # Algorithmic strategies
    "algorithmic_restructure": {
        "fragment": (
            "Consider restructuring the algorithm entirely. Look for opportunities "
            "to use different data structures, change the computational approach, "
            "or apply well-known algorithmic patterns (divide-and-conquer, dynamic "
            "programming, greedy, memoization)."
        ),
        "tags": ["high_risk", "exploration", "algorithmic"],
        "suggested_contexts": ["low_fitness", "plateau"],
    },
    "incremental_refinement": {
        "fragment": (
            "Make small, targeted improvements to the existing approach. Focus on "
            "optimizing constants, simplifying expressions, removing unnecessary "
            "computations, or improving edge case handling."
        ),
        "tags": ["low_risk", "exploitation", "refinement"],
        "suggested_contexts": ["high_fitness", "near_optimum"],
    },
    # Performance strategies
    "vectorization": {
        "fragment": (
            "Look for opportunities to vectorize operations using NumPy or similar "
            "libraries. Eliminate Python loops where possible and leverage SIMD "
            "instructions through array operations. Consider broadcasting for "
            "element-wise operations."
        ),
        "tags": ["performance", "numpy", "loops"],
        "suggested_contexts": ["has_loops", "numerical"],
    },
    "memory_optimization": {
        "fragment": (
            "Focus on memory access patterns and allocation. Consider cache locality, "
            "reduce unnecessary allocations, use in-place operations where safe, "
            "minimize data copying, and prefer views over copies."
        ),
        "tags": ["performance", "memory", "cache"],
        "suggested_contexts": ["large_data", "memory_bound"],
    },
    "parallelization": {
        "fragment": (
            "Identify independent computations that can run in parallel. Consider "
            "multiprocessing for CPU-bound work, threading for I/O-bound operations, "
            "or vectorized operations that leverage multiple cores."
        ),
        "tags": ["performance", "parallel", "concurrency"],
        "suggested_contexts": ["cpu_bound", "independent_ops"],
    },
    # Code quality strategies
    "simplification": {
        "fragment": (
            "Simplify the code by removing redundancy, combining similar operations, "
            "and using more expressive idioms. Simpler code often performs better "
            "and is easier to optimize further."
        ),
        "tags": ["simplification", "readability", "pythonic"],
        "suggested_contexts": ["high_complexity", "long_code"],
    },
    "mathematical_reformulation": {
        "fragment": (
            "Look for mathematical identities, algebraic simplifications, or "
            "alternative formulations that compute the same result more efficiently. "
            "Consider closed-form solutions, approximations, or series expansions."
        ),
        "tags": ["math", "algebra", "reformulation"],
        "suggested_contexts": ["numerical", "mathematical"],
    },
    # Exploration strategies
    "creative_alternative": {
        "fragment": (
            "Think creatively about completely different approaches. What would an "
            "expert in a different field try? Are there unconventional solutions "
            "from biology, physics, or other domains worth exploring?"
        ),
        "tags": ["creative", "high_risk", "exploration"],
        "suggested_contexts": ["plateau", "stuck"],
    },
    "hybrid_approach": {
        "fragment": (
            "Consider combining elements from the top-performing programs. Take the "
            "best ideas from each inspiration and synthesize them into a new unified "
            "approach that leverages multiple strengths."
        ),
        "tags": ["synthesis", "combination", "hybrid"],
        "suggested_contexts": ["diverse_inspirations", "mid_evolution"],
    },
    # Robustness strategies
    "numerical_stability": {
        "fragment": (
            "Pay attention to numerical stability. Avoid subtracting similar large "
            "numbers, use log-space for products of small numbers, handle edge cases "
            "(zeros, infinities, NaN), and consider numerical precision issues."
        ),
        "tags": ["numerical", "stability", "edge_cases"],
        "suggested_contexts": ["numerical", "floating_point"],
    },
    "error_handling": {
        "fragment": (
            "Consider edge cases and error conditions. What happens with empty inputs, "
            "extreme values, or unexpected data types? Add appropriate validation "
            "and graceful handling without sacrificing performance."
        ),
        "tags": ["robustness", "validation", "edge_cases"],
        "suggested_contexts": ["unstable", "errors"],
    },
}


class MetaPromptEvolver:
    """Evolves prompt strategies using multi-armed bandit algorithms

    Tracks which strategies lead to fitness improvements and adapts
    selection probabilities accordingly. Supports:
    - Thompson Sampling (Bayesian, good exploration-exploitation balance)
    - UCB (Upper Confidence Bound, deterministic exploration)
    - Epsilon-Greedy (simple random exploration)

    Strategies can be tracked:
    - Globally (across all programs)
    - Per-island (for island-based evolution)
    - Per-context (fitness range, generation, etc.)
    """

    def __init__(self, config: MetaPromptConfig):
        self.config = config

        # Strategy definitions: name -> {fragment, tags, suggested_contexts}
        self.strategies: dict[str, dict[str, Any]] = {}

        # Global statistics
        self.global_stats: dict[str, StrategyStats] = {}

        # Per-island statistics (if enabled)
        self.island_stats: dict[int, dict[str, StrategyStats]] = defaultdict(dict)

        # Context-specific statistics (if enabled)
        self.context_stats: dict[str, dict[str, StrategyStats]] = defaultdict(dict)

        # State tracking
        self.total_iterations: int = 0
        self.exploration_rate: float = 1.0

        # Track last selection for reward attribution
        self._last_strategy: str | None = None
        self._last_island: int | None = None
        self._last_context: dict | None = None

        # Load strategies
        self._load_strategies()

        logger.info(
            f"Initialized MetaPromptEvolver with {len(self.strategies)} strategies, "
            f"algorithm={config.selection_algorithm}"
        )

    def _load_strategies(self) -> None:
        """Load meta-prompt strategies from file or defaults"""
        # Start with defaults
        self.strategies = dict(DEFAULT_STRATEGIES)

        # Load custom strategies if specified
        if self.config.strategies_file:
            strategies_path = Path(self.config.strategies_file)
            if strategies_path.exists():
                try:
                    with open(strategies_path) as f:
                        custom = json.load(f)
                    if "strategies" in custom:
                        self.strategies.update(custom["strategies"])
                    logger.info(f"Loaded custom strategies from {strategies_path}")
                except Exception as e:
                    logger.warning(f"Failed to load custom strategies: {e}")

        # Initialize global stats for all strategies
        for name in self.strategies:
            self.global_stats[name] = StrategyStats(
                name=name,
                successes=self.config.thompson_prior_alpha,
                failures=self.config.thompson_prior_beta,
            )

    def select_strategy(
        self,
        island_idx: int | None = None,
        context: dict | None = None,
    ) -> tuple[str, str]:
        """Select a meta-prompt strategy using the configured algorithm

        Args:
            island_idx: Current island (for per-island adaptation)
            context: Current context dict with keys like:
                - fitness: Current fitness score (0-1)
                - generation: Current generation number
                - complexity: Code complexity
                - code_length: Length of code in characters

        Returns:
            Tuple of (strategy_name, strategy_fragment)
        """
        self.total_iterations += 1

        # Warmup: random selection to gather initial data
        if self.total_iterations <= self.config.warmup_iterations:
            name = random.choice(list(self.strategies.keys()))
            self._record_selection(name, island_idx, context)
            return name, self.strategies[name]["fragment"]

        # Get relevant stats based on granularity configuration
        stats = self._get_relevant_stats(island_idx, context)

        # Ensure all strategies have stats
        for name in self.strategies:
            if name not in stats:
                stats[name] = StrategyStats(
                    name=name,
                    successes=self.config.thompson_prior_alpha,
                    failures=self.config.thompson_prior_beta,
                )

        # Apply exploration decay
        self.exploration_rate = max(
            self.config.min_exploration,
            self.exploration_rate * self.config.exploration_decay,
        )

        # Select using configured algorithm
        if self.config.selection_algorithm == "thompson_sampling":
            selected = self._thompson_sampling_select(stats)
        elif self.config.selection_algorithm == "ucb":
            selected = self._ucb_select(stats)
        elif self.config.selection_algorithm == "epsilon_greedy":
            selected = self._epsilon_greedy_select(stats)
        else:
            logger.warning(f"Unknown algorithm: {self.config.selection_algorithm}, using random")
            selected = random.choice(list(self.strategies.keys()))

        self._record_selection(selected, island_idx, context)
        return selected, self.strategies[selected]["fragment"]

    def _record_selection(self, name: str, island_idx: int | None, context: dict | None) -> None:
        """Record selection for later reward attribution"""
        self._last_strategy = name
        self._last_island = island_idx
        self._last_context = context

    def _thompson_sampling_select(self, stats: dict[str, StrategyStats]) -> str:
        """Thompson Sampling: sample from posterior, select max

        Each strategy has a Beta distribution representing our belief about
        its success rate. We sample from each distribution and select the
        strategy with the highest sample.
        """
        samples = {name: s.thompson_sample() for name, s in stats.items()}
        return max(samples, key=lambda k: samples[k])

    def _ucb_select(self, stats: dict[str, StrategyStats]) -> str:
        """UCB: select strategy with highest upper confidence bound

        UCB balances exploitation (high mean reward) with exploration
        (high uncertainty for under-sampled strategies).
        """
        scores = {
            name: s.ucb_score(self.total_iterations, self.config.ucb_exploration_constant)
            for name, s in stats.items()
        }
        return max(scores, key=lambda k: scores[k])

    def _epsilon_greedy_select(self, stats: dict[str, StrategyStats]) -> str:
        """Epsilon-greedy: explore with probability epsilon

        With probability epsilon * exploration_rate, select randomly.
        Otherwise, select the strategy with highest mean reward.
        """
        if random.random() < self.config.epsilon * self.exploration_rate:
            return random.choice(list(stats.keys()))

        # Exploit: select best mean reward
        return max(stats.keys(), key=lambda n: stats[n].mean_reward)

    def _get_relevant_stats(
        self,
        island_idx: int | None,
        context: dict | None,
    ) -> dict[str, StrategyStats]:
        """Get statistics relevant to current selection context

        Blends global, island-specific, and context-specific stats
        based on configuration.
        """
        # Start with global stats (copy to avoid mutation)
        stats = {name: self._copy_stats(s) for name, s in self.global_stats.items()}

        # Blend in island-specific stats if enabled
        if self.config.per_island_strategies and island_idx is not None:
            island_specific = self.island_stats.get(island_idx, {})
            for name, island_stat in island_specific.items():
                if name in stats:
                    stats[name] = self._blend_stats(stats[name], island_stat, weight=0.7)
                else:
                    stats[name] = self._copy_stats(island_stat)

        # Blend in context-specific stats if enabled
        if self.config.context_aware and context:
            context_key = self._context_to_key(context)
            context_specific = self.context_stats.get(context_key, {})
            for name, ctx_stat in context_specific.items():
                if name in stats:
                    stats[name] = self._blend_stats(stats[name], ctx_stat, weight=0.5)
                else:
                    stats[name] = self._copy_stats(ctx_stat)

        return stats

    def _copy_stats(self, stats: StrategyStats) -> StrategyStats:
        """Create a copy of StrategyStats"""
        copy = StrategyStats(name=stats.name)
        copy.total_uses = stats.total_uses
        copy.total_reward = stats.total_reward
        copy.successes = stats.successes
        copy.failures = stats.failures
        copy.sum_squared_reward = stats.sum_squared_reward
        return copy

    def _blend_stats(
        self,
        global_stat: StrategyStats,
        local_stat: StrategyStats,
        weight: float = 0.5,
    ) -> StrategyStats:
        """Blend global and local statistics

        Args:
            global_stat: Global statistics
            local_stat: Local (island or context) statistics
            weight: Weight for local stats (0-1)
        """
        blended = StrategyStats(name=global_stat.name)

        # Weighted combination
        blended.total_uses = global_stat.total_uses + local_stat.total_uses
        blended.total_reward = (
            1 - weight
        ) * global_stat.total_reward + weight * local_stat.total_reward
        blended.successes = (1 - weight) * global_stat.successes + weight * local_stat.successes
        blended.failures = (1 - weight) * global_stat.failures + weight * local_stat.failures
        blended.sum_squared_reward = (
            1 - weight
        ) * global_stat.sum_squared_reward + weight * local_stat.sum_squared_reward

        return blended

    def _context_to_key(self, context: dict) -> str:
        """Convert context dict to hashable key for caching"""
        # Discretize continuous values into buckets
        fitness = context.get("fitness", 0.5)
        fitness_bucket = "low" if fitness < 0.3 else "mid" if fitness < 0.7 else "high"

        generation = context.get("generation", 0)
        gen_bucket = "early" if generation < 50 else "mid" if generation < 200 else "late"

        return f"{fitness_bucket}_{gen_bucket}"

    def update_reward(
        self,
        strategy_name: str,
        reward: float,
        island_idx: int | None = None,
        context: dict | None = None,
    ) -> None:
        """Update strategy statistics with observed reward

        Args:
            strategy_name: Which strategy was used
            reward: Computed reward (positive = improvement)
            island_idx: Which island this occurred on
            context: Context in which strategy was applied
        """
        # Update global stats
        if strategy_name not in self.global_stats:
            self.global_stats[strategy_name] = StrategyStats(name=strategy_name)
        self.global_stats[strategy_name].update(reward, context)

        # Update island-specific stats
        if self.config.per_island_strategies and island_idx is not None:
            if strategy_name not in self.island_stats[island_idx]:
                self.island_stats[island_idx][strategy_name] = StrategyStats(name=strategy_name)
            self.island_stats[island_idx][strategy_name].update(reward, context)

        # Update context-specific stats
        if self.config.context_aware and context:
            context_key = self._context_to_key(context)
            if strategy_name not in self.context_stats[context_key]:
                self.context_stats[context_key][strategy_name] = StrategyStats(name=strategy_name)
            self.context_stats[context_key][strategy_name].update(reward, context)

        logger.debug(
            f"Updated strategy '{strategy_name}': reward={reward:.4f}, "
            f"mean={self.global_stats[strategy_name].mean_reward:.4f}, "
            f"uses={self.global_stats[strategy_name].total_uses}"
        )

    def report_outcome(
        self,
        parent_fitness: float,
        child_fitness: float,
        island_idx: int | None = None,
    ) -> None:
        """Report mutation outcome for the last selected strategy

        Convenience method that computes reward and updates stats.

        Args:
            parent_fitness: Parent program's fitness score
            child_fitness: Child program's fitness score
            island_idx: Which island (uses last selection if None)
        """
        if not self._last_strategy:
            return

        reward = self.compute_reward(parent_fitness, child_fitness)
        self.update_reward(
            strategy_name=self._last_strategy,
            reward=reward,
            island_idx=island_idx or self._last_island,
            context=self._last_context,
        )

        # Clear last selection
        self._last_strategy = None
        self._last_island = None
        self._last_context = None

    def compute_reward(
        self,
        parent_fitness: float,
        child_fitness: float,
    ) -> float:
        """Compute reward for a mutation based on fitness change

        Args:
            parent_fitness: Parent program's fitness
            child_fitness: Child program's fitness

        Returns:
            Reward value (positive = improvement)
        """
        delta = child_fitness - parent_fitness

        if self.config.reward_type == "improvement":
            # Raw improvement delta
            if abs(delta) > self.config.improvement_threshold:
                return delta
            return 0.0

        elif self.config.reward_type == "normalized":
            # Normalize by parent fitness to handle different scales
            if parent_fitness > 0.001:
                return delta / parent_fitness
            return delta

        elif self.config.reward_type == "rank":
            # Binary: improved or not
            return 1.0 if delta > self.config.improvement_threshold else 0.0

        return delta

    def get_strategy_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all strategy performance for logging/visualization

        Returns:
            Dict mapping strategy name to performance metrics
        """
        summary = {}
        for name, stats in self.global_stats.items():
            summary[name] = {
                "uses": stats.total_uses,
                "mean_reward": stats.mean_reward,
                "success_rate": stats.success_rate,
                "recent_trend": stats.get_recent_trend(),
                "successes": stats.successes,
                "failures": stats.failures,
            }
        return summary

    def get_island_summary(self, island_idx: int) -> dict[str, dict[str, Any]]:
        """Get strategy performance summary for a specific island"""
        if island_idx not in self.island_stats:
            return {}

        summary = {}
        for name, stats in self.island_stats[island_idx].items():
            summary[name] = {
                "uses": stats.total_uses,
                "mean_reward": stats.mean_reward,
                "success_rate": stats.success_rate,
                "recent_trend": stats.get_recent_trend(),
            }
        return summary

    def save_state(self) -> dict[str, Any]:
        """Serialize state for checkpoint"""
        return {
            "total_iterations": self.total_iterations,
            "exploration_rate": self.exploration_rate,
            "global_stats": {name: stats.to_dict() for name, stats in self.global_stats.items()},
            "island_stats": {
                str(island): {name: stats.to_dict() for name, stats in island_stats.items()}
                for island, island_stats in self.island_stats.items()
            },
            "context_stats": {
                ctx: {name: stats.to_dict() for name, stats in ctx_stats.items()}
                for ctx, ctx_stats in self.context_stats.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint"""
        self.total_iterations = state.get("total_iterations", 0)
        self.exploration_rate = state.get("exploration_rate", 1.0)

        # Restore global stats
        self.global_stats = {}
        for name, data in state.get("global_stats", {}).items():
            self.global_stats[name] = StrategyStats.from_dict(data)

        # Ensure all strategies have stats (in case new strategies were added)
        for name in self.strategies:
            if name not in self.global_stats:
                self.global_stats[name] = StrategyStats(
                    name=name,
                    successes=self.config.thompson_prior_alpha,
                    failures=self.config.thompson_prior_beta,
                )

        # Restore island stats
        self.island_stats = defaultdict(dict)
        for island_str, island_data in state.get("island_stats", {}).items():
            island = int(island_str)
            for name, data in island_data.items():
                self.island_stats[island][name] = StrategyStats.from_dict(data)

        # Restore context stats
        self.context_stats = defaultdict(dict)
        for ctx, ctx_data in state.get("context_stats", {}).items():
            for name, data in ctx_data.items():
                self.context_stats[ctx][name] = StrategyStats.from_dict(data)

        logger.info(
            f"Restored meta-prompt state: {self.total_iterations} iterations, "
            f"{len(self.global_stats)} strategies"
        )

    def log_stats(self, iteration: int) -> None:
        """Log strategy statistics for monitoring"""
        summary = self.get_strategy_summary()

        # Sort by mean reward
        sorted_strategies = sorted(summary.items(), key=lambda x: x[1]["mean_reward"], reverse=True)

        logger.info(f"Meta-prompt strategy stats at iteration {iteration}:")
        for name, stats in sorted_strategies[:5]:  # Top 5
            logger.info(
                f"  {name}: uses={stats['uses']}, "
                f"mean={stats['mean_reward']:.4f}, "
                f"success={stats['success_rate']:.2%}, "
                f"trend={stats['recent_trend']:.4f}"
            )
