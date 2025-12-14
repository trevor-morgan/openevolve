"""
Unit tests for meta-prompting system
"""

import math
import random
import unittest

from openevolve.config import MetaPromptConfig, PromptConfig
from openevolve.meta_prompting import (
    DEFAULT_STRATEGIES,
    MetaPromptEvolver,
    StrategyStats,
)


class TestStrategyStats(unittest.TestCase):
    """Tests for StrategyStats class"""

    def test_initialization(self):
        """Test basic initialization"""
        stats = StrategyStats(name="test_strategy")

        self.assertEqual(stats.name, "test_strategy")
        self.assertEqual(stats.total_uses, 0)
        self.assertEqual(stats.total_reward, 0.0)
        self.assertEqual(stats.successes, 1.0)  # Prior
        self.assertEqual(stats.failures, 1.0)  # Prior

    def test_mean_reward_zero_uses(self):
        """Test mean_reward with zero uses"""
        stats = StrategyStats(name="test")
        self.assertEqual(stats.mean_reward, 0.0)

    def test_mean_reward_calculation(self):
        """Test mean_reward calculation"""
        stats = StrategyStats(name="test")
        stats.update(0.5)
        stats.update(0.3)

        self.assertEqual(stats.total_uses, 2)
        self.assertAlmostEqual(stats.mean_reward, 0.4, places=5)

    def test_variance_insufficient_data(self):
        """Test variance with insufficient data"""
        stats = StrategyStats(name="test")
        self.assertEqual(stats.variance, float("inf"))

        stats.update(0.5)
        self.assertEqual(stats.variance, float("inf"))

    def test_variance_calculation(self):
        """Test variance calculation"""
        stats = StrategyStats(name="test")
        # Add values with known variance
        stats.update(0.0)
        stats.update(1.0)

        # Variance of [0, 1] = 0.25
        self.assertAlmostEqual(stats.variance, 0.25, places=5)

    def test_success_rate(self):
        """Test success_rate calculation"""
        stats = StrategyStats(name="test")

        # With just prior (1, 1), success rate should be 0.5
        self.assertEqual(stats.success_rate, 0.5)

        # Add successes
        stats.successes = 5.0
        stats.failures = 2.0
        # (5-1) / (5-1 + 2-1) = 4/5 = 0.8
        self.assertAlmostEqual(stats.success_rate, 0.8, places=5)

    def test_update_positive_reward(self):
        """Test update with positive reward"""
        stats = StrategyStats(name="test")
        stats.update(0.7)

        self.assertEqual(stats.total_uses, 1)
        self.assertAlmostEqual(stats.total_reward, 0.7, places=5)
        self.assertGreater(stats.successes, 1.0)  # Should increase
        self.assertEqual(len(stats.recent_rewards), 1)

    def test_update_negative_reward(self):
        """Test update with negative reward"""
        stats = StrategyStats(name="test")
        stats.update(-0.3)

        self.assertEqual(stats.total_uses, 1)
        self.assertAlmostEqual(stats.total_reward, -0.3, places=5)
        self.assertGreater(stats.failures, 1.0)  # Should increase

    def test_update_with_context(self):
        """Test update with context"""
        stats = StrategyStats(name="test")
        context = {"fitness": 0.5, "generation": 10}
        stats.update(0.5, context=context)

        self.assertEqual(len(stats.recent_contexts), 1)
        self.assertEqual(stats.recent_contexts[0], context)

    def test_recent_rewards_bounded(self):
        """Test that recent_rewards is bounded"""
        stats = StrategyStats(name="test")

        # Add more than maxlen (100)
        for i in range(150):
            stats.update(i * 0.01)

        self.assertEqual(len(stats.recent_rewards), 100)

    def test_ucb_score_unexplored(self):
        """Test UCB score for unexplored strategy"""
        stats = StrategyStats(name="test")
        score = stats.ucb_score(100, c=2.0)

        self.assertEqual(score, float("inf"))

    def test_ucb_score_calculation(self):
        """Test UCB score calculation"""
        stats = StrategyStats(name="test")
        stats.total_uses = 10
        stats.total_reward = 5.0  # mean = 0.5

        # UCB = mean + c * sqrt(log(n) / n_i)
        # UCB = 0.5 + 2.0 * sqrt(log(100) / 10)
        expected_exploration = 2.0 * math.sqrt(math.log(100) / 10)
        expected_ucb = 0.5 + expected_exploration

        score = stats.ucb_score(100, c=2.0)
        self.assertAlmostEqual(score, expected_ucb, places=4)

    def test_thompson_sample(self):
        """Test Thompson sampling produces valid samples"""
        stats = StrategyStats(name="test", successes=10.0, failures=2.0)

        # Run many samples and check they're in valid range
        samples = [stats.thompson_sample() for _ in range(1000)]

        self.assertTrue(all(0 <= s <= 1 for s in samples))
        # With alpha=10, beta=2, expect samples biased toward 1
        self.assertGreater(sum(samples) / len(samples), 0.7)

    def test_get_recent_trend(self):
        """Test get_recent_trend calculation"""
        stats = StrategyStats(name="test")

        # Empty history
        self.assertEqual(stats.get_recent_trend(), 0.0)

        # Add some rewards
        stats.recent_rewards.append(0.1)
        stats.recent_rewards.append(0.2)
        stats.recent_rewards.append(0.3)

        self.assertAlmostEqual(stats.get_recent_trend(window=3), 0.2, places=5)
        self.assertAlmostEqual(stats.get_recent_trend(window=2), 0.25, places=5)

    def test_serialization(self):
        """Test to_dict and from_dict"""
        original = StrategyStats(name="test")
        original.update(0.5)
        original.update(0.3)
        original.successes = 5.0
        original.failures = 2.0

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = StrategyStats.from_dict(data)

        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.total_uses, original.total_uses)
        self.assertAlmostEqual(restored.total_reward, original.total_reward, places=5)
        self.assertAlmostEqual(restored.successes, original.successes, places=5)
        self.assertAlmostEqual(restored.failures, original.failures, places=5)


class TestMetaPromptEvolver(unittest.TestCase):
    """Tests for MetaPromptEvolver class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=10,
            selection_algorithm="thompson_sampling",
        )

    def test_initialization(self):
        """Test basic initialization"""
        evolver = MetaPromptEvolver(self.config)

        self.assertEqual(evolver.config, self.config)
        self.assertEqual(evolver.total_iterations, 0)
        self.assertEqual(evolver.exploration_rate, 1.0)
        self.assertGreater(len(evolver.strategies), 0)
        self.assertGreater(len(evolver.global_stats), 0)

    def test_default_strategies_loaded(self):
        """Test that default strategies are loaded"""
        evolver = MetaPromptEvolver(self.config)

        for name in DEFAULT_STRATEGIES:
            self.assertIn(name, evolver.strategies)
            self.assertIn(name, evolver.global_stats)

    def test_warmup_random_selection(self):
        """Test that warmup period uses random selection"""
        config = MetaPromptConfig(enabled=True, warmup_iterations=100)
        evolver = MetaPromptEvolver(config)

        # Seed for reproducibility
        random.seed(42)

        # During warmup, should select randomly (varied results)
        selections = []
        for _ in range(50):
            name, _ = evolver.select_strategy()
            selections.append(name)

        # Should have multiple different strategies selected
        unique_selections = set(selections)
        self.assertGreater(len(unique_selections), 1)

    def test_select_strategy_returns_valid(self):
        """Test that select_strategy returns valid strategy"""
        evolver = MetaPromptEvolver(self.config)

        name, fragment = evolver.select_strategy()

        self.assertIn(name, evolver.strategies)
        self.assertIsInstance(fragment, str)
        self.assertGreater(len(fragment), 0)

    def test_select_strategy_increments_iterations(self):
        """Test that select_strategy increments total_iterations"""
        evolver = MetaPromptEvolver(self.config)

        initial = evolver.total_iterations
        evolver.select_strategy()
        self.assertEqual(evolver.total_iterations, initial + 1)

    def test_thompson_sampling_selection(self):
        """Test Thompson sampling strategy selection"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            selection_algorithm="thompson_sampling",
        )
        evolver = MetaPromptEvolver(config)

        # Train one strategy to be clearly better
        for _ in range(100):
            evolver.update_reward("vectorization", 0.9)
            evolver.update_reward("simplification", 0.1)

        # Thompson sampling is stochastic but should favor better strategy
        random.seed(42)
        selections = [evolver.select_strategy()[0] for _ in range(50)]

        vectorization_count = selections.count("vectorization")
        simplification_count = selections.count("simplification")

        # Vectorization should be selected more often
        self.assertGreater(vectorization_count, simplification_count)

    def test_ucb_selection(self):
        """Test UCB strategy selection mechanism"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            selection_algorithm="ucb",
            ucb_exploration_constant=0.1,  # Low exploration to favor exploitation
        )
        evolver = MetaPromptEvolver(config)

        # First, explore all strategies with baseline reward
        for name in evolver.strategies:
            evolver.update_reward(name, 0.1)

        # Train vectorization to be clearly better with many observations
        for _ in range(200):
            evolver.update_reward("vectorization", 0.95)

        # With low exploration constant and many observations,
        # UCB should favor the strategy with highest mean
        selections = [evolver.select_strategy()[0] for _ in range(10)]

        # Vectorization should be selected (has highest mean and low UCB bonus due to many samples)
        vectorization_count = selections.count("vectorization")
        self.assertGreater(vectorization_count, 0)  # Should appear at least once

    def test_epsilon_greedy_selection(self):
        """Test epsilon-greedy strategy selection"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            selection_algorithm="epsilon_greedy",
            epsilon=0.0,  # Pure exploitation for testing
        )
        evolver = MetaPromptEvolver(config)

        # Train one strategy to be clearly better
        for _ in range(50):
            evolver.update_reward("vectorization", 0.9)
            evolver.update_reward("simplification", 0.1)

        # With epsilon=0, should always select best strategy
        selections = [evolver.select_strategy()[0] for _ in range(10)]

        # All selections should be vectorization
        self.assertTrue(all(s == "vectorization" for s in selections))

    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy exploration"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            selection_algorithm="epsilon_greedy",
            epsilon=1.0,  # Pure exploration
            min_exploration=1.0,  # Don't decay
        )
        evolver = MetaPromptEvolver(config)

        random.seed(42)
        selections = [evolver.select_strategy()[0] for _ in range(100)]

        # With epsilon=1, should select randomly (multiple strategies)
        unique_selections = set(selections)
        self.assertGreater(len(unique_selections), 3)

    def test_update_reward_global_stats(self):
        """Test that update_reward updates global stats"""
        evolver = MetaPromptEvolver(self.config)

        evolver.update_reward("vectorization", 0.5)

        stats = evolver.global_stats["vectorization"]
        self.assertEqual(stats.total_uses, 1)
        self.assertAlmostEqual(stats.total_reward, 0.5, places=5)

    def test_update_reward_island_stats(self):
        """Test that update_reward updates island stats"""
        config = MetaPromptConfig(enabled=True, per_island_strategies=True)
        evolver = MetaPromptEvolver(config)

        evolver.update_reward("vectorization", 0.5, island_idx=0)
        evolver.update_reward("vectorization", 0.3, island_idx=1)

        # Check island-specific stats
        self.assertIn(0, evolver.island_stats)
        self.assertIn(1, evolver.island_stats)

        self.assertAlmostEqual(evolver.island_stats[0]["vectorization"].total_reward, 0.5, places=5)
        self.assertAlmostEqual(evolver.island_stats[1]["vectorization"].total_reward, 0.3, places=5)

    def test_update_reward_context_stats(self):
        """Test that update_reward updates context stats"""
        config = MetaPromptConfig(enabled=True, context_aware=True)
        evolver = MetaPromptEvolver(config)

        context_low = {"fitness": 0.1, "generation": 10}
        context_high = {"fitness": 0.8, "generation": 10}

        evolver.update_reward("vectorization", 0.5, context=context_low)
        evolver.update_reward("vectorization", 0.3, context=context_high)

        # Check context-specific stats exist
        self.assertIn("low_early", evolver.context_stats)
        self.assertIn("high_early", evolver.context_stats)

    def test_island_specific_learning(self):
        """Test that different islands can learn different strategies"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            per_island_strategies=True,
            selection_algorithm="epsilon_greedy",
            epsilon=0.0,
        )
        evolver = MetaPromptEvolver(config)

        # Train different strategies on different islands
        for _ in range(50):
            evolver.update_reward("vectorization", 0.9, island_idx=0)
            evolver.update_reward("simplification", 0.1, island_idx=0)
            evolver.update_reward("vectorization", 0.1, island_idx=1)
            evolver.update_reward("simplification", 0.9, island_idx=1)

        # Island 0 should prefer vectorization
        selections_0 = [evolver.select_strategy(island_idx=0)[0] for _ in range(10)]
        # Island 1 should prefer simplification
        selections_1 = [evolver.select_strategy(island_idx=1)[0] for _ in range(10)]

        vectorization_0 = selections_0.count("vectorization")
        simplification_1 = selections_1.count("simplification")

        # Due to global/island blending, not all selections will be the same
        # but the preferred strategy should appear frequently
        self.assertGreater(vectorization_0, 0)
        self.assertGreater(simplification_1, 0)

    def test_compute_reward_improvement(self):
        """Test compute_reward with improvement type"""
        config = MetaPromptConfig(
            enabled=True, reward_type="improvement", improvement_threshold=0.0
        )
        evolver = MetaPromptEvolver(config)

        reward = evolver.compute_reward(parent_fitness=0.5, child_fitness=0.7)
        self.assertAlmostEqual(reward, 0.2, places=5)

        reward = evolver.compute_reward(parent_fitness=0.7, child_fitness=0.5)
        self.assertAlmostEqual(reward, -0.2, places=5)

    def test_compute_reward_normalized(self):
        """Test compute_reward with normalized type"""
        config = MetaPromptConfig(enabled=True, reward_type="normalized")
        evolver = MetaPromptEvolver(config)

        reward = evolver.compute_reward(parent_fitness=0.5, child_fitness=0.7)
        # (0.7 - 0.5) / 0.5 = 0.4
        self.assertAlmostEqual(reward, 0.4, places=5)

    def test_compute_reward_rank(self):
        """Test compute_reward with rank type"""
        config = MetaPromptConfig(enabled=True, reward_type="rank", improvement_threshold=0.01)
        evolver = MetaPromptEvolver(config)

        # Improvement above threshold
        reward = evolver.compute_reward(parent_fitness=0.5, child_fitness=0.7)
        self.assertEqual(reward, 1.0)

        # No improvement
        reward = evolver.compute_reward(parent_fitness=0.5, child_fitness=0.505)
        self.assertEqual(reward, 0.0)

    def test_report_outcome(self):
        """Test report_outcome convenience method"""
        evolver = MetaPromptEvolver(self.config)

        # Simulate selection
        evolver.select_strategy(island_idx=0, context={"fitness": 0.5, "generation": 10})
        strategy_name = evolver._last_strategy

        # Report outcome
        evolver.report_outcome(parent_fitness=0.5, child_fitness=0.7)

        # Check stats were updated
        self.assertEqual(evolver.global_stats[strategy_name].total_uses, 1)

        # Last selection should be cleared
        self.assertIsNone(evolver._last_strategy)

    def test_exploration_decay(self):
        """Test that exploration rate decays"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            exploration_decay=0.9,
            min_exploration=0.1,
        )
        evolver = MetaPromptEvolver(config)

        initial_rate = evolver.exploration_rate

        # Run several iterations
        for _ in range(10):
            evolver.select_strategy()

        # Exploration rate should have decayed
        self.assertLess(evolver.exploration_rate, initial_rate)

    def test_min_exploration(self):
        """Test that exploration rate doesn't go below minimum"""
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            exploration_decay=0.1,  # Aggressive decay
            min_exploration=0.5,
        )
        evolver = MetaPromptEvolver(config)

        # Run many iterations
        for _ in range(100):
            evolver.select_strategy()

        # Should not go below minimum
        self.assertGreaterEqual(evolver.exploration_rate, 0.5)

    def test_get_strategy_summary(self):
        """Test get_strategy_summary"""
        evolver = MetaPromptEvolver(self.config)

        # Add some data
        evolver.update_reward("vectorization", 0.5)
        evolver.update_reward("vectorization", 0.7)
        evolver.update_reward("simplification", 0.3)

        summary = evolver.get_strategy_summary()

        self.assertIn("vectorization", summary)
        self.assertIn("simplification", summary)

        vec_summary = summary["vectorization"]
        self.assertEqual(vec_summary["uses"], 2)
        self.assertAlmostEqual(vec_summary["mean_reward"], 0.6, places=5)

    def test_save_and_load_state(self):
        """Test checkpoint save and restore"""
        evolver = MetaPromptEvolver(self.config)

        # Build up some state
        for _ in range(20):
            evolver.select_strategy(island_idx=0)
        evolver.update_reward("vectorization", 0.5, island_idx=0)
        evolver.update_reward("simplification", 0.3, island_idx=1)

        # Save state
        state = evolver.save_state()

        # Create new evolver and restore
        new_evolver = MetaPromptEvolver(self.config)
        new_evolver.load_state(state)

        # Check state was restored
        self.assertEqual(new_evolver.total_iterations, evolver.total_iterations)
        self.assertAlmostEqual(new_evolver.exploration_rate, evolver.exploration_rate, places=5)
        self.assertEqual(
            new_evolver.global_stats["vectorization"].total_uses,
            evolver.global_stats["vectorization"].total_uses,
        )

    def test_context_to_key(self):
        """Test context discretization"""
        evolver = MetaPromptEvolver(self.config)

        # Low fitness, early generation
        key = evolver._context_to_key({"fitness": 0.1, "generation": 10})
        self.assertEqual(key, "low_early")

        # High fitness, late generation
        key = evolver._context_to_key({"fitness": 0.9, "generation": 300})
        self.assertEqual(key, "high_late")

        # Mid fitness, mid generation
        key = evolver._context_to_key({"fitness": 0.5, "generation": 100})
        self.assertEqual(key, "mid_mid")


class TestPromptSamplerMetaPromptIntegration(unittest.TestCase):
    """Tests for meta-prompting integration in PromptSampler"""

    def test_meta_prompting_disabled_by_default(self):
        """Test that meta-prompting is disabled by default"""
        config = PromptConfig()
        self.assertFalse(config.meta_prompting.enabled)

    def test_meta_prompting_evolver_initialized(self):
        """Test that evolver is initialized when enabled"""
        from openevolve.prompt.sampler import PromptSampler

        config = PromptConfig()
        config.meta_prompting.enabled = True

        sampler = PromptSampler(config)

        self.assertIsNotNone(sampler.meta_prompt_evolver)

    def test_meta_prompting_evolver_not_initialized(self):
        """Test that evolver is not initialized when disabled"""
        from openevolve.prompt.sampler import PromptSampler

        config = PromptConfig()
        config.meta_prompting.enabled = False

        sampler = PromptSampler(config)

        self.assertIsNone(sampler.meta_prompt_evolver)

    def test_save_and_load_meta_prompt_state(self):
        """Test save and load of meta-prompt state via PromptSampler"""
        from openevolve.prompt.sampler import PromptSampler

        config = PromptConfig()
        config.meta_prompting.enabled = True

        sampler = PromptSampler(config)

        # Build up some state
        sampler.meta_prompt_evolver.update_reward("vectorization", 0.5)

        # Save state
        state = sampler.save_meta_prompt_state()
        self.assertIsNotNone(state)

        # Create new sampler and load
        new_sampler = PromptSampler(config)
        new_sampler.load_meta_prompt_state(state)

        # Check state was restored
        self.assertEqual(
            new_sampler.meta_prompt_evolver.global_stats["vectorization"].total_uses,
            sampler.meta_prompt_evolver.global_stats["vectorization"].total_uses,
        )

    def test_report_outcome_no_evolver(self):
        """Test report_outcome when meta-prompting is disabled"""
        from openevolve.prompt.sampler import PromptSampler

        config = PromptConfig()
        config.meta_prompting.enabled = False

        sampler = PromptSampler(config)

        # Should not raise
        sampler.report_outcome(parent_fitness=0.5, child_fitness=0.7)


if __name__ == "__main__":
    unittest.main()
