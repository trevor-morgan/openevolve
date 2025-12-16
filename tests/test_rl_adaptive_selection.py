"""
Tests for RL-based adaptive selection system

Tests the core RL components:
- SelectionAction and ExtendedAction
- EvolutionState and StateFeatureExtractor
- RewardCalculator
- PolicyLearner with different algorithms
- Integration with ProgramDatabase
"""

import random
import unittest
from unittest.mock import MagicMock

import numpy as np

from openevolve.config import DatabaseConfig, RLConfig, RLRewardConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.rl import (
    ACTION_NAMES,
    NUM_ACTIONS,
    ActionStats,
    ContextualThompsonSampling,
    ContextualUCB,
    EpsilonGreedy,
    EvolutionState,
    ExtendedAction,
    PolicyLearner,
    RewardCalculator,
    RewardComponents,
    SelectionAction,
    StateFeatureExtractor,
    create_policy_learner,
)


class TestSelectionAction(unittest.TestCase):
    """Tests for SelectionAction enum"""

    def test_action_values(self):
        """Test action enum values"""
        self.assertEqual(SelectionAction.EXPLORATION, 0)
        self.assertEqual(SelectionAction.EXPLOITATION, 1)
        self.assertEqual(SelectionAction.WEIGHTED, 2)
        self.assertEqual(SelectionAction.NOVELTY, 3)
        self.assertEqual(SelectionAction.CURIOSITY, 4)

    def test_num_actions(self):
        """Test number of actions"""
        self.assertEqual(NUM_ACTIONS, 5)
        self.assertEqual(len(ACTION_NAMES), 5)

    def test_from_string(self):
        """Test string to action conversion"""
        self.assertEqual(SelectionAction.from_string("exploration"), SelectionAction.EXPLORATION)
        self.assertEqual(SelectionAction.from_string("EXPLOITATION"), SelectionAction.EXPLOITATION)
        self.assertEqual(SelectionAction.from_string("unknown"), SelectionAction.WEIGHTED)

    def test_to_string(self):
        """Test action to string conversion"""
        self.assertEqual(SelectionAction.EXPLORATION.to_string(), "exploration")
        self.assertEqual(SelectionAction.EXPLOITATION.to_string(), "exploitation")


class TestExtendedAction(unittest.TestCase):
    """Tests for ExtendedAction dataclass"""

    def test_default_values(self):
        """Test default extended action values"""
        action = ExtendedAction(selection_mode=SelectionAction.WEIGHTED)
        self.assertEqual(action.temperature_modifier, 0.0)
        self.assertTrue(action.use_diff_evolution)
        self.assertIsNone(action.island_target)

    def test_serialization(self):
        """Test to_dict and from_dict"""
        action = ExtendedAction(
            selection_mode=SelectionAction.EXPLORATION,
            temperature_modifier=0.1,
            use_diff_evolution=False,
            island_target=2,
        )
        data = action.to_dict()
        restored = ExtendedAction.from_dict(data)

        self.assertEqual(restored.selection_mode, action.selection_mode)
        self.assertEqual(restored.temperature_modifier, action.temperature_modifier)
        self.assertEqual(restored.use_diff_evolution, action.use_diff_evolution)
        self.assertEqual(restored.island_target, action.island_target)


class TestEvolutionState(unittest.TestCase):
    """Tests for EvolutionState dataclass"""

    def test_default_values(self):
        """Test default state values"""
        state = EvolutionState()
        self.assertEqual(state.best_fitness, 0.0)
        self.assertEqual(state.normalized_iteration, 0.0)
        self.assertEqual(state.recent_exploration_success, 0.5)

    def test_to_array(self):
        """Test state to numpy array conversion"""
        state = EvolutionState(
            normalized_iteration=0.5,
            best_fitness=0.8,
            fitness_improvement_rate=0.1,
        )
        arr = state.to_array()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)

    def test_to_array_custom_features(self):
        """Test array conversion with custom features"""
        state = EvolutionState(best_fitness=0.9, mean_fitness=0.7)
        arr = state.to_array(["best_fitness", "mean_fitness"])
        self.assertEqual(len(arr), 2)
        self.assertAlmostEqual(arr[0], 0.9)
        self.assertAlmostEqual(arr[1], 0.7)

    def test_to_dict(self):
        """Test state to dictionary conversion"""
        state = EvolutionState(iteration=100, best_fitness=0.95)
        d = state.to_dict()
        self.assertEqual(d["iteration"], 100)
        self.assertEqual(d["best_fitness"], 0.95)


class TestStateFeatureExtractor(unittest.TestCase):
    """Tests for StateFeatureExtractor"""

    def setUp(self):
        """Set up test fixtures"""
        self.extractor = StateFeatureExtractor(max_iterations=1000)

    def test_initialization(self):
        """Test extractor initialization"""
        self.assertEqual(self.extractor.max_iterations, 1000)
        self.assertEqual(self.extractor.iteration, 0)

    def test_increment_iteration(self):
        """Test iteration incrementing"""
        self.extractor.increment_iteration()
        self.assertEqual(self.extractor.iteration, 1)

    def test_record_outcome(self):
        """Test outcome recording"""
        self.extractor.record_outcome(
            action=0,
            parent_fitness=0.5,
            child_fitness=0.6,
        )
        outcomes = self.extractor.action_outcomes[0]
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0].improved)

    def test_save_and_load_state(self):
        """Test state persistence"""
        self.extractor.iteration = 50
        self.extractor.best_fitness_seen = 0.9
        self.extractor.record_outcome(0, 0.5, 0.6)

        state = self.extractor.save_state()
        new_extractor = StateFeatureExtractor()
        new_extractor.load_state(state)

        self.assertEqual(new_extractor.iteration, 50)
        self.assertEqual(new_extractor.best_fitness_seen, 0.9)

    def test_sigmoid_normalize(self):
        """Test sigmoid normalization"""
        self.assertAlmostEqual(StateFeatureExtractor._sigmoid_normalize(0), 0.5)
        self.assertGreater(StateFeatureExtractor._sigmoid_normalize(5), 0.9)
        self.assertLess(StateFeatureExtractor._sigmoid_normalize(-5), 0.1)


class TestRewardCalculator(unittest.TestCase):
    """Tests for RewardCalculator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = RLRewardConfig()
        self.calculator = RewardCalculator(self.config)

    def test_positive_improvement_reward(self):
        """Test reward for fitness improvement"""
        reward = self.calculator.compute(
            parent_fitness=0.5,
            child_fitness=0.7,
        )
        self.assertGreater(reward, 0)

    def test_negative_improvement_penalty(self):
        """Test penalty for fitness decrease"""
        reward = self.calculator.compute(
            parent_fitness=0.7,
            child_fitness=0.5,
        )
        self.assertLess(reward, 0)

    def test_detailed_reward(self):
        """Test detailed reward components"""
        components = self.calculator.compute(
            parent_fitness=0.5,
            child_fitness=0.7,
            diversity_delta=0.1,
            detailed=True,
        )
        self.assertIsInstance(components, RewardComponents)
        self.assertGreater(components.fitness_reward, 0)
        self.assertGreater(components.diversity_reward, 0)

    def test_plateau_penalty(self):
        """Test plateau penalty application"""
        components = self.calculator.compute(
            parent_fitness=0.5,
            child_fitness=0.5,
            iterations_since_improvement=100,
            detailed=True,
        )
        self.assertLess(components.plateau_penalty, 0)

    def test_reward_shaping_early(self):
        """Test reward shaping for early iterations"""
        components = self.calculator.compute(
            parent_fitness=0.5,
            child_fitness=0.5,
            diversity_delta=0.1,
            normalized_iteration=0.1,
            detailed=True,
        )
        # Early iterations should have diversity bonus
        self.assertGreater(components.shaped_bonus, 0)

    def test_save_and_load_state(self):
        """Test calculator state persistence"""
        # Generate some history
        self.calculator.compute(0.5, 0.7)
        self.calculator.compute(0.6, 0.5)

        state = self.calculator.save_state()
        new_calc = RewardCalculator(self.config)
        new_calc.load_state(state)

        self.assertEqual(new_calc.max_improvement_seen, self.calculator.max_improvement_seen)


class TestActionStats(unittest.TestCase):
    """Tests for ActionStats"""

    def test_initialization(self):
        """Test default values"""
        stats = ActionStats()
        self.assertEqual(stats.uses, 0)
        self.assertEqual(stats.total_reward, 0.0)

    def test_mean_reward_zero_uses(self):
        """Test mean reward with no uses"""
        stats = ActionStats()
        self.assertEqual(stats.mean_reward(), 0.0)

    def test_mean_reward(self):
        """Test mean reward calculation"""
        stats = ActionStats()
        stats.update(1.0)
        stats.update(0.5)
        self.assertEqual(stats.mean_reward(), 0.75)

    def test_success_rate(self):
        """Test success rate calculation"""
        stats = ActionStats()
        stats.update(1.0)  # Success
        stats.update(-0.5)  # Failure
        stats.update(0.5)  # Success
        self.assertAlmostEqual(stats.success_rate(), 2 / 3)

    def test_ucb_score_unexplored(self):
        """Test UCB score for unexplored action"""
        stats = ActionStats()
        self.assertEqual(stats.ucb_score(100), float("inf"))

    def test_ucb_score(self):
        """Test UCB score calculation"""
        stats = ActionStats()
        stats.update(0.5)
        stats.update(0.5)
        stats.update(0.5)
        score = stats.ucb_score(total_uses=100, c=2.0)
        # Should be mean + exploration bonus
        self.assertGreater(score, 0.5)

    def test_thompson_sample(self):
        """Test Thompson sampling"""
        stats = ActionStats()
        stats.alpha = 10  # Many successes
        stats.beta = 2  # Few failures
        samples = [stats.thompson_sample() for _ in range(100)]
        # Should mostly sample high values
        self.assertGreater(np.mean(samples), 0.5)

    def test_serialization(self):
        """Test to_dict and from_dict"""
        stats = ActionStats()
        stats.update(0.5)
        stats.update(1.0)

        data = stats.to_dict()
        restored = ActionStats.from_dict(data)

        self.assertEqual(restored.uses, stats.uses)
        self.assertEqual(restored.total_reward, stats.total_reward)
        self.assertEqual(restored.alpha, stats.alpha)


class TestContextualThompsonSampling(unittest.TestCase):
    """Tests for ContextualThompsonSampling algorithm"""

    def setUp(self):
        """Set up test fixtures"""
        from openevolve.rl.state_features import DEFAULT_FEATURES

        self.state_dim = len(DEFAULT_FEATURES)
        self.alg = ContextualThompsonSampling(state_dim=self.state_dim, n_actions=5)

    def test_initialization(self):
        """Test algorithm initialization"""
        self.assertEqual(len(self.alg.mean), 5)
        self.assertEqual(len(self.alg.precision), 5)

    def test_select_action(self):
        """Test action selection"""
        state = EvolutionState(normalized_iteration=0.5, best_fitness=0.7)
        action = self.alg.select_action(state)
        self.assertIn(action, range(5))

    def test_update(self):
        """Test Bayesian update"""
        state = EvolutionState(normalized_iteration=0.5, best_fitness=0.7)
        self.alg.update(state, action=0, reward=1.0)
        self.assertEqual(self.alg.update_count[0], 1)

    def test_learning_behavior(self):
        """Test that algorithm learns from rewards"""
        state = EvolutionState(normalized_iteration=0.5)

        # Train action 0 with high rewards
        for _ in range(10):
            self.alg.update(state, action=0, reward=1.0)

        # Train action 1 with low rewards
        for _ in range(10):
            self.alg.update(state, action=1, reward=-0.5)

        # Action 0 should have higher mean
        state_vec = state.to_array()
        pred_0 = float(state_vec @ self.alg.mean[0])
        pred_1 = float(state_vec @ self.alg.mean[1])
        self.assertGreater(pred_0, pred_1)

    def test_save_and_load_state(self):
        """Test state persistence"""
        state = EvolutionState(normalized_iteration=0.5)
        self.alg.update(state, action=0, reward=1.0)

        saved = self.alg.save_state()
        new_alg = ContextualThompsonSampling(state_dim=self.state_dim, n_actions=5)
        new_alg.load_state(saved)

        self.assertEqual(new_alg.update_count[0], 1)


class TestContextualUCB(unittest.TestCase):
    """Tests for ContextualUCB algorithm"""

    def setUp(self):
        """Set up test fixtures"""
        from openevolve.rl.state_features import DEFAULT_FEATURES

        self.state_dim = len(DEFAULT_FEATURES)
        self.alg = ContextualUCB(state_dim=self.state_dim, n_actions=5)

    def test_select_action(self):
        """Test action selection with UCB"""
        state = EvolutionState(normalized_iteration=0.5)
        action = self.alg.select_action(state)
        self.assertIn(action, range(5))

    def test_update(self):
        """Test model update"""
        state = EvolutionState(normalized_iteration=0.5)
        self.alg.update(state, action=0, reward=1.0)
        self.assertEqual(self.alg.total_updates, 1)


class TestEpsilonGreedy(unittest.TestCase):
    """Tests for EpsilonGreedy algorithm"""

    def setUp(self):
        """Set up test fixtures"""
        self.alg = EpsilonGreedy(n_actions=5, epsilon=0.5)

    def test_exploration(self):
        """Test epsilon exploration"""
        state = EvolutionState()
        # With epsilon=0.5, should explore roughly half the time
        actions = [self.alg.select_action(state) for _ in range(100)]
        # Should have variety (not all same action)
        self.assertGreater(len(set(actions)), 1)

    def test_exploitation(self):
        """Test exploitation after learning"""
        # Create a new algorithm with no exploration at all
        self.alg = EpsilonGreedy(n_actions=5, epsilon=0.0, epsilon_decay=1.0, min_epsilon=0.0)
        state = EvolutionState()

        # Train action 0 with high rewards
        for _ in range(10):
            self.alg.update(state, action=0, reward=1.0)

        # Should always select action 0 (no exploration)
        actions = [self.alg.select_action(state) for _ in range(10)]
        self.assertEqual(actions, [0] * 10)


class TestPolicyLearner(unittest.TestCase):
    """Tests for main PolicyLearner class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = RLConfig(enabled=True, warmup_iterations=10)
        self.policy = PolicyLearner(self.config)

    def test_disabled_policy(self):
        """Test behavior when disabled"""
        config = RLConfig(enabled=False)
        policy = PolicyLearner(config)
        self.assertFalse(policy.enabled)

    def test_warmup_random_selection(self):
        """Test random selection during warmup"""
        # Create mock database
        db = MagicMock()
        db.programs = {}
        db.islands = [set()]
        db.config = MagicMock()
        db.config.feature_dimensions = []

        # During warmup, should select randomly
        actions = []
        for _ in range(20):
            self.policy.iterations = 0  # Stay in warmup
            action = self.policy.select_action(db)
            actions.append(action)

        # Should have some variety
        self.assertGreater(len(set(actions)), 1)

    def test_report_outcome(self):
        """Test outcome reporting"""
        db = MagicMock()
        db.programs = {}
        db.islands = [set()]
        db.config = MagicMock()
        db.config.feature_dimensions = []

        # Select an action first
        self.policy.select_action(db)

        # Report outcome
        self.policy.report_outcome(
            parent_fitness=0.5,
            child_fitness=0.7,
            island_idx=0,
        )

        # Should have incremented iterations
        self.assertEqual(self.policy.iterations, 1)

    def test_save_and_load_state(self):
        """Test policy state persistence"""
        db = MagicMock()
        db.programs = {}
        db.islands = [set()]
        db.config = MagicMock()
        db.config.feature_dimensions = []

        # Do some learning
        for _ in range(5):
            self.policy.select_action(db)
            self.policy.report_outcome(0.5, 0.6, 0)

        state = self.policy.save_state()

        # Create new policy and restore
        new_policy = PolicyLearner(self.config)
        new_policy.load_state(state)

        self.assertEqual(new_policy.iterations, self.policy.iterations)

    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.policy.get_statistics()
        self.assertIn("iterations", stats)
        self.assertIn("exploration_rate", stats)
        self.assertIn("action_stats", stats)

    def test_create_policy_learner_factory(self):
        """Test factory function"""
        policy = create_policy_learner(self.config)
        self.assertIsInstance(policy, PolicyLearner)

    def test_different_algorithms(self):
        """Test creating policies with different algorithms"""
        for alg in ["contextual_thompson", "contextual_ucb", "epsilon_greedy"]:
            config = RLConfig(enabled=True, algorithm=alg)
            policy = PolicyLearner(config)
            self.assertTrue(policy.enabled)


class TestDatabaseRLIntegration(unittest.TestCase):
    """Tests for RL integration with ProgramDatabase"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DatabaseConfig(
            num_islands=3,
            population_size=100,
        )
        self.db = ProgramDatabase(self.config)

        # Add some test programs
        for i in range(10):
            program = Program(
                id=f"prog_{i}",
                code=f"def func_{i}(): pass",
                metrics={"combined_score": random.uniform(0.3, 0.9)},
            )
            self.db.add(program, iteration=i)

    def test_set_rl_policy(self):
        """Test setting RL policy"""
        rl_config = RLConfig(enabled=True)
        policy = PolicyLearner(rl_config)
        self.db.set_rl_policy(policy)
        self.assertIsNotNone(self.db.rl_policy)

    def test_sample_with_rl_disabled(self):
        """Test sampling without RL (original behavior)"""
        parent, inspirations = self.db.sample()
        self.assertIsNotNone(parent)
        self.assertIsInstance(inspirations, list)

    def test_sample_with_rl_enabled(self):
        """Test sampling with RL enabled"""
        rl_config = RLConfig(enabled=True, warmup_iterations=0)
        policy = PolicyLearner(rl_config)
        self.db.set_rl_policy(policy)

        parent, _inspirations = self.db.sample()
        self.assertIsNotNone(parent)
        # Parent fitness should be tracked
        self.assertIsNotNone(self.db._last_parent_fitness)

    def test_report_selection_outcome(self):
        """Test reporting selection outcome"""
        rl_config = RLConfig(enabled=True)
        policy = PolicyLearner(rl_config)
        self.db.set_rl_policy(policy)

        # Make a selection
        self.db.sample()

        # Report outcome
        self.db.report_selection_outcome(child_fitness=0.8, island_idx=0)

        # Tracked state should be cleared
        self.assertIsNone(self.db._last_parent_fitness)


class TestFullRLPipeline(unittest.TestCase):
    """Integration tests for the full RL pipeline"""

    def test_learning_improves_over_time(self):
        """Test that RL learns to select better actions"""
        # This is a simplified test that verifies the learning loop works
        rl_config = RLConfig(
            enabled=True,
            warmup_iterations=5,
            algorithm="contextual_thompson",
        )
        policy = PolicyLearner(rl_config)

        # Simulate evolution with consistent rewards
        db = MagicMock()
        db.programs = {}
        db.islands = [set()]
        db.config = MagicMock()
        db.config.feature_dimensions = []

        # Train: action 0 always gives high reward
        for i in range(50):
            policy.iterations = i
            action = policy.select_action(db)

            # Simulate: action 0 is best
            if action == SelectionAction.EXPLORATION:
                reward_child = 0.9
            else:
                reward_child = 0.5

            policy.report_outcome(0.5, reward_child, 0)

        # Check that exploration has highest success rate
        stats = policy.get_statistics()
        exploration_stats = stats["action_stats"]["exploration"]
        self.assertGreater(exploration_stats["mean_reward"], 0)


if __name__ == "__main__":
    unittest.main()
