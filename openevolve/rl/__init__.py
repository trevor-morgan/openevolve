"""
RL-based adaptive selection for OpenEvolve

This module provides reinforcement learning capabilities for learning
optimal selection policies during evolution. It complements meta-prompting:
RL handles "what to select", meta-prompting handles "how to prompt".

Main components:
- PolicyLearner: Main interface for action selection and learning
- StateFeatureExtractor: Extracts rich state features from evolution state
- RewardCalculator: Computes multi-objective rewards
- SelectionAction: Action space definition

Usage:
    from openevolve.rl import PolicyLearner, SelectionAction
    from openevolve.config import RLConfig

    # Create policy learner
    config = RLConfig(enabled=True, algorithm="contextual_thompson")
    policy = PolicyLearner(config)

    # During evolution
    action = policy.select_action(database, island_idx=0)

    # ... execute action and evaluate child ...

    # Report outcome
    policy.report_outcome(parent_fitness, child_fitness, island_idx=0)
"""

from openevolve.rl.actions import (
    ACTION_NAMES,
    NUM_ACTIONS,
    ExtendedAction,
    SelectionAction,
)
from openevolve.rl.policy import (
    ActionStats,
    BasePolicyLearner,
    ContextualThompsonSampling,
    ContextualUCB,
    EpsilonGreedy,
    PolicyLearner,
    create_policy_learner,
)
from openevolve.rl.reward import RewardCalculator, RewardComponents
from openevolve.rl.state_features import (
    DEFAULT_FEATURES,
    ActionOutcome,
    EvolutionState,
    StateFeatureExtractor,
)

__all__ = [
    # Actions
    "SelectionAction",
    "ExtendedAction",
    "NUM_ACTIONS",
    "ACTION_NAMES",
    # Policy
    "PolicyLearner",
    "create_policy_learner",
    "BasePolicyLearner",
    "ContextualThompsonSampling",
    "ContextualUCB",
    "EpsilonGreedy",
    "ActionStats",
    # State
    "EvolutionState",
    "StateFeatureExtractor",
    "ActionOutcome",
    "DEFAULT_FEATURES",
    # Reward
    "RewardCalculator",
    "RewardComponents",
]
