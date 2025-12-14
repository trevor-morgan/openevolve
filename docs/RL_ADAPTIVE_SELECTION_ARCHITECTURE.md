# RL-Based Adaptive Selection Architecture

## Overview

This document describes the design of an RL-based adaptive selection system for OpenEvolve. The system learns optimal selection policies during evolution, adapting exploration/exploitation trade-offs based on the current state of evolution.

## Design Philosophy

### Core Principles

1. **Unified Learning Framework**: Single RL system that can control multiple decisions (selection, temperature, diff/rewrite)
2. **Non-Invasive Integration**: Existing functionality works unchanged when disabled
3. **Complementary to Meta-Prompting**: RL handles "what to select", meta-prompting handles "how to prompt"
4. **Online + Offline Learning**: Learns during evolution AND from historical traces
5. **Multi-Objective Rewards**: Balances fitness improvement, diversity, and novelty

### Integration with Existing Systems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OpenEvolve Controller                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────────┐  │
│  │   Database   │◄───│  RL Policy       │◄───│  State Feature Extractor  │  │
│  │   .sample()  │    │  Learner         │    │  - Fitness stats          │  │
│  └──────┬───────┘    └────────┬─────────┘    │  - Diversity metrics      │  │
│         │                     │              │  - Progress indicators    │  │
│         │                     │              │  - Island health          │  │
│         ▼                     │              └───────────────────────────┘  │
│  ┌──────────────┐             │                                             │
│  │   Prompt     │             │              ┌───────────────────────────┐  │
│  │   Sampler    │◄────────────┼──────────────│  Meta-Prompting           │  │
│  │  + Meta-     │             │              │  (Strategy Selection)     │  │
│  │   Prompting  │             │              └───────────────────────────┘  │
│  └──────┬───────┘             │                                             │
│         │                     ▼                                             │
│         │            ┌──────────────────┐    ┌───────────────────────────┐  │
│         │            │  Reward          │◄───│  Evolution Trace          │  │
│         │            │  Calculator      │    │  (Historical Data)        │  │
│         │            └──────────────────┘    └───────────────────────────┘  │
│         ▼                                                                   │
│  ┌──────────────┐    ┌──────────────────┐                                   │
│  │   LLM        │───►│  Evaluator       │───► Fitness + Artifacts          │
│  │   Ensemble   │    │                  │                                   │
│  └──────────────┘    └──────────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## State Representation

### State Features (Rich Context)

The RL agent observes a comprehensive state vector:

```python
@dataclass
class EvolutionState:
    """Rich state representation for RL policy"""

    # Fitness Statistics (global)
    best_fitness: float              # Current best fitness
    mean_fitness: float              # Population mean
    fitness_std: float               # Population standard deviation
    fitness_improvement_rate: float  # Recent improvement rate (last N iterations)

    # Progress Indicators
    iteration: int                   # Current iteration
    normalized_iteration: float      # iteration / max_iterations
    generations_without_improvement: int

    # Diversity Metrics
    population_diversity: float      # Edit distance diversity
    archive_coverage: float          # MAP-Elites grid coverage
    unique_solutions: int            # Number of unique programs

    # Island-Specific (if using islands)
    island_idx: int
    island_best_fitness: float
    island_mean_fitness: float
    island_diversity: float
    inter_island_variance: float     # How different are islands?

    # Selection History
    recent_exploration_success: float  # Success rate of exploration
    recent_exploitation_success: float # Success rate of exploitation
    recent_weighted_success: float     # Success rate of weighted sampling

    # Meta-Prompting Integration
    current_strategy_success_rate: float  # If meta-prompting enabled
```

### Feature Normalization

All features are normalized to [0, 1] or [-1, 1] for stable learning:

```python
def normalize_features(state: EvolutionState) -> np.ndarray:
    """Convert state to normalized feature vector"""
    return np.array([
        state.normalized_iteration,
        sigmoid(state.best_fitness),
        (state.fitness_std / (state.mean_fitness + 1e-8)),  # Coefficient of variation
        tanh(state.fitness_improvement_rate * 10),
        min(state.generations_without_improvement / 100, 1.0),
        state.population_diversity,
        state.archive_coverage,
        state.recent_exploration_success,
        state.recent_exploitation_success,
        # ... etc
    ])
```

## Action Space

### Primary Actions: Selection Mode

```python
class SelectionAction(Enum):
    EXPLORATION = 0      # Random sampling for diversity
    EXPLOITATION = 1     # Elite/archive sampling for refinement
    WEIGHTED = 2         # Fitness-proportional sampling
    NOVELTY = 3          # Novelty-seeking (maximize behavioral difference)
    CURIOSITY = 4        # Sample programs with high uncertainty
```

### Extended Actions (Optional)

```python
@dataclass
class ExtendedAction:
    """Full action space for comprehensive control"""
    selection_mode: SelectionAction
    temperature_modifier: float      # -0.3 to +0.3 adjustment
    use_diff_evolution: bool         # True = diff, False = full rewrite
    island_target: int | None        # Specific island or None for current
```

## Reward Signal

### Multi-Objective Reward Function

```python
def compute_reward(
    parent_fitness: float,
    child_fitness: float,
    diversity_delta: float,
    novelty_score: float,
    weights: RewardWeights
) -> float:
    """
    Compute reward balancing multiple objectives

    Default weights:
    - fitness_improvement: 0.6
    - diversity_bonus: 0.2
    - novelty_bonus: 0.1
    - efficiency_bonus: 0.1  (reward for fewer LLM calls to improvement)
    """

    # Primary: Fitness improvement
    fitness_reward = (child_fitness - parent_fitness) * weights.fitness

    # Secondary: Diversity maintenance
    diversity_reward = diversity_delta * weights.diversity

    # Tertiary: Novelty (behavioral uniqueness)
    novelty_reward = novelty_score * weights.novelty

    # Efficiency: Bonus for quick improvements
    # (handled at episode level)

    return fitness_reward + diversity_reward + novelty_reward
```

### Reward Shaping

```python
# Shaped reward to encourage exploration early, exploitation late
def shaped_reward(base_reward: float, state: EvolutionState) -> float:
    # Early iterations: bonus for diversity
    exploration_bonus = (1 - state.normalized_iteration) * state.diversity_delta * 0.2

    # Late iterations: bonus for fitness
    exploitation_bonus = state.normalized_iteration * state.fitness_improvement * 0.2

    # Plateau penalty: negative reward for stagnation
    plateau_penalty = -0.1 if state.generations_without_improvement > 50 else 0

    return base_reward + exploration_bonus + exploitation_bonus + plateau_penalty
```

## Policy Learning Algorithms

### Primary: Contextual Thompson Sampling

Extension of meta-prompting's Thompson Sampling to handle rich state:

```python
class ContextualThompsonSampling:
    """
    Contextual bandit with Thompson Sampling

    Uses Bayesian linear regression to model reward as function of state-action.
    """

    def __init__(self, state_dim: int, n_actions: int, prior_variance: float = 1.0):
        self.n_actions = n_actions
        # Bayesian linear regression parameters per action
        self.mean = [np.zeros(state_dim) for _ in range(n_actions)]
        self.precision = [np.eye(state_dim) / prior_variance for _ in range(n_actions)]

    def select_action(self, state: np.ndarray) -> int:
        """Sample from posterior and select best action"""
        sampled_rewards = []
        for a in range(self.n_actions):
            # Sample weights from posterior
            cov = np.linalg.inv(self.precision[a])
            weights = np.random.multivariate_normal(self.mean[a], cov)
            # Predict reward
            sampled_rewards.append(state @ weights)
        return int(np.argmax(sampled_rewards))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Bayesian update after observing reward"""
        # Update precision matrix
        self.precision[action] += np.outer(state, state)
        # Update mean
        self.mean[action] = np.linalg.solve(
            self.precision[action],
            self.precision[action] @ self.mean[action] + reward * state
        )
```

### Alternative: Neural Contextual Bandit

For more complex state-action relationships:

```python
class NeuralContextualBandit:
    """
    Neural network-based contextual bandit

    Uses a small MLP to predict action values from state.
    Supports both online updates and batch training from traces.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        self.network = MLP(
            input_dim=state_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=n_actions
        )
        self.optimizer = Adam(self.network.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.network(torch.tensor(state))
            return q_values.argmax().item()

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Add to replay buffer and optionally train"""
        self.replay_buffer.add(state, action, reward, next_state)
        if len(self.replay_buffer) >= 32:
            self._train_step()
```

### Offline Learning from Evolution Traces

```python
class OfflineTraceTrainer:
    """
    Train policy from historical evolution traces

    Converts evolution traces to (state, action, reward) tuples
    and trains the policy offline.
    """

    def train_from_traces(self, traces: list[EvolutionTrace], policy: PolicyLearner):
        """Extract training data from traces and train policy"""

        for trace in traces:
            # Reconstruct state from trace metadata
            state = self.extract_state_from_trace(trace)

            # Infer action from trace (what selection mode was used)
            action = self.infer_action_from_trace(trace)

            # Compute reward from fitness improvement
            reward = self.compute_reward_from_trace(trace)

            # Update policy
            policy.update(state, action, reward)

    def extract_state_from_trace(self, trace: EvolutionTrace) -> np.ndarray:
        """Reconstruct state features from trace data"""
        return np.array([
            trace.iteration / 1000,  # Normalized iteration
            trace.parent_metrics.get('combined_score', 0),
            trace.improvement_delta.get('combined_score', 0),
            trace.generation / 100,
            # ... reconstruct other features from metadata
        ])
```

## Integration Points

### 1. Database Integration

```python
# database.py modifications

class ProgramDatabase:
    def __init__(self, config: DatabaseConfig, rl_policy: PolicyLearner | None = None):
        self.rl_policy = rl_policy
        self.state_extractor = StateFeatureExtractor(self)

    def sample(self, num_inspirations: int = None) -> tuple[Program, list[Program]]:
        if self.rl_policy and self.rl_policy.enabled:
            # Get current state
            state = self.state_extractor.extract()

            # Select action via learned policy
            action = self.rl_policy.select_action(state)

            # Execute action
            parent = self._execute_selection_action(action)
        else:
            # Fall back to existing logic
            parent = self._sample_parent()

        inspirations = self._sample_inspirations(parent, n=num_inspirations)
        return parent, inspirations

    def report_selection_outcome(
        self,
        action: SelectionAction,
        parent_fitness: float,
        child_fitness: float,
        diversity_delta: float
    ):
        """Report outcome for RL learning"""
        if self.rl_policy and self.rl_policy.enabled:
            state = self.state_extractor.extract()
            reward = self.reward_calculator.compute(
                parent_fitness, child_fitness, diversity_delta
            )
            self.rl_policy.update(state, action, reward)
```

### 2. Controller Integration

```python
# controller.py modifications

class OpenEvolveController:
    async def initialize(self):
        # ... existing initialization ...

        # Initialize RL policy if enabled
        if self.config.rl.enabled:
            self.rl_policy = PolicyLearner(
                config=self.config.rl,
                state_dim=self.config.rl.state_dim,
                n_actions=len(SelectionAction)
            )

            # Load from checkpoint if available
            if checkpoint_path:
                self.rl_policy.load_state(checkpoint_path / "rl_policy_state.json")

            # Optionally pre-train from historical traces
            if self.config.rl.pretrain_from_traces:
                traces = load_evolution_traces(self.config.rl.trace_path)
                self.rl_policy.pretrain(traces)

            # Inject into database
            self.database.rl_policy = self.rl_policy
```

### 3. Evolution Trace Integration

```python
# Extend evolution trace to capture RL-relevant data

@dataclass
class EvolutionTrace:
    # ... existing fields ...

    # RL-specific fields
    selection_action: int | None = None      # Which action was taken
    state_features: list[float] | None = None  # State at selection time
    selection_mode: str | None = None        # Human-readable action name
```

### 4. Checkpoint Integration

```python
# RL state saved alongside other checkpoint data

def _save_checkpoint(self, checkpoint_path: str):
    # ... existing saves ...

    # Save RL policy state
    if self.rl_policy:
        rl_state = self.rl_policy.save_state()
        rl_path = os.path.join(checkpoint_path, "rl_policy_state.json")
        with open(rl_path, "w") as f:
            json.dump(rl_state, f, indent=2)
```

## Configuration

```yaml
# config.yaml

rl:
  enabled: false                    # Enable RL-based selection

  # Algorithm selection
  algorithm: "contextual_thompson"  # "contextual_thompson", "neural_bandit", "ucb"

  # State features
  state_features:
    - "normalized_iteration"
    - "best_fitness"
    - "fitness_improvement_rate"
    - "population_diversity"
    - "archive_coverage"
    - "recent_exploration_success"
    - "recent_exploitation_success"
    - "island_variance"

  # Reward weights
  reward:
    fitness_weight: 0.6
    diversity_weight: 0.2
    novelty_weight: 0.1
    efficiency_weight: 0.1

  # Learning parameters
  learning_rate: 0.01
  discount_factor: 0.99            # For temporal credit assignment
  exploration_bonus: 0.1           # UCB exploration constant

  # Warmup
  warmup_iterations: 100           # Random policy during warmup

  # Offline training
  pretrain_from_traces: false      # Pre-train from historical traces
  trace_path: null                 # Path to evolution traces

  # Extended actions (optional)
  control_temperature: false       # Also learn temperature adjustment
  control_diff_mode: false         # Also learn diff vs full rewrite

  # Neural bandit specific
  neural:
    hidden_dim: 64
    batch_size: 32
    replay_buffer_size: 10000
```

## Usage Examples

### Basic Usage

```python
from openevolve import OpenEvolve

# Enable RL in config
config = {
    "rl": {
        "enabled": True,
        "algorithm": "contextual_thompson"
    }
}

# Run evolution - RL learns automatically
evo = OpenEvolve(config)
best_program, metrics = await evo.run(
    initial_program="...",
    evaluator="...",
    iterations=1000
)
```

### Pre-training from Historical Data

```python
from openevolve.rl import PolicyLearner, OfflineTraceTrainer
from openevolve.evolution_trace import extract_evolution_trace_from_checkpoint

# Load traces from previous runs
traces = extract_evolution_trace_from_checkpoint("checkpoints/run_001/")

# Create and pre-train policy
policy = PolicyLearner(config)
trainer = OfflineTraceTrainer()
trainer.train_from_traces(traces, policy)

# Save pre-trained policy
policy.save_state("pretrained_policy.json")
```

### Analyzing Learned Policy

```python
# After evolution, analyze what the policy learned
policy = evo.rl_policy

# Get action probabilities for different states
early_state = {"normalized_iteration": 0.1, "diversity": 0.8, ...}
late_state = {"normalized_iteration": 0.9, "diversity": 0.3, ...}

print("Early evolution preferences:", policy.get_action_probs(early_state))
# Expected: Higher exploration probability

print("Late evolution preferences:", policy.get_action_probs(late_state))
# Expected: Higher exploitation probability
```

## Performance Considerations

1. **Lightweight**: Contextual bandits add <1ms per selection
2. **Memory**: ~1MB for 10k-sample replay buffer
3. **No External Dependencies**: Pure NumPy implementation for bandits
4. **Optional PyTorch**: Neural bandit only if torch available

## Testing Strategy

1. **Unit Tests**: Individual components (state extractor, reward calculator, policy)
2. **Integration Tests**: Full pipeline with mock evaluator
3. **Regression Tests**: Ensure RL-disabled mode unchanged
4. **Learning Tests**: Verify policy improves on synthetic tasks

## Future Extensions

1. **Hierarchical RL**: High-level strategy selection + low-level parameter tuning
2. **Multi-Agent**: Each island has its own policy
3. **Transfer Learning**: Pre-train on one task, fine-tune on another
4. **Curriculum Learning**: Automatically adjust problem difficulty
