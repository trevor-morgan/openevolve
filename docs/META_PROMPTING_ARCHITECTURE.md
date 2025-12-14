# Meta-Prompting Architecture for OpenEvolve

## Executive Summary

Meta-prompting enables OpenEvolve to **evolve its mutation strategies alongside the code it optimizes**. Instead of using fixed prompts, the system learns which prompt strategies lead to improvements for specific problem types, islands, and fitness ranges.

## Design Principles

1. **Non-invasive Integration**: Hooks into existing `PromptSampler` without breaking current functionality
2. **Statistically Rigorous**: Uses multi-armed bandit algorithms with proper uncertainty quantification
3. **Checkpoint Compatible**: Full state persistence and restoration
4. **Island-Aware**: Strategies can specialize per-island for different exploration niches
5. **Observable**: Rich metrics for debugging and analysis

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Controller                                      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐│
│  │  PromptSampler  │───▶│ MetaPromptEvolver │───▶│ MetaPromptStrategy      ││
│  │                 │    │                  │    │ (per strategy tracking) ││
│  │ build_prompt()  │    │ select_strategy()│    │ - Thompson Sampling     ││
│  │                 │    │ update_reward()  │    │ - UCB                   ││
│  └────────┬────────┘    └────────┬─────────┘    │ - Epsilon-Greedy        ││
│           │                      │              └─────────────────────────┘│
│           │                      │                                          │
│           ▼                      ▼                                          │
│  ┌─────────────────┐    ┌──────────────────┐                               │
│  │TemplateManager  │    │StrategyTracker   │                               │
│  │                 │    │                  │                               │
│  │ get_fragment()  │    │ per-island stats │                               │
│  │ meta_fragments  │    │ effectiveness    │                               │
│  └─────────────────┘    │ history          │                               │
│                         └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Database                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Program                                                                  ││
│  │  - metadata["meta_prompt_strategy"]: str                                ││
│  │  - metadata["meta_prompt_context"]: dict                                ││
│  │  - metadata["strategy_reward"]: float (computed post-evaluation)        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ MetaPromptState (new, persisted in checkpoint)                          ││
│  │  - strategy_stats: dict[str, StrategyStats]                             ││
│  │  - island_strategy_stats: dict[int, dict[str, StrategyStats]]          ││
│  │  - context_strategy_stats: dict[str, dict[str, StrategyStats]]         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. MetaPromptConfig (config.py)

```python
@dataclass
class MetaPromptConfig:
    """Configuration for meta-prompting system"""

    # Feature flag
    enabled: bool = False

    # Strategy selection algorithm
    selection_algorithm: Literal["thompson_sampling", "ucb", "epsilon_greedy"] = "thompson_sampling"

    # Algorithm-specific parameters
    ucb_exploration_constant: float = 2.0      # C in UCB formula
    epsilon: float = 0.1                        # For epsilon-greedy
    thompson_prior_alpha: float = 1.0           # Beta distribution prior
    thompson_prior_beta: float = 1.0

    # Reward calculation
    reward_type: Literal["improvement", "rank", "normalized"] = "improvement"
    improvement_threshold: float = 0.0          # Minimum delta for positive reward
    reward_decay: float = 0.99                  # Exponential decay for old observations

    # Blending with base prompt
    meta_prompt_weight: float = 0.1             # Weight of meta-prompt in final prompt
    meta_prompt_position: Literal["prefix", "suffix", "section"] = "section"

    # Strategy granularity
    per_island_strategies: bool = True          # Track strategies per island
    context_aware: bool = True                  # Adapt based on fitness range, generation, etc.

    # Exploration vs exploitation over time
    exploration_decay: float = 0.995            # Reduce exploration over iterations
    min_exploration: float = 0.05               # Minimum exploration rate

    # History and persistence
    max_history_per_strategy: int = 1000        # Cap history to prevent memory bloat
    warmup_iterations: int = 50                 # Random selection during warmup
```

### 2. Strategy Categories

Strategies are organized hierarchically for different mutation goals:

```python
# openevolve/prompts/defaults/meta_fragments.json
{
    "strategies": {
        # Algorithmic strategies
        "algorithmic_restructure": {
            "fragment": "Consider restructuring the algorithm entirely. Look for opportunities to use different data structures, change the computational approach, or apply well-known algorithmic patterns (divide-and-conquer, dynamic programming, greedy).",
            "tags": ["high_risk", "exploration", "algorithmic"],
            "suggested_contexts": ["low_fitness", "plateau"]
        },
        "incremental_refinement": {
            "fragment": "Make small, targeted improvements to the existing approach. Focus on optimizing constants, simplifying expressions, or removing unnecessary computations.",
            "tags": ["low_risk", "exploitation", "refinement"],
            "suggested_contexts": ["high_fitness", "near_optimum"]
        },

        # Performance strategies
        "vectorization": {
            "fragment": "Look for opportunities to vectorize operations using NumPy, eliminate Python loops, and leverage SIMD instructions through array operations.",
            "tags": ["performance", "numpy", "loops"],
            "suggested_contexts": ["has_loops", "numerical"]
        },
        "memory_optimization": {
            "fragment": "Focus on memory access patterns. Consider cache locality, reduce allocations, use in-place operations, and minimize data copying.",
            "tags": ["performance", "memory", "cache"],
            "suggested_contexts": ["large_data", "memory_bound"]
        },
        "parallelization": {
            "fragment": "Identify independent computations that can run in parallel. Consider multiprocessing for CPU-bound work or async for I/O-bound operations.",
            "tags": ["performance", "parallel", "concurrency"],
            "suggested_contexts": ["cpu_bound", "independent_ops"]
        },

        # Code quality strategies
        "simplification": {
            "fragment": "Simplify the code by removing redundancy, combining similar operations, and using more expressive Python idioms. Simpler code often performs better.",
            "tags": ["simplification", "readability", "pythonic"],
            "suggested_contexts": ["high_complexity", "long_code"]
        },
        "mathematical_reformulation": {
            "fragment": "Look for mathematical identities, algebraic simplifications, or alternative formulations that compute the same result more efficiently.",
            "tags": ["math", "algebra", "reformulation"],
            "suggested_contexts": ["numerical", "mathematical"]
        },

        # Exploration strategies
        "creative_alternative": {
            "fragment": "Think creatively about completely different approaches. What would an expert in a different field try? Are there unconventional solutions worth exploring?",
            "tags": ["creative", "high_risk", "exploration"],
            "suggested_contexts": ["plateau", "stuck"]
        },
        "hybrid_approach": {
            "fragment": "Consider combining elements from the top-performing programs. Take the best ideas from each and synthesize them into a new approach.",
            "tags": ["synthesis", "combination", "hybrid"],
            "suggested_contexts": ["diverse_inspirations", "mid_evolution"]
        },

        # Domain-specific (examples)
        "numerical_stability": {
            "fragment": "Pay attention to numerical stability. Avoid subtracting similar numbers, use log-space for products, and handle edge cases (zeros, infinities, NaN).",
            "tags": ["numerical", "stability", "edge_cases"],
            "suggested_contexts": ["numerical", "floating_point"]
        }
    },

    "contexts": {
        "low_fitness": {"fitness_range": [0.0, 0.3]},
        "mid_fitness": {"fitness_range": [0.3, 0.7]},
        "high_fitness": {"fitness_range": [0.7, 1.0]},
        "plateau": {"no_improvement_generations": 10},
        "early_evolution": {"generation_range": [0, 50]},
        "mid_evolution": {"generation_range": [50, 200]},
        "late_evolution": {"generation_range": [200, null]},
        "high_complexity": {"complexity_percentile": 0.8},
        "long_code": {"code_length_percentile": 0.8}
    }
}
```

### 3. StrategyStats Class

```python
@dataclass
class StrategyStats:
    """Statistics for a single meta-prompt strategy"""

    name: str

    # Core statistics
    total_uses: int = 0
    total_reward: float = 0.0

    # For Thompson Sampling (Beta distribution)
    successes: float = 1.0    # Alpha (prior + observed)
    failures: float = 1.0     # Beta (prior + observed)

    # For UCB
    sum_squared_reward: float = 0.0

    # Detailed history (bounded)
    recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_contexts: deque[dict] = field(default_factory=lambda: deque(maxlen=100))

    # Derived metrics (computed)
    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.total_uses)

    @property
    def variance(self) -> float:
        if self.total_uses < 2:
            return float('inf')
        mean = self.mean_reward
        return (self.sum_squared_reward / self.total_uses) - (mean ** 2)

    @property
    def ucb_score(self, total_iterations: int, c: float = 2.0) -> float:
        """Upper Confidence Bound score"""
        if self.total_uses == 0:
            return float('inf')
        exploitation = self.mean_reward
        exploration = c * math.sqrt(math.log(total_iterations) / self.total_uses)
        return exploitation + exploration

    def thompson_sample(self) -> float:
        """Sample from Beta posterior for Thompson Sampling"""
        return random.betavariate(self.successes, self.failures)

    def update(self, reward: float, context: dict | None = None):
        """Update statistics with new observation"""
        self.total_uses += 1
        self.total_reward += reward
        self.sum_squared_reward += reward ** 2
        self.recent_rewards.append(reward)

        # Update Beta distribution (for binary reward interpretation)
        if reward > 0:
            self.successes += reward
        else:
            self.failures += abs(reward) if reward < 0 else 0.1

        if context:
            self.recent_contexts.append(context)

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> "StrategyStats":
        """Deserialize from checkpoint"""
        stats = cls(name=data["name"])
        stats.total_uses = data["total_uses"]
        stats.total_reward = data["total_reward"]
        stats.successes = data["successes"]
        stats.failures = data["failures"]
        stats.sum_squared_reward = data["sum_squared_reward"]
        stats.recent_rewards = deque(data.get("recent_rewards", []), maxlen=100)
        return stats
```

### 4. MetaPromptEvolver Class

```python
class MetaPromptEvolver:
    """
    Evolves prompt strategies using multi-armed bandit algorithms.

    Tracks which strategies lead to fitness improvements and adapts
    selection probabilities accordingly.
    """

    def __init__(self, config: MetaPromptConfig):
        self.config = config
        self.strategies: dict[str, str] = {}  # name -> fragment
        self.strategy_tags: dict[str, list[str]] = {}
        self.strategy_contexts: dict[str, list[str]] = {}

        # Global statistics
        self.global_stats: dict[str, StrategyStats] = {}

        # Per-island statistics (if enabled)
        self.island_stats: dict[int, dict[str, StrategyStats]] = defaultdict(dict)

        # Context-specific statistics (if enabled)
        self.context_stats: dict[str, dict[str, StrategyStats]] = defaultdict(dict)

        # State tracking
        self.total_iterations: int = 0
        self.exploration_rate: float = 1.0

        # Load strategies from fragments
        self._load_strategies()

    def _load_strategies(self):
        """Load meta-prompt strategies from fragments.json"""
        # Load from openevolve/prompts/defaults/meta_fragments.json
        # Populate self.strategies, self.strategy_tags, self.strategy_contexts
        pass

    def select_strategy(
        self,
        island_idx: int | None = None,
        context: dict | None = None,
    ) -> tuple[str, str]:
        """
        Select a meta-prompt strategy using the configured algorithm.

        Args:
            island_idx: Current island (for per-island adaptation)
            context: Current context (fitness, generation, code features)

        Returns:
            (strategy_name, strategy_fragment)
        """
        self.total_iterations += 1

        # Warmup: random selection
        if self.total_iterations < self.config.warmup_iterations:
            name = random.choice(list(self.strategies.keys()))
            return name, self.strategies[name]

        # Get relevant stats based on granularity
        stats = self._get_relevant_stats(island_idx, context)

        # Ensure all strategies have stats
        for name in self.strategies:
            if name not in stats:
                stats[name] = StrategyStats(name=name)

        # Apply exploration decay
        self.exploration_rate = max(
            self.config.min_exploration,
            self.exploration_rate * self.config.exploration_decay
        )

        # Select using configured algorithm
        if self.config.selection_algorithm == "thompson_sampling":
            selected = self._thompson_sampling_select(stats)
        elif self.config.selection_algorithm == "ucb":
            selected = self._ucb_select(stats)
        elif self.config.selection_algorithm == "epsilon_greedy":
            selected = self._epsilon_greedy_select(stats)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.selection_algorithm}")

        return selected, self.strategies[selected]

    def _thompson_sampling_select(self, stats: dict[str, StrategyStats]) -> str:
        """Thompson Sampling: sample from posterior, select max"""
        samples = {name: s.thompson_sample() for name, s in stats.items()}
        return max(samples, key=samples.get)

    def _ucb_select(self, stats: dict[str, StrategyStats]) -> str:
        """UCB: select strategy with highest upper confidence bound"""
        scores = {
            name: s.ucb_score(self.total_iterations, self.config.ucb_exploration_constant)
            for name, s in stats.items()
        }
        return max(scores, key=scores.get)

    def _epsilon_greedy_select(self, stats: dict[str, StrategyStats]) -> str:
        """Epsilon-greedy: explore with probability epsilon"""
        if random.random() < self.config.epsilon * self.exploration_rate:
            return random.choice(list(stats.keys()))

        # Exploit: select best mean reward
        return max(stats.keys(), key=lambda n: stats[n].mean_reward)

    def _get_relevant_stats(
        self,
        island_idx: int | None,
        context: dict | None,
    ) -> dict[str, StrategyStats]:
        """Get statistics relevant to current selection context"""
        # Start with global stats
        stats = dict(self.global_stats)

        # Blend in island-specific stats if enabled
        if self.config.per_island_strategies and island_idx is not None:
            island_specific = self.island_stats.get(island_idx, {})
            for name, island_stat in island_specific.items():
                if name in stats:
                    # Weighted blend of global and island stats
                    stats[name] = self._blend_stats(stats[name], island_stat, weight=0.7)
                else:
                    stats[name] = island_stat

        # Blend in context-specific stats if enabled
        if self.config.context_aware and context:
            context_key = self._context_to_key(context)
            context_specific = self.context_stats.get(context_key, {})
            for name, ctx_stat in context_specific.items():
                if name in stats:
                    stats[name] = self._blend_stats(stats[name], ctx_stat, weight=0.5)
                else:
                    stats[name] = ctx_stat

        return stats

    def update_reward(
        self,
        strategy_name: str,
        reward: float,
        island_idx: int | None = None,
        context: dict | None = None,
    ):
        """
        Update strategy statistics with observed reward.

        Args:
            strategy_name: Which strategy was used
            reward: Computed reward (improvement delta, normalized, etc.)
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

    def compute_reward(
        self,
        parent_metrics: dict[str, float],
        child_metrics: dict[str, float],
        feature_dimensions: list[str] | None = None,
    ) -> float:
        """
        Compute reward for a mutation based on fitness change.

        Args:
            parent_metrics: Parent program's metrics
            child_metrics: Child program's metrics (after mutation)
            feature_dimensions: Dimensions to exclude from fitness

        Returns:
            Reward value (positive = improvement)
        """
        from openevolve.utils.metrics_utils import get_fitness_score

        parent_fitness = get_fitness_score(parent_metrics, feature_dimensions)
        child_fitness = get_fitness_score(child_metrics, feature_dimensions)

        delta = child_fitness - parent_fitness

        if self.config.reward_type == "improvement":
            # Raw improvement delta
            if delta > self.config.improvement_threshold:
                return delta
            elif delta < -self.config.improvement_threshold:
                return delta  # Negative reward for regression
            else:
                return 0.0  # No significant change

        elif self.config.reward_type == "normalized":
            # Normalize by parent fitness to handle different scales
            if parent_fitness > 0:
                return delta / parent_fitness
            else:
                return delta

        elif self.config.reward_type == "rank":
            # Binary: improved or not
            return 1.0 if delta > self.config.improvement_threshold else 0.0

        return delta

    def _context_to_key(self, context: dict) -> str:
        """Convert context dict to hashable key"""
        # Discretize continuous values
        fitness_bucket = "low" if context.get("fitness", 0) < 0.3 else \
                        "mid" if context.get("fitness", 0) < 0.7 else "high"
        gen_bucket = "early" if context.get("generation", 0) < 50 else \
                    "mid" if context.get("generation", 0) < 200 else "late"
        return f"{fitness_bucket}_{gen_bucket}"

    def _blend_stats(
        self,
        global_stat: StrategyStats,
        local_stat: StrategyStats,
        weight: float = 0.5,
    ) -> StrategyStats:
        """Blend global and local statistics"""
        blended = StrategyStats(name=global_stat.name)
        blended.total_uses = global_stat.total_uses + local_stat.total_uses
        blended.total_reward = (
            (1 - weight) * global_stat.total_reward +
            weight * local_stat.total_reward
        )
        blended.successes = (
            (1 - weight) * global_stat.successes +
            weight * local_stat.successes
        )
        blended.failures = (
            (1 - weight) * global_stat.failures +
            weight * local_stat.failures
        )
        return blended

    def get_strategy_summary(self) -> dict:
        """Get summary of all strategy performance for logging/visualization"""
        return {
            name: {
                "uses": stats.total_uses,
                "mean_reward": stats.mean_reward,
                "success_rate": stats.successes / (stats.successes + stats.failures),
                "recent_trend": (
                    sum(stats.recent_rewards) / len(stats.recent_rewards)
                    if stats.recent_rewards else 0
                ),
            }
            for name, stats in self.global_stats.items()
        }

    def save_state(self) -> dict:
        """Serialize state for checkpoint"""
        return {
            "total_iterations": self.total_iterations,
            "exploration_rate": self.exploration_rate,
            "global_stats": {
                name: stats.to_dict()
                for name, stats in self.global_stats.items()
            },
            "island_stats": {
                island: {
                    name: stats.to_dict()
                    for name, stats in island_stats.items()
                }
                for island, island_stats in self.island_stats.items()
            },
            "context_stats": {
                ctx: {
                    name: stats.to_dict()
                    for name, stats in ctx_stats.items()
                }
                for ctx, ctx_stats in self.context_stats.items()
            },
        }

    def load_state(self, state: dict):
        """Restore state from checkpoint"""
        self.total_iterations = state.get("total_iterations", 0)
        self.exploration_rate = state.get("exploration_rate", 1.0)

        self.global_stats = {
            name: StrategyStats.from_dict(data)
            for name, data in state.get("global_stats", {}).items()
        }

        self.island_stats = defaultdict(dict)
        for island, island_data in state.get("island_stats", {}).items():
            self.island_stats[int(island)] = {
                name: StrategyStats.from_dict(data)
                for name, data in island_data.items()
            }

        self.context_stats = defaultdict(dict)
        for ctx, ctx_data in state.get("context_stats", {}).items():
            self.context_stats[ctx] = {
                name: StrategyStats.from_dict(data)
                for name, data in ctx_data.items()
            }
```

---

## Integration Points

### 5. PromptSampler Integration

Modify `openevolve/prompt/sampler.py`:

```python
class PromptSampler:
    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(custom_template_dir=config.template_dir)

        # Initialize meta-prompt evolver if enabled
        self.meta_prompt_evolver: MetaPromptEvolver | None = None
        if config.meta_prompting.enabled:
            self.meta_prompt_evolver = MetaPromptEvolver(config.meta_prompting)

        # Track last selected strategy for reward attribution
        self._last_strategy: str | None = None
        self._last_context: dict | None = None

    def build_prompt(
        self,
        current_program: str = "",
        program_metrics: dict[str, float] = {},
        island_idx: int | None = None,
        generation: int = 0,
        **kwargs,
    ) -> dict[str, str]:
        """Build prompt with optional meta-prompt strategy"""

        # ... existing prompt building logic ...

        # Add meta-prompt if enabled
        meta_prompt_section = ""
        if self.meta_prompt_evolver:
            context = {
                "fitness": get_fitness_score(program_metrics, kwargs.get("feature_dimensions")),
                "generation": generation,
                "complexity": program_metrics.get("complexity", 0),
                "code_length": len(current_program),
            }

            strategy_name, strategy_fragment = self.meta_prompt_evolver.select_strategy(
                island_idx=island_idx,
                context=context,
            )

            # Store for later reward attribution
            self._last_strategy = strategy_name
            self._last_context = context

            # Format meta-prompt section
            meta_prompt_section = f"\n## Strategy Guidance\n{strategy_fragment}\n"

        # Insert meta-prompt based on position config
        if meta_prompt_section:
            if self.config.meta_prompting.meta_prompt_position == "prefix":
                user_message = meta_prompt_section + user_message
            elif self.config.meta_prompting.meta_prompt_position == "suffix":
                user_message = user_message + meta_prompt_section
            else:  # "section"
                # Insert after improvement areas, before code
                user_message = self._insert_meta_prompt_section(
                    user_message, meta_prompt_section
                )

        return {"system": system_message, "user": user_message}

    def report_outcome(
        self,
        parent_metrics: dict[str, float],
        child_metrics: dict[str, float],
        island_idx: int | None = None,
        feature_dimensions: list[str] | None = None,
    ):
        """Report mutation outcome for meta-prompt learning"""
        if not self.meta_prompt_evolver or not self._last_strategy:
            return

        reward = self.meta_prompt_evolver.compute_reward(
            parent_metrics=parent_metrics,
            child_metrics=child_metrics,
            feature_dimensions=feature_dimensions,
        )

        self.meta_prompt_evolver.update_reward(
            strategy_name=self._last_strategy,
            reward=reward,
            island_idx=island_idx,
            context=self._last_context,
        )

        # Clear for next iteration
        self._last_strategy = None
        self._last_context = None
```

### 6. Iteration Worker Integration

Modify `openevolve/iteration.py`:

```python
async def run_iteration(...):
    # ... existing sampling and prompt building ...

    # Build prompt (meta-prompt selection happens here)
    prompt = prompt_sampler.build_prompt(
        current_program=parent.code,
        program_metrics=parent.metrics,
        island_idx=parent.metadata.get("island"),
        generation=parent.generation,
        feature_dimensions=database.config.feature_dimensions,
        # ... other params ...
    )

    # ... LLM generation and evaluation ...

    # Report outcome for meta-prompt learning
    if child_program and child_program.metrics:
        prompt_sampler.report_outcome(
            parent_metrics=parent.metrics,
            child_metrics=child_program.metrics,
            island_idx=parent.metadata.get("island"),
            feature_dimensions=database.config.feature_dimensions,
        )

    # Store strategy in child metadata for analysis
    if child_program and prompt_sampler._last_strategy:
        child_program.metadata["meta_prompt_strategy"] = prompt_sampler._last_strategy
        child_program.metadata["meta_prompt_context"] = prompt_sampler._last_context
```

### 7. Checkpoint Integration

Modify `openevolve/controller.py`:

```python
def _save_checkpoint(self, iteration: int):
    # ... existing checkpoint logic ...

    # Save meta-prompt state
    if self.prompt_sampler.meta_prompt_evolver:
        meta_prompt_state = self.prompt_sampler.meta_prompt_evolver.save_state()
        with open(checkpoint_path / "meta_prompt_state.json", "w") as f:
            json.dump(meta_prompt_state, f, indent=2)

def _load_checkpoint(self, checkpoint_path: Path):
    # ... existing checkpoint loading ...

    # Load meta-prompt state
    meta_prompt_path = checkpoint_path / "meta_prompt_state.json"
    if meta_prompt_path.exists() and self.prompt_sampler.meta_prompt_evolver:
        with open(meta_prompt_path) as f:
            meta_prompt_state = json.load(f)
        self.prompt_sampler.meta_prompt_evolver.load_state(meta_prompt_state)
```

---

## Configuration Example

```yaml
# config.yaml
prompt:
  # Existing prompt config
  num_top_programs: 3
  num_diverse_programs: 2
  include_artifacts: true

  # Meta-prompting configuration
  meta_prompting:
    enabled: true
    selection_algorithm: "thompson_sampling"  # or "ucb", "epsilon_greedy"

    # Thompson Sampling priors (Beta distribution)
    thompson_prior_alpha: 1.0
    thompson_prior_beta: 1.0

    # Reward calculation
    reward_type: "improvement"  # "improvement", "normalized", "rank"
    improvement_threshold: 0.001

    # Prompt blending
    meta_prompt_weight: 0.1
    meta_prompt_position: "section"  # "prefix", "suffix", "section"

    # Granularity
    per_island_strategies: true
    context_aware: true

    # Exploration schedule
    exploration_decay: 0.995
    min_exploration: 0.05
    warmup_iterations: 50
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_meta_prompting.py

class TestStrategyStats:
    def test_update_statistics(self):
        stats = StrategyStats(name="test")
        stats.update(reward=0.5)
        stats.update(reward=0.3)

        assert stats.total_uses == 2
        assert stats.mean_reward == 0.4
        assert stats.successes > 1.0  # Prior + rewards

    def test_thompson_sampling(self):
        stats = StrategyStats(name="test", successes=10.0, failures=2.0)
        samples = [stats.thompson_sample() for _ in range(1000)]

        # Should be biased toward high values
        assert np.mean(samples) > 0.7

    def test_serialization(self):
        original = StrategyStats(name="test")
        original.update(0.5)
        original.update(0.3)

        serialized = original.to_dict()
        restored = StrategyStats.from_dict(serialized)

        assert restored.total_uses == original.total_uses
        assert restored.mean_reward == original.mean_reward


class TestMetaPromptEvolver:
    def test_warmup_random_selection(self):
        config = MetaPromptConfig(enabled=True, warmup_iterations=10)
        evolver = MetaPromptEvolver(config)

        # During warmup, should select randomly
        selections = [evolver.select_strategy()[0] for _ in range(10)]
        assert len(set(selections)) > 1  # Multiple strategies selected

    def test_exploitation_after_learning(self):
        config = MetaPromptConfig(
            enabled=True,
            warmup_iterations=0,
            selection_algorithm="epsilon_greedy",
            epsilon=0.0,  # Pure exploitation
        )
        evolver = MetaPromptEvolver(config)

        # Train one strategy to be clearly better
        for _ in range(100):
            evolver.update_reward("algorithmic_restructure", 0.9)
            evolver.update_reward("incremental_refinement", 0.1)

        # Should consistently select the better strategy
        selections = [evolver.select_strategy()[0] for _ in range(10)]
        assert all(s == "algorithmic_restructure" for s in selections)

    def test_island_specific_learning(self):
        config = MetaPromptConfig(enabled=True, per_island_strategies=True)
        evolver = MetaPromptEvolver(config)

        # Different strategies work better on different islands
        for _ in range(50):
            evolver.update_reward("vectorization", 0.8, island_idx=0)
            evolver.update_reward("simplification", 0.2, island_idx=0)
            evolver.update_reward("vectorization", 0.2, island_idx=1)
            evolver.update_reward("simplification", 0.8, island_idx=1)

        # Selections should differ by island
        island_0_selections = [evolver.select_strategy(island_idx=0)[0] for _ in range(20)]
        island_1_selections = [evolver.select_strategy(island_idx=1)[0] for _ in range(20)]

        assert island_0_selections.count("vectorization") > island_0_selections.count("simplification")
        assert island_1_selections.count("simplification") > island_1_selections.count("vectorization")

    def test_checkpoint_persistence(self):
        config = MetaPromptConfig(enabled=True)
        evolver = MetaPromptEvolver(config)

        # Train
        for _ in range(50):
            evolver.update_reward("test_strategy", 0.7)

        # Save and restore
        state = evolver.save_state()

        new_evolver = MetaPromptEvolver(config)
        new_evolver.load_state(state)

        assert new_evolver.total_iterations == evolver.total_iterations
        assert new_evolver.global_stats["test_strategy"].mean_reward == \
               evolver.global_stats["test_strategy"].mean_reward
```

### Integration Tests

```python
# tests/integration/test_meta_prompting_integration.py

async def test_full_evolution_with_meta_prompting():
    """Test that meta-prompting integrates with full evolution pipeline"""
    config = Config()
    config.prompt.meta_prompting.enabled = True
    config.prompt.meta_prompting.warmup_iterations = 5

    controller = Controller(config)

    # Run short evolution
    await controller.run(iterations=20)

    # Verify meta-prompt state exists
    assert controller.prompt_sampler.meta_prompt_evolver is not None
    assert controller.prompt_sampler.meta_prompt_evolver.total_iterations > 0

    # Verify strategies were tracked
    summary = controller.prompt_sampler.meta_prompt_evolver.get_strategy_summary()
    assert len(summary) > 0
    assert any(s["uses"] > 0 for s in summary.values())


async def test_checkpoint_resume_preserves_meta_state():
    """Test that checkpoint preserves and restores meta-prompt learning"""
    config = Config()
    config.prompt.meta_prompting.enabled = True

    # First run
    controller1 = Controller(config)
    await controller1.run(iterations=50)

    original_stats = controller1.prompt_sampler.meta_prompt_evolver.get_strategy_summary()
    checkpoint_path = controller1.last_checkpoint_path

    # Resume from checkpoint
    controller2 = Controller(config)
    controller2.load_checkpoint(checkpoint_path)

    restored_stats = controller2.prompt_sampler.meta_prompt_evolver.get_strategy_summary()

    # Stats should match
    for name in original_stats:
        assert restored_stats[name]["uses"] == original_stats[name]["uses"]
        assert abs(restored_stats[name]["mean_reward"] - original_stats[name]["mean_reward"]) < 0.001
```

---

## Observability and Debugging

### Strategy Performance Logging

```python
# In controller.py, add periodic logging

def _log_meta_prompt_stats(self, iteration: int):
    if not self.prompt_sampler.meta_prompt_evolver:
        return

    summary = self.prompt_sampler.meta_prompt_evolver.get_strategy_summary()

    logger.info(f"Meta-prompt strategy stats at iteration {iteration}:")
    for name, stats in sorted(summary.items(), key=lambda x: -x[1]["mean_reward"]):
        logger.info(
            f"  {name}: uses={stats['uses']}, "
            f"mean_reward={stats['mean_reward']:.4f}, "
            f"success_rate={stats['success_rate']:.2%}"
        )
```

### Visualization Support

Add to `scripts/visualizer.py`:

```python
def plot_meta_prompt_performance(checkpoint_path: Path):
    """Visualize meta-prompt strategy performance over time"""
    meta_state_path = checkpoint_path / "meta_prompt_state.json"
    if not meta_state_path.exists():
        return

    with open(meta_state_path) as f:
        state = json.load(f)

    # Plot strategy performance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Strategy usage distribution
    # 2. Mean reward by strategy
    # 3. Success rate over time
    # 4. Island-specific performance heatmap
```

---

## Migration Path

### Phase 1: Core Implementation
1. Add `MetaPromptConfig` to `config.py`
2. Create `openevolve/meta_prompting.py` with `StrategyStats` and `MetaPromptEvolver`
3. Add meta-prompt fragments to `meta_fragments.json`

### Phase 2: Integration
4. Integrate `MetaPromptEvolver` into `PromptSampler`
5. Add reward reporting in `iteration.py`
6. Add checkpoint save/restore in `controller.py`

### Phase 3: Testing & Documentation
7. Add unit tests
8. Add integration tests
9. Update `default_config.yaml` with examples
10. Update README with meta-prompting documentation

---

## Future Enhancements

1. **Neural Strategy Selection**: Replace bandit algorithms with a learned policy network
2. **Cross-Problem Transfer**: Share strategy effectiveness across different optimization problems
3. **Strategy Generation**: Use LLM to generate new strategies based on what's working
4. **Hierarchical Strategies**: Compose multiple strategies for complex mutations
5. **A/B Testing Mode**: Explicitly compare strategies with statistical significance testing
