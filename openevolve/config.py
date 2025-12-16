"""
Configuration handling for OpenEvolve
"""

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: str | None = None
    name: str = None

    # Custom LLM client
    init_client: Callable | None = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: str | None = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: int | None = None

    # Reasoning parameters
    reasoning_effort: str | None = None


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: str | None = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: list[LLMModelConfig] = field(default_factory=list)

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: list[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: str | None = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # Only validate if this looks like a user config (has some model info)
        # Don't validate during internal/default initialization
        if (
            self.primary_model
            or self.secondary_model
            or self.primary_model_weight
            or self.secondary_model_weight
        ) and not self.models:
            raise ValueError(
                "No LLM models configured. Please specify 'models' array or "
                "'primary_model' in your configuration."
            )

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

    def rebuild_models(self) -> None:
        """Rebuild the models list after primary_model/secondary_model field changes"""
        # Clear existing models lists
        self.models = []
        self.evaluator_models = []

        # Re-run model generation logic from __post_init__
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)


@dataclass
class MetaPromptConfig:
    """Configuration for meta-prompting system - evolving prompt strategies

    Meta-prompting enables OpenEvolve to learn which prompt strategies lead to
    fitness improvements and adapt selection probabilities using multi-armed
    bandit algorithms.
    """

    # Feature flag
    enabled: bool = False

    # Strategy selection algorithm
    # - "thompson_sampling": Bayesian approach, good exploration-exploitation balance
    # - "ucb": Upper Confidence Bound, deterministic exploration
    # - "epsilon_greedy": Simple random exploration with probability epsilon
    selection_algorithm: str = "thompson_sampling"

    # Algorithm-specific parameters
    ucb_exploration_constant: float = 2.0  # C in UCB formula: mean + C * sqrt(log(n)/n_i)
    epsilon: float = 0.1  # For epsilon-greedy: probability of random exploration
    thompson_prior_alpha: float = 1.0  # Beta distribution prior (successes)
    thompson_prior_beta: float = 1.0  # Beta distribution prior (failures)

    # Reward calculation
    # - "improvement": Raw fitness delta (child - parent)
    # - "normalized": Delta normalized by parent fitness
    # - "rank": Binary 1.0 if improved, 0.0 otherwise
    reward_type: str = "improvement"
    improvement_threshold: float = 0.0  # Minimum delta for positive reward
    reward_decay: float = 0.99  # Exponential decay for old observations

    # Blending with base prompt
    meta_prompt_weight: float = 0.1  # Weight of meta-prompt in final prompt
    # - "prefix": Add before user prompt
    # - "suffix": Add after user prompt
    # - "section": Insert as dedicated section (recommended)
    meta_prompt_position: str = "section"

    # Strategy granularity
    per_island_strategies: bool = True  # Track strategies per island
    context_aware: bool = True  # Adapt based on fitness range, generation, etc.

    # Exploration vs exploitation over time
    exploration_decay: float = 0.995  # Reduce exploration over iterations
    min_exploration: float = 0.05  # Minimum exploration rate

    # History and persistence
    max_history_per_strategy: int = 1000  # Cap history to prevent memory bloat
    warmup_iterations: int = 50  # Random selection during warmup

    # Custom strategies file (optional)
    strategies_file: str | None = None  # Path to custom meta_fragments.json


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: str | None = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: dict[str, list[str]] = field(default_factory=dict)

    # Meta-prompting configuration
    meta_prompting: MetaPromptConfig = field(default_factory=MetaPromptConfig)

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: int | None = (
        500  # Suggest simplifying if program exceeds this many characters
    )
    include_changes_under_chars: int | None = (
        100  # Include change descriptions in features if under this length
    )
    concise_implementation_max_lines: int | None = (
        10  # Label as "concise" if program has this many lines or fewer
    )
    comprehensive_implementation_min_lines: int | None = (
        50  # Label as "comprehensive" if program has this many lines or more
    )

    # Backward compatibility - deprecated
    code_length_threshold: int | None = None  # Deprecated: use suggest_simplification_after_chars


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: str | None = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5
    programs_per_island: int | None = None

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    # Note: diversity_metric fixed to "edit_distance"
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    # CRITICAL: For custom dimensions, evaluators must return RAW VALUES, not bin indices
    # Built-in: "complexity", "diversity", "score" (always available)
    # Custom: Any metric from your evaluator (must be continuous values)
    feature_dimensions: list[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
        metadata={
            "help": "List of feature dimensions for MAP-Elites grid. "
            "Built-in dimensions: 'complexity', 'diversity', 'score'. "
            "Custom dimensions: Must match metric names from evaluator. "
            "IMPORTANT: Evaluators must return raw continuous values for custom dimensions, "
            "NOT pre-computed bin indices. OpenEvolve handles all scaling and binning internally."
        },
    )
    feature_bins: int | dict[str, int] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: int | None = 42

    # Artifact storage
    artifacts_base_path: str | None = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30

    embedding_model: str | None = None
    similarity_threshold: float = 0.99

    # Vector Database
    vector_store_type: str = "memory"  # Options: "memory", "milvus"
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "openevolve_programs"


def _clear_legacy_llm_fields(llm_dict: dict[str, Any]) -> dict[str, Any]:
    """Prevent LLMConfig __post_init__ from duplicating models during roundtrips."""
    llm_dict = dict(llm_dict)
    llm_dict["primary_model"] = None
    llm_dict["primary_model_weight"] = None
    llm_dict["secondary_model"] = None
    llm_dict["secondary_model_weight"] = None
    return llm_dict


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    # Note: resource limits not implemented
    memory_limit_mb: int | None = None
    cpu_limit: float | None = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Compute-budgeted cascade: cap max stage based on parent fitness.
    budgeted_cascade_enabled: bool = False
    budget_stage3_parent_threshold: float = 0.6
    budget_max_stage_low: int = 2
    budget_max_stage_high: int = 3

    # Parallel evaluation
    parallel_evaluations: int = 1
    # Note: distributed evaluation not implemented
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program

    # Dependency management
    auto_install_dependencies: bool = False  # Automatically install missing dependencies


@dataclass
class EvolutionTraceConfig:
    """Configuration for evolution trace logging"""

    enabled: bool = False
    format: str = "jsonl"  # Options: "jsonl", "json", "hdf5"
    include_code: bool = False
    include_prompts: bool = True
    output_path: str | None = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class RLRewardConfig:
    """Configuration for RL reward calculation"""

    # Reward component weights (should sum to ~1.0)
    fitness_weight: float = 0.6  # Weight for fitness improvement
    diversity_weight: float = 0.2  # Weight for diversity maintenance
    novelty_weight: float = 0.1  # Weight for behavioral novelty
    efficiency_weight: float = 0.1  # Weight for efficiency (fewer iterations to improve)

    # Reward shaping
    improvement_threshold: float = 0.0  # Minimum improvement for positive reward
    plateau_penalty: float = 0.1  # Penalty per N iterations without improvement
    plateau_window: int = 50  # Iterations to consider for plateau detection


@dataclass
class RLNeuralConfig:
    """Configuration for neural network-based policy (optional)"""

    hidden_dim: int = 64  # Hidden layer dimension
    num_layers: int = 2  # Number of hidden layers
    batch_size: int = 32  # Batch size for training
    replay_buffer_size: int = 10000  # Experience replay buffer size
    learning_rate: float = 1e-3  # Learning rate for neural network
    target_update_freq: int = 100  # Frequency to update target network


@dataclass
class RLConfig:
    """Configuration for RL-based adaptive selection

    The RL system learns optimal selection policies during evolution,
    adapting exploration/exploitation trade-offs based on the current
    state of evolution. It complements meta-prompting: RL handles
    "what to select", meta-prompting handles "how to prompt".
    """

    # Feature flag
    enabled: bool = False

    # Algorithm selection
    # - "contextual_thompson": Bayesian contextual bandit with Thompson Sampling (recommended)
    # - "contextual_ucb": Contextual UCB algorithm
    # - "neural_bandit": Neural network-based contextual bandit (requires more data)
    # - "epsilon_greedy": Simple epsilon-greedy (baseline)
    algorithm: str = "contextual_thompson"

    # State features to observe
    # Available features:
    # - "normalized_iteration": Current iteration / max iterations
    # - "best_fitness": Best fitness so far (normalized)
    # - "mean_fitness": Population mean fitness
    # - "fitness_std": Population fitness standard deviation
    # - "fitness_improvement_rate": Recent improvement rate
    # - "population_diversity": Edit distance diversity
    # - "archive_coverage": MAP-Elites grid coverage fraction
    # - "generations_without_improvement": Stagnation indicator
    # - "recent_exploration_success": Success rate of exploration actions
    # - "recent_exploitation_success": Success rate of exploitation actions
    # - "island_variance": Variance between island fitnesses
    state_features: list[str] = field(
        default_factory=lambda: [
            "normalized_iteration",
            "best_fitness",
            "fitness_improvement_rate",
            "population_diversity",
            "archive_coverage",
            "generations_without_improvement",
            "recent_exploration_success",
            "recent_exploitation_success",
        ]
    )

    # Reward configuration
    reward: RLRewardConfig = field(default_factory=RLRewardConfig)

    # Learning parameters
    learning_rate: float = 0.01  # For Bayesian updates
    discount_factor: float = 0.99  # For temporal credit assignment
    exploration_bonus: float = 2.0  # UCB exploration constant

    # Warmup and exploration
    warmup_iterations: int = 100  # Random policy during warmup
    exploration_decay: float = 0.995  # Decay rate for exploration
    min_exploration: float = 0.05  # Minimum exploration probability

    # Action tracking for credit assignment
    action_history_size: int = 100  # How many recent actions to track per type
    success_window: int = 20  # Window for computing success rates

    # Offline training
    pretrain_from_traces: bool = False  # Pre-train from historical traces
    trace_path: str | None = None  # Path to evolution traces for pre-training

    # Extended action control (optional)
    control_temperature: bool = False  # Also learn temperature adjustment
    control_diff_mode: bool = False  # Also learn diff vs full rewrite decision
    temperature_range: tuple[float, float] = (0.3, 1.0)  # Temperature bounds

    # Neural bandit specific (only used if algorithm="neural_bandit")
    neural: RLNeuralConfig = field(default_factory=RLNeuralConfig)

    # Per-island learning
    per_island_policies: bool = True  # Separate policy per island

    # Checkpoint and persistence
    save_detailed_stats: bool = True  # Save detailed action statistics


@dataclass
class SkepticConfig:
    """Configuration for adversarial skeptic in discovery mode"""

    # Attack configuration
    num_attack_rounds: int = 3  # Number of adversarial rounds per hypothesis
    attack_timeout: float = 30.0  # Timeout per attack in seconds

    # Adaptive budget: scale attack rounds with fitness.
    adaptive_attack_rounds: bool = False
    min_attack_rounds: int = 1
    max_attack_rounds: int | None = None  # Defaults to num_attack_rounds

    # Attack type probabilities
    edge_case_prob: float = 0.4  # Edge case inputs (empty, null, huge)
    type_confusion_prob: float = 0.3  # Wrong types, mixed types
    overflow_prob: float = 0.2  # Numerical overflow/underflow
    malformed_prob: float = 0.1  # Malformed/corrupted inputs

    # Execution settings
    use_sandbox: bool = True  # Use sandboxed execution
    max_memory_mb: int = 512  # Memory limit for execution
    max_output_size: int = 10000  # Max output characters to capture

    # Remote execution settings (for code requiring GPU/special deps)
    remote_execution: bool = False  # Execute attacks on remote host via SSH
    remote_host: str | None = None  # SSH host (e.g., "user@host")
    remote_python: str | None = (
        None  # Python interpreter on remote (e.g., "/path/to/venv/bin/python")
    )
    remote_work_dir: str = "/tmp/openevolve_skeptic"  # Working directory on remote

    # Dependency detection - skip execution if imports can't be satisfied
    skip_missing_deps: bool = True  # Skip execution if required imports missing locally
    required_imports_for_remote: list[str] | None = (
        None  # Imports that trigger remote execution (e.g., ["torch"])
    )

    # Static analysis settings
    static_analysis_enabled: bool = True  # Check for dangerous patterns (eval, exec, etc.)

    # Reproduction settings
    enable_blind_reproduction: bool = False  # Whether to use blind reproduction test

    # Optional function entrypoint to target in adversarial attacks.
    # If set, the skeptic will call this function instead of guessing.
    entrypoint: str | None = None

    # Optional task-specific skeptic plugins.
    plugins: list[str] = field(default_factory=list)

    # Test harness defaults
    default_atol: float = 1e-3  # Default absolute tolerance for numeric comparisons
    default_rtol: float = 0.0  # Default relative tolerance for numeric comparisons


@dataclass
class HeisenbergConfig:
    """Configuration for Ontological Expansion (Heisenberg Engine)

    The Heisenberg Engine detects when optimization is fundamentally stuck
    due to missing variables in the model (not just bad solutions) and
    automatically expands the state space by discovering hidden variables.

    Key insight: There's a difference between:
    - "We haven't optimized well enough" (keep trying)
    - "Our model cannot represent the solution" (need new variables)
    """

    # Enable/disable
    enabled: bool = False

    # Crisis detection
    min_plateau_iterations: int = 50  # Min iterations before declaring plateau
    fitness_improvement_threshold: float = 0.001  # Below this = no improvement
    variance_window: int = 20  # Window size for variance calculation
    crisis_confidence_threshold: float = 0.7  # Min confidence to trigger crisis
    cooldown_iterations: int = 30  # Wait after crisis before detecting another

    # Probe synthesis
    max_probes_per_crisis: int = 5  # Max probes to generate per crisis
    probe_timeout: float = 60.0  # Timeout for probe execution (seconds)

    # Statistical validation
    validation_trials: int = 5  # Number of trials for validation
    min_correlation_threshold: float = 0.6  # Min correlation to accept variable

    # Ontology limits
    max_ontology_generations: int = 10  # Max times to expand ontology
    max_variables_per_ontology: int = 50  # Max variables in single ontology

    # Soft reset behavior
    programs_to_keep_on_reset: int = 10  # Top N programs to keep after expansion

    # Auto-instrumentation
    auto_instrument: bool = True  # Auto-instrument code to capture traces
    instrumentation_level: str = "standard"  # "minimal", "standard", "comprehensive"

    # Collaborative Discovery - Multi-Agent Novel Physics Generation
    collaborative_discovery_enabled: bool = False  # Use agent debate instead of probes
    max_debate_rounds: int = 5  # Number of debate rounds between agents
    min_consensus_for_synthesis: float = 0.6  # Agreement needed to synthesize
    elimination_threshold: float = 0.3  # Ideas below this confidence are dropped

    # Domain context for collaborative agents (auto-extracted from problem description if not set)
    domain_context: str | None = None


@dataclass
class GoldenPathConfig:
    """Configuration for the Golden Path - Autonomous Ontological Discovery

    The Golden Path framework enables TRUE ontological discovery - finding hidden
    variables and patterns that don't exist in the current representation.

    Unlike parameter optimization, the Golden Path discovers NEW dimensions
    of the problem space through:
    - Prescience: Crisis detection (sees when evolution hits true walls)
    - Mentat: Program mining (extracts patterns from successful programs)
    - SietchFinder: Hidden variable discovery (proposes new dimensions)
    - GomJabbar: Validation (tests if discoveries are real)
    - SpiceAgony: Integration (adds validated variables to the system)

    Named after Dune's Golden Path - the prescient vision that requires
    seeing beyond what is currently visible.
    """

    # Enable/disable
    enabled: bool = False

    # Prescience (Crisis Detection) settings
    prescience_short_window: int = 10  # Recent history for gradient
    prescience_medium_window: int = 30  # Medium-term trends
    prescience_long_window: int = 100  # Long-term pattern analysis
    gradient_threshold: float = 0.001  # Below this = stagnant
    variance_threshold: float = 0.0001  # Below this = clustered
    diversity_threshold: float = 0.3  # Below this = converged

    # Mentat (Pattern Mining) settings
    min_programs_for_analysis: int = 20  # Need this many programs to mine
    top_n_patterns: int = 10  # Keep top N patterns
    min_correlation_threshold: float = 0.3  # Minimum correlation to consider
    min_discriminative_power: float = 0.2  # Minimum effect size

    # SietchFinder (Hidden Variable Discovery) settings
    max_hypotheses_per_round: int = 5  # Max hidden variables to propose
    use_pattern_mining: bool = True  # Derive variables from patterns
    use_llm_hypothesis: bool = True  # Use LLM to propose variables
    use_domain_templates: bool = True  # Apply domain-specific templates

    # GomJabbar (Validation) settings
    validation_min_correlation: float = 0.15  # Minimum correlation to validate
    validation_max_p_value: float = 0.05  # Maximum p-value for significance
    validation_min_incremental_r2: float = 0.02  # Minimum improvement
    validation_cv_folds: int = 5  # Cross-validation folds
    validation_bootstrap_iterations: int = 100  # Bootstrap iterations for CI

    # SpiceAgony (Integration) settings
    auto_integrate: bool = True  # Automatically integrate validated variables
    default_variable_weight: float = 0.1  # Weight in final score for new vars
    backup_before_modify: bool = True  # Backup evaluator before modification

    # Orchestration settings
    min_programs_for_discovery: int = 30  # Need this many before activating
    max_discovery_rounds: int = 5  # Maximum discovery rounds
    cooldown_after_discovery: int = 20  # Iterations to wait after discovery

    # Output settings
    save_discoveries_to_file: bool = True  # Export discovered variables
    discoveries_output_path: str = "golden_path_discoveries.py"


@dataclass
class EpistemicArchiveConfig:
    """Configuration for Epistemic Archive (Behavioral Diversity)"""

    # Frontier settings for curiosity sampling
    frontier_threshold_high: float = 0.7  # High bin index threshold
    frontier_threshold_low: float = 0.3  # Low bin index threshold
    frontier_bonus: float = 0.5  # Curiosity score bonus for frontier programs

    # Fitness prediction settings
    prediction_prior_mean: float = 0.5  # Prior mean for fitness prediction
    prediction_prior_weight: float = 2.0  # Weight of prior in Bayesian update


@dataclass
class DiscoveryConfig:
    """Configuration for Discovery Mode - Open-Ended Scientific Discovery"""

    # Enable discovery mode
    enabled: bool = False

    # Problem description (required if enabled)
    problem_description: str = ""

    # Problem evolution settings
    problem_evolution_enabled: bool = True
    evolve_problem_after_solutions: int = 5  # Evolve problem after N successful solutions

    # Paired open-ended co-evolution (POET-style)
    coevolution_enabled: bool = False  # Maintain multiple active problems
    max_active_problems: int = 5  # Max concurrent problems when coevolution enabled
    novelty_threshold: float = 0.15  # Gate new problems by novelty
    min_problem_difficulty: float = 0.5
    max_problem_difficulty: float = 10.0
    min_islands_per_problem: int = 1  # Keep at least N islands per active problem

    # Minimal-criterion transfer screening for new problems (coevolution only).
    # A candidate is admitted only if existing solvers achieve:
    #   min_transfer_fitness <= best_transfer_fitness < max_transfer_fitness.
    transfer_trial_programs: int = 3  # How many top solvers to test on candidate
    min_transfer_fitness: float = 0.3  # Too hard if best transfer below this
    max_transfer_fitness: float | None = (
        None  # Too easy if best transfer >= this (defaults to solution_threshold)
    )
    transfer_max_stage: int = 2  # Max cascade stage to use for screening (1/2/3)

    # Adversarial skeptic settings
    skeptic_enabled: bool = True
    skeptic: SkepticConfig = field(default_factory=SkepticConfig)

    # Epistemic archive settings
    surprise_tracking_enabled: bool = True
    curiosity_sampling_enabled: bool = True
    phenotype_dimensions: list[str] = field(default_factory=lambda: ["complexity", "efficiency"])
    phenotype_bins: int = 10  # Number of bins for phenotype grid discretization
    # Optional phenotype dimensions to mirror into program metrics for MAP-Elites.
    phenotype_feature_dimensions: list[str] = field(default_factory=list)
    epistemic_archive: EpistemicArchiveConfig = field(default_factory=EpistemicArchiveConfig)

    # Thresholds
    solution_threshold: float = 0.8  # Fitness threshold to consider problem "solved"
    surprise_bonus_threshold: float = 0.2  # Surprise level to trigger bonus exploration

    # Logging
    log_discoveries: bool = True
    discovery_log_path: str | None = None  # Defaults to output_dir/discovery_log.jsonl

    # Heisenberg Engine (Ontological Expansion)
    heisenberg: HeisenbergConfig = field(default_factory=HeisenbergConfig)

    # Golden Path (Autonomous Ontological Discovery)
    golden_path: GoldenPathConfig = field(default_factory=GoldenPathConfig)


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: str | None = None
    random_seed: int | None = 42
    language: str = None
    file_suffix: str = ".py"

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000

    # Early stopping settings
    early_stopping_patience: int | None = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    # Parallel controller settings
    max_tasks_per_child: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # List of nested config keys to skip in top-level processing
        nested_keys = [
            "llm",
            "prompt",
            "database",
            "evaluator",
            "evolution_trace",
            "rl",
            "discovery",
        ]

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in nested_keys and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            prompt_dict = config_dict["prompt"].copy()
            # Handle nested meta_prompting config
            if "meta_prompting" in prompt_dict:
                prompt_dict["meta_prompting"] = MetaPromptConfig(**prompt_dict["meta_prompting"])
            config.prompt = PromptConfig(**prompt_dict)
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])
        if "evolution_trace" in config_dict:
            config.evolution_trace = EvolutionTraceConfig(**config_dict["evolution_trace"])

        # Handle RL config
        if "rl" in config_dict:
            rl_dict = config_dict["rl"].copy()
            # Handle nested reward config
            if "reward" in rl_dict:
                rl_dict["reward"] = RLRewardConfig(**rl_dict["reward"])
            # Handle nested neural config
            if "neural" in rl_dict:
                rl_dict["neural"] = RLNeuralConfig(**rl_dict["neural"])
            config.rl = RLConfig(**rl_dict)

        # Handle discovery config
        if "discovery" in config_dict:
            discovery_dict = config_dict["discovery"].copy()
            # Handle nested skeptic config
            if "skeptic" in discovery_dict:
                discovery_dict["skeptic"] = SkepticConfig(**discovery_dict["skeptic"])
            # Handle nested heisenberg config
            if "heisenberg" in discovery_dict:
                discovery_dict["heisenberg"] = HeisenbergConfig(**discovery_dict["heisenberg"])
            # Handle nested golden_path config
            if "golden_path" in discovery_dict:
                discovery_dict["golden_path"] = GoldenPathConfig(**discovery_dict["golden_path"])
            # Handle nested epistemic_archive config
            if "epistemic_archive" in discovery_dict:
                discovery_dict["epistemic_archive"] = EpistemicArchiveConfig(
                    **discovery_dict["epistemic_archive"]
                )
            config.discovery = DiscoveryConfig(**discovery_dict)

        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_worker_dict(self) -> dict[str, Any]:
        """Serialize config for multiprocessing worker initialization."""
        data = asdict(self)
        llm_dict = data.get("llm") or {}
        data["llm"] = _clear_legacy_llm_fields(llm_dict)
        return data

    @classmethod
    def from_worker_dict(cls, data: dict[str, Any]) -> "Config":
        """Reconstruct a Config from `to_worker_dict` output."""
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
