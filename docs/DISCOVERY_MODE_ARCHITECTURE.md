# Discovery Mode Architecture

## Overview

This document describes the **Discovery Engine** - an extension to OpenEvolve that enables true scientific discovery by evolving both questions AND answers.

## The Problem with Standard Optimization

Standard evolutionary agents (including vanilla OpenEvolve) suffer from two fundamental limitations:

1. **Static Problem Spaces**: They iterate on a fixed code template, meaning they can only "fill in the blanks" rather than ask new questions.

2. **Sycophantic Evaluation**: Using "LLM-as-a-Judge" encourages persuasive writing over factual correctness.

## The Solution: Three Modules

### Module 1: Problem Evolver (Explorer)

**Purpose**: Evolve the questions, not just the answers.

**Location**: `openevolve/discovery/problem_space.py`

**Key Classes**:
- `ProblemSpace`: Represents an evolvable problem with constraints, objectives, and difficulty
- `ProblemEvolver`: Mutates solved problems into harder variants

**Integration Point**: After N solutions are found, automatically evolve the problem:

```python
# In the main loop
if solutions_found >= config.evolve_problem_after_solutions:
    new_problem = await problem_evolver.evolve(
        current_problem,
        solution_characteristics
    )
```

**Mutation Types**:
| Type | Probability | Example |
|------|-------------|---------|
| Add Constraint | 40% | "Sort without using comparisons" |
| Modify Objective | 30% | "Optimize for memory, not speed" |
| Increase Difficulty | 20% | "Handle 10x larger inputs" |
| Expand Scope | 10% | "Support streaming data" |

---

### Module 2: Adversarial Skeptic

**Purpose**: Replace LLM-as-a-Judge with falsification-based testing.

**Location**: `openevolve/discovery/skeptic.py`

**Key Classes**:
- `AdversarialSkeptic`: Generates and executes adversarial attacks
- `FalsificationResult`: Records attack outcomes

**Integration Point**: Before accepting a program into the archive:

```python
# In process_program()
survived, results = await skeptic.falsify(program, description)
if not survived:
    return False, {"falsification_passed": False}
```

**Attack Types**:
| Attack | Description | Example Input |
|--------|-------------|---------------|
| Edge Case | Boundary conditions | `[]`, `[0]`, `None` |
| Type Confusion | Wrong types | `[1, "a", 2.5]` |
| Overflow | Numerical limits | `[float('inf')]`, `[1e300]` |
| Malformed | Corrupted data | `b'\xff\xfe'` |

**Three-Phase Testing**:
1. **Static Analysis**: AST parsing, security checks
2. **Adversarial Execution**: Run with hostile inputs
3. **Blind Reproduction** (optional): Can another LLM reproduce from description alone?

---

### Module 3: Epistemic Archive

**Purpose**: Store solutions by BEHAVIOR, not just fitness.

**Location**: `openevolve/discovery/epistemic_archive.py`

**Key Classes**:
- `Phenotype`: Behavioral characteristics (complexity, efficiency, approach)
- `PhenotypeExtractor`: Extracts phenotypes from code
- `SurpriseMetric`: Tracks prediction vs. actual fitness
- `EpistemicArchive`: Enhanced MAP-Elites with surprise tracking

**Integration Point**: Works alongside existing `ProgramDatabase`:

```python
# Enhanced add operation
was_novel, surprise = archive.add_with_phenotype(
    program,
    predicted_fitness=predicted
)

if surprise.surprise_score > 0.2:
    # This is interesting - explore more in this region!
    pass
```

**Phenotype Dimensions**:
| Dimension | Description | Extraction Method |
|-----------|-------------|-------------------|
| complexity | Structural complexity | AST node count |
| efficiency | Runtime performance | Benchmark timing |
| robustness | Error handling | Try/except count |
| approach_signature | Algorithm type | Structural hash |

**Surprise-Based Curiosity**:
```
Surprise = |Predicted Fitness - Actual Fitness|

High surprise → Unexpected result → Scientific interest!
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Discovery Engine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Problem    │  │  Adversarial │  │  Epistemic   │          │
│  │   Evolver    │  │   Skeptic    │  │   Archive    │          │
│  │  (Explorer)  │  │  (Skeptic)   │  │  (Librarian) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   OpenEvolve Core                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │  │
│  │  │ Controller │  │  Database  │  │  Evaluator │          │  │
│  │  │            │  │ (MAP-Elites)│  │  (Cascade) │          │  │
│  │  └────────────┘  └────────────┘  └────────────┘          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │  │
│  │  │    LLM     │  │   Prompt   │  │  Iteration │          │  │
│  │  │  Ensemble  │  │  Sampler   │  │   Worker   │          │  │
│  │  └────────────┘  └────────────┘  └────────────┘          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
1. Genesis Problem
   │
   ▼
2. Sample Parent from Archive ──────────────────┐
   │                                            │
   ▼                                            │
3. Generate Child (LLM)                         │
   │                                            │
   ▼                                            │
4. Evaluate (Cascade Stages)                    │
   │                                            │
   ▼                                            │
5. Adversarial Falsification                    │
   │  ├─ FAIL → Rejected (not added)           │
   │  └─ PASS → Continue                       │
   ▼                                            │
6. Phenotype Extraction                         │
   │                                            │
   ▼                                            │
7. Surprise Calculation                         │
   │  └─ High surprise → Log discovery event   │
   ▼                                            │
8. Add to Archive ──────────────────────────────┘
   │
   ▼
9. Check Solution Threshold
   │  └─ If >= threshold → Mark problem solved
   ▼
10. Check Evolution Trigger
    │  └─ If N solutions → Evolve Problem → Go to step 2
    ▼
11. Continue iteration → Go to step 2
```

---

## Configuration

### Discovery Config
```yaml
discovery:
  # Problem evolution
  problem_evolution_enabled: true
  evolve_problem_after_solutions: 5

  # POET-style paired co-evolution (multiple active problems)
  # NOTE: requires your evaluator to accept `problem_context` (or **kwargs) and ideally
  # to actually use it (e.g., adjust scoring/constraints). Otherwise coevolution is disabled.
  coevolution_enabled: true
  max_active_problems: 3
  novelty_threshold: 0.12
  min_problem_difficulty: 0.8
  max_problem_difficulty: 5.0
  min_islands_per_problem: 1

  # Minimal-criterion transfer screening for candidate problems
  # Screen by briefly re-evaluating top solvers on the candidate problem context.
  # Admit only if:
  #   min_transfer_fitness <= best_transfer_fitness < max_transfer_fitness
  transfer_trial_programs: 3
  min_transfer_fitness: 0.3
  max_transfer_fitness: null  # defaults to solution_threshold
  transfer_max_stage: 2       # cap cascade stage for screening

  # Adversarial skeptic
	  skeptic_enabled: true
	  skeptic:
	    num_attack_rounds: 3
	    attack_timeout: 30.0
	    enable_blind_reproduction: false
	    # Optional adaptive budget + plugins
	    adaptive_attack_rounds: false
	    min_attack_rounds: 1
	    max_attack_rounds: null
	    plugins: []

  # Archive enhancements
  surprise_tracking_enabled: true
  curiosity_sampling_enabled: true
	  phenotype_dimensions:
	    - "complexity"
	    - "efficiency"
	  # Optional phenotype dims to mirror into metrics for MAP-Elites
	  phenotype_feature_dimensions: []

  # Thresholds
  solution_threshold: 0.8
  surprise_bonus_threshold: 0.2
```

---

## File Structure

```
openevolve/
├── discovery/
│   ├── __init__.py              # Public API
│   ├── problem_space.py         # ProblemSpace, ProblemEvolver
│   ├── skeptic.py               # AdversarialSkeptic, FalsificationResult
│   ├── epistemic_archive.py     # Phenotype, EpistemicArchive
│   ├── engine.py                # DiscoveryEngine (main integration)
│   │
│   │   # Heisenberg Engine (Ontological Expansion)
│   ├── ontology.py              # Variable, Ontology, OntologyManager
│   ├── crisis_detector.py       # EpistemicCrisis, CrisisDetector
│   ├── instrument_synthesizer.py # Probe, InstrumentSynthesizer
│   └── code_instrumenter.py     # CodeInstrumenter, TraceCollector
│
├── examples/
│   ├── discovery_mode/
│   │   ├── run_discovery.py     # Example runner
│   │   ├── initial_sort.py      # Starting program
│   │   ├── evaluator.py         # Multi-stage evaluator
│   │   ├── config.yaml          # Discovery config
│   │   └── README.md            # Usage guide
│   │
│   └── heisenberg_demo/         # Heisenberg Engine demo
│       ├── initial_program.py   # Starting sort program
│       ├── evaluator.py         # Evaluator with hidden cache variable
│       ├── config.yaml          # Config with Heisenberg enabled
│       └── README.md            # Usage guide
│
├── tests/
│   ├── test_discovery.py        # Discovery mode tests
│   └── test_heisenberg.py       # Heisenberg Engine tests
│
└── docs/
    └── DISCOVERY_MODE_ARCHITECTURE.md  # This document
```

---

## Module 4: Heisenberg Engine (Ontological Expansion)

**Purpose**: Discover NEW variables when optimization is stuck.

**Location**: `openevolve/discovery/ontology.py`, `crisis_detector.py`, `instrument_synthesizer.py`, `code_instrumenter.py`

**Key Insight**: Traditional AI optimizes relationships between KNOWN variables. True scientific discovery requires finding NEW variables. The Heisenberg Engine detects when optimization is fundamentally stuck because the model is missing a variable.

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `Variable` | `ontology.py` | Represents a discovered or known variable |
| `Ontology` | `ontology.py` | Collection of variables with lineage tracking |
| `OntologyManager` | `ontology.py` | Creates, expands, and tracks ontologies |
| `EpistemicCrisis` | `crisis_detector.py` | Represents a detected plateau/crisis |
| `CrisisDetector` | `crisis_detector.py` | Monitors fitness for plateau detection |
| `Probe` | `instrument_synthesizer.py` | LLM-generated code to discover hidden patterns |
| `InstrumentSynthesizer` | `instrument_synthesizer.py` | Creates and executes probes |
| `CodeInstrumenter` | `code_instrumenter.py` | Auto-instruments code for tracing |

### How It Works

```
1. Normal Evolution
   │
   ▼
2. Crisis Detector monitors fitness history
   │  ├─ Plateau detected? (no improvement over N iterations)
   │  ├─ Systematic bias? (consistent error patterns)
   │  └─ Unexplained variance? (similar code, different results)
   │
   ▼ (if crisis detected)
3. InstrumentSynthesizer generates probes
   │  └─ Probes analyze artifacts for hidden patterns
   │
   ▼
4. Execute probes, validate discoveries
   │  └─ Statistical validation (5 trials, check correlation)
   │
   ▼
5. OntologyManager expands the state space
   │  └─ Add new variable to known variables
   │
   ▼
6. Soft Reset
   │  ├─ Keep top N programs
   │  ├─ Update problem description with new variable
   │  └─ Reset crisis detector
   │
   ▼
7. Continue evolution with expanded ontology
   └─ LLM now knows about the new variable!
```

### Crisis Types

| Type | Description | Example |
|------|-------------|---------|
| `plateau` | Fitness stagnates despite diverse attempts | Stuck at 0.7 for 50 iterations |
| `systematic_bias` | Consistent error patterns | Always fails on large inputs |
| `unexplained_variance` | High variability with similar code | Same algorithm, 2x perf difference |

### Probe Types

| Type | Purpose | Discovers |
|------|---------|-----------|
| `state` | Analyze intermediate values | Hidden accumulators, cached values |
| `gradient` | Analyze fitness landscape | Unexplored improvement directions |
| `coverage` | Find unexplored regions | Untested input ranges |
| `numerical` | Detect stability issues | Precision, overflow problems |

### Integration Point

```python
# In process_program() after evaluation
if self.crisis_detector is not None:
    self.crisis_detector.record_evaluation(
        iteration=current_iteration,
        metrics=program.metrics,
        artifacts=program.metadata.get("artifacts", {})
    )

    crisis = self.crisis_detector.detect_crisis()
    if crisis:
        await self._handle_epistemic_crisis(crisis, program)
```

### Configuration

```yaml
discovery:
  heisenberg:
    enabled: true

    # Crisis detection
    min_plateau_iterations: 50
    fitness_improvement_threshold: 0.001
    crisis_confidence_threshold: 0.7

    # Probe synthesis
    max_probes_per_crisis: 5
    probe_timeout: 60.0

    # Validation
    validation_trials: 5
    min_correlation_threshold: 0.6

    # Soft reset
    programs_to_keep_on_reset: 10

    # Instrumentation
    auto_instrument: true
    instrumentation_level: "standard"
```

### Example: Cache Locality Discovery

```
Problem: Optimize sorting algorithm

1. Evolution improves O(n²) → O(n log n)
2. Fitness plateaus at 0.68
3. Crisis detected (type: plateau, confidence: 0.85)
4. Probe analyzes artifacts, finds:
   - cache_hit_rate correlates with performance (r=0.73)
5. Ontology expands: adds "memory_access_pattern"
6. Soft reset, problem updated:
   "Optimize sorting considering memory access patterns"
7. Evolution resumes, reaches 0.91 fitness

New variable discovered: cache locality matters!
```

---

## Implementation Phases

### Phase 1: Core Modules (Complete)
- [x] `ProblemSpace` and `ProblemEvolver`
- [x] `AdversarialSkeptic`
- [x] `EpistemicArchive` with phenotype tracking
- [x] `DiscoveryEngine` integration

### Phase 2: Deep Integration (Complete)
- [x] Modify `Controller.run()` to call DiscoveryEngine
- [x] Add discovery config to main `Config` class
- [x] Integrate problem context into prompt templates
- [x] Add discovery metrics to evolution trace

### Phase 3: Heisenberg Engine (Complete)
- [x] Crisis detection (plateau, bias, variance)
- [x] Probe synthesis with LLM
- [x] Code instrumentation for tracing
- [x] Statistical validation of discoveries
- [x] Ontology expansion with soft reset
- [x] Checkpoint/resume support

### Phase 4: Advanced Features
- [ ] Blind reproduction test with separate LLM
- [ ] Adaptive attack selection based on history
- [ ] Cross-run knowledge transfer
- [ ] Multi-objective problem evolution

### Phase 5: Tooling
- [ ] Visualizer for problem evolution tree
- [ ] Discovery event timeline UI
- [ ] Phenotype space explorer
- [ ] Surprise heatmaps
- [ ] Ontology lineage visualizer

---

## Key Differences from Standard OpenEvolve

| Aspect | Standard | Discovery Mode |
|--------|----------|----------------|
| Problem | Fixed evaluator file | Evolving ProblemSpace |
| Evaluation | Cascade stages + LLM feedback | Cascade + Adversarial Falsification |
| Archive | MAP-Elites by metrics | MAP-Elites by phenotype |
| Sampling | Fitness-weighted | Curiosity-weighted (surprise) |
| Goal | Maximize single metric | Explore problem/solution space |
| Output | Best program | Problem evolution tree + diverse solutions |

---

## Usage Example

```python
from openevolve.controller import OpenEvolve
from openevolve.discovery import DiscoveryEngine, DiscoveryConfig

# Initialize OpenEvolve
openevolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py",
    config=config,
)

# Create Discovery Engine
discovery = DiscoveryEngine(
    config=DiscoveryConfig(
        problem_evolution_enabled=True,
        skeptic_enabled=True,
    ),
    openevolve=openevolve,
)

# Set genesis problem
discovery.set_genesis_problem(
    description="Implement an efficient sorting algorithm",
    evaluator_path="evaluator.py",
)

# Run with discovery
for iteration in range(max_iterations):
    # Standard evolution step
    result = await run_iteration(...)

    # Discovery processing
    is_valid, metadata = await discovery.process_program(result.child_program)

    if not metadata["falsification_passed"]:
        continue  # Program was falsified

    if metadata["is_solution"]:
        print(f"Solution found! Problem may evolve...")

# Get final statistics
stats = discovery.get_statistics()
print(f"Problem evolved {stats['problem_evolutions']} times")
print(f"Final difficulty: {stats['current_problem']['difficulty']}")
```

---

## Research Connections

This architecture draws from several research traditions:

1. **Quality-Diversity (QD)** algorithms like MAP-Elites
   - Lehman & Stanley, "Abandoning Objectives" (2011)
   - Mouret & Clune, "Illuminating search spaces" (2015)

2. **Open-Ended Evolution**
   - Stanley et al., "Open-Endedness: The Last Grand Challenge" (2017)
   - POET: Wang et al., "Paired Open-Ended Trailblazer" (2019)

3. **Adversarial Testing**
   - Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
   - Property-based testing (QuickCheck, Hypothesis)

4. **Curiosity-Driven Learning**
   - Pathak et al., "Curiosity-driven Exploration" (2017)
   - Intrinsic motivation in RL

5. **Philosophy of Science**
   - Karl Popper, "The Logic of Scientific Discovery" (1959)
   - Falsificationism as the basis for scientific validity

---

## Module 5: Golden Path (Autonomous Ontological Discovery)

**Purpose**: Discover hidden variables using external tools for TRUE ontological discovery.

**Location**: `openevolve/discovery/golden_path/`

**Key Insight**: The Heisenberg Engine relies on LLM-generated probes, which are limited to concepts the LLM already knows. The Golden Path uses **external discovery tools** (symbolic regression, causal discovery, etc.) to find patterns that don't have names yet.

### Architecture (Dune-themed)

```
GoldenPath (Orchestrator)
    │
    ├── Prescience - Crisis detection, sees when evolution is truly stuck
    │
    ├── Mentat - Program mining, extracts patterns from successful programs
    │
    ├── DiscoveryToolkit - Orchestrates external discovery tools
    │   │
    │   ├── SymbolicRegressionTool (PySR/gplearn)
    │   │   └── Discovers mathematical formulas from data
    │   │
    │   ├── CausalDiscoveryTool (DoWhy/causal-learn)
    │   │   └── Finds causal relationships, not just correlations
    │   │
    │   ├── CodeAnalysisTool (Python AST)
    │   │   └── Structural patterns in code
    │   │
    │   ├── WebResearchTool (arXiv, Semantic Scholar)
    │   │   └── Searches scientific literature
    │   │
    │   └── WolframTool (Wolfram Alpha API)
    │       └── Mathematical insights and solutions
    │
    ├── SietchFinder - Proposes hidden variables from patterns + tool outputs
    │
    ├── GomJabbar - Validates discoveries with rigorous statistics
    │
    └── SpiceAgony - Integrates validated variables into the system
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `GoldenPath` | `golden_path.py` | Main orchestrator |
| `GoldenPathConfig` | `golden_path.py` | Configuration |
| `Prescience` | `prescience.py` | Crisis detection |
| `CrisisType` | `prescience.py` | Types: ONTOLOGY_GAP, LOCAL_OPTIMUM, etc. |
| `Mentat` | `mentat.py` | Pattern mining from programs |
| `SietchFinder` | `sietch_finder.py` | Hidden variable discovery |
| `HiddenVariable` | `sietch_finder.py` | Represents a discovered variable |
| `GomJabbar` | `gom_jabbar.py` | Statistical validation |
| `ValidationResult` | `gom_jabbar.py` | Validation outcome |
| `SpiceAgony` | `spice_agony.py` | Runtime variable injection |
| `DiscoveryToolkit` | `toolkit.py` | Tool orchestration |
| `DiscoveryTool` | `toolkit.py` | Base class for tools |
| `Discovery` | `toolkit.py` | Tool output |

### Discovery Flow

```
1. Normal Evolution
   │
   ▼
2. Prescience monitors for crises
   │  ├─ LOCAL_OPTIMUM: Population converged, need diversity
   │  ├─ ONTOLOGY_GAP: Model cannot represent solution ← TRIGGERS GOLDEN PATH
   │  ├─ COMPLEXITY_BARRIER: Programs too complex
   │  └─ CATASTROPHIC: Major failure detected
   │
   ▼ (if ONTOLOGY_GAP detected)
3. Mentat mines programs for structural patterns
   │  └─ Extracts: function count, loop depth, numeric ranges, etc.
   │
   ▼
4. DiscoveryToolkit runs external tools
   │  ├─ Symbolic Regression: fitness ~ f(features)
   │  ├─ Causal Discovery: X → Y relationships
   │  └─ Code Analysis: AST patterns
   │
   ▼
5. SietchFinder proposes hidden variables
   │  └─ Combines patterns + tool outputs + LLM hypotheses
   │
   ▼
6. GomJabbar validates each hypothesis
   │  ├─ Correlation test (r > 0.15)
   │  ├─ Statistical significance (p < 0.05)
   │  ├─ Incremental R² (adds explanatory power)
   │  ├─ Cross-validation (robust across folds)
   │  └─ Bootstrap CI (confidence interval excludes zero)
   │
   ▼
7. SpiceAgony integrates validated variables
   │  ├─ Runtime injection (no file changes)
   │  └─ Evaluator modification (optional)
   │
   ▼
8. Evolution continues with expanded ontology
```

### DiscoveryToolkit: External Tools

The toolkit allows integrating external frameworks for true discovery:

| Tool | Backend | Discovers |
|------|---------|-----------|
| `SymbolicRegressionTool` | PySR, gplearn, numpy | Mathematical formulas: `fitness ~ sqrt(x) * log(y)` |
| `CausalDiscoveryTool` | DoWhy, causal-learn, numpy | Causal graphs, intervention effects |
| `CodeAnalysisTool` | Python AST | Structural patterns in code |
| `WebResearchTool` | aiohttp | Papers from arXiv, Semantic Scholar |
| `WolframTool` | aiohttp | Analytical solutions via Wolfram Alpha |

**Adding Custom Tools**:

```python
from openevolve.discovery.golden_path import DiscoveryTool, Discovery, DiscoveryType, ToolContext

class MyCustomTool(DiscoveryTool):
    name = "my_tool"
    description = "My custom discovery tool"
    discovery_types = [DiscoveryType.HYPOTHESIS]
    dependencies = ["my_package"]

    async def discover(self, context: ToolContext) -> List[Discovery]:
        # Analyze context.programs
        # Return discoveries with computation_code for validation
        return [
            Discovery(
                name="my_discovery",
                description="Found interesting pattern",
                discovery_type=DiscoveryType.HYPOTHESIS,
                content={"key": "value"},
                computation_code="def compute_my_discovery(code, metrics): ...",
                confidence=0.7,
            )
        ]

# Register with toolkit
toolkit.register_tool(MyCustomTool())
```

### Validation (GomJabbar)

Every hypothesis undergoes rigorous validation:

| Test | Threshold | Purpose |
|------|-----------|---------|
| Correlation | \|r\| > 0.15 | Does it relate to fitness? |
| P-value | p < 0.05 | Is it statistically significant? |
| Incremental R² | ΔR² > 0.02 | Does it add information beyond existing metrics? |
| Cross-validation | CV > 0.6 | Is it robust across data splits? |
| Bootstrap CI | Excludes 0 | Is the effect reliable? |
| Computation success | > 80% | Can we compute it for most programs? |

### Configuration

```yaml
discovery:
  golden_path:
    enabled: true

    # Prescience (Crisis Detection)
    prescience_short_window: 10
    prescience_medium_window: 30
    prescience_long_window: 100
    gradient_threshold: 0.001
    variance_threshold: 0.0001
    diversity_threshold: 0.3

    # Mentat (Pattern Mining)
    min_programs_for_analysis: 20
    top_n_patterns: 10
    min_correlation_threshold: 0.3
    min_discriminative_power: 0.2

    # SietchFinder (Hidden Variable Discovery)
    max_hypotheses_per_round: 5
    use_pattern_mining: true
    use_llm_hypothesis: true
    use_domain_templates: true

    # GomJabbar (Validation)
    validation_min_correlation: 0.15
    validation_max_p_value: 0.05
    validation_min_incremental_r2: 0.02
    validation_cv_folds: 5
    validation_bootstrap_iterations: 100

    # SpiceAgony (Integration)
    auto_integrate: true
    default_variable_weight: 0.1

    # Orchestration
    min_programs_for_discovery: 30
    max_discovery_rounds: 5
    cooldown_after_discovery: 20

    # Output
    save_discoveries_to_file: true
    discoveries_output_path: "golden_path_discoveries.py"
```

### Example: Discovering Hidden Physics

```
Problem: Optimize magnetic mirror coil configuration

1. Evolution reaches fitness 0.85, plateaus
2. Prescience detects ONTOLOGY_GAP (high confidence)
3. Golden Path activates:

   Mentat: Found 8 patterns
     - n_coils correlates with fitness (r=0.45)
     - code_length correlates with fitness (r=0.38)

   DiscoveryToolkit:
     - Symbolic Regression: fitness ~ 0.3 + 0.1*sqrt(n_coils)
     - Causal Discovery: n_coils → fitness (partial r=0.42)
     - Code Analysis: 12 correlated structural patterns

   SietchFinder: Proposed 7 hidden variables

   GomJabbar: Validated 3/7 variables
     ✓ coil_count (r=0.45, incr_r2=0.15)
     ✓ symmetry_index (r=0.38, incr_r2=0.08)
     ✗ random_metric (r=0.05, failed correlation test)

   SpiceAgony: Integrated 2 new variables

4. Evolution resumes with expanded ontology
5. Fitness improves: 0.85 → 0.93
```

### Key Difference from Heisenberg

| Aspect | Heisenberg Engine | Golden Path |
|--------|-------------------|-------------|
| Discovery method | LLM-generated probes | External tools + LLM |
| Pattern source | Artifacts analysis | Code structure + metrics |
| Validation | 5 trials, correlation | Full statistical battery |
| Variables found | Named concepts LLM knows | Mathematical relationships (nameless) |
| Integration | Ontology expansion | Runtime injection |

The Golden Path finds patterns **that don't have names** - they emerge from data, not from human concepts.

---

## File Structure (Updated)

```
openevolve/
├── discovery/
│   ├── __init__.py              # Public API
│   ├── problem_space.py         # ProblemSpace, ProblemEvolver
│   ├── skeptic.py               # AdversarialSkeptic
│   ├── epistemic_archive.py     # Phenotype, EpistemicArchive
│   ├── engine.py                # DiscoveryEngine (main integration)
│   │
│   │   # Heisenberg Engine (Ontological Expansion)
│   ├── ontology.py              # Variable, Ontology, OntologyManager
│   ├── crisis_detector.py       # EpistemicCrisis, CrisisDetector
│   ├── instrument_synthesizer.py # Probe, InstrumentSynthesizer
│   ├── code_instrumenter.py     # CodeInstrumenter
│   │
│   │   # Golden Path (Autonomous Ontological Discovery)
│   └── golden_path/
│       ├── __init__.py          # Public API
│       ├── golden_path.py       # GoldenPath orchestrator
│       ├── prescience.py        # Crisis detection
│       ├── mentat.py            # Pattern mining
│       ├── sietch_finder.py     # Hidden variable discovery
│       ├── gom_jabbar.py        # Statistical validation
│       ├── spice_agony.py       # Variable integration
│       ├── toolkit.py           # DiscoveryToolkit orchestrator
│       │
│       └── tools/               # External discovery tools
│           ├── __init__.py
│           ├── symbolic_regression.py  # PySR/gplearn
│           ├── causal_discovery.py     # DoWhy/causal-learn
│           ├── code_analysis.py        # AST analysis
│           ├── web_research.py         # arXiv, Semantic Scholar
│           └── wolfram.py              # Wolfram Alpha
│
├── examples/
│   ├── discovery_mode/          # Discovery mode demo
│   ├── heisenberg_demo/         # Heisenberg Engine demo
│   └── magnetic_mirror_frc/     # Golden Path demo (fusion optimization)
│
└── docs/
    └── DISCOVERY_MODE_ARCHITECTURE.md  # This document
```

---

## Implementation Status

### Phase 1: Core Modules (Complete)
- [x] ProblemSpace and ProblemEvolver
- [x] AdversarialSkeptic
- [x] EpistemicArchive with phenotype tracking
- [x] DiscoveryEngine integration

### Phase 2: Deep Integration (Complete)
- [x] Modify Controller.run() to call DiscoveryEngine
- [x] Add discovery config to main Config class
- [x] Integrate problem context into prompt templates
- [x] Add discovery metrics to evolution trace

### Phase 3: Heisenberg Engine (Complete)
- [x] Crisis detection (plateau, bias, variance)
- [x] Probe synthesis with LLM
- [x] Code instrumentation for tracing
- [x] Statistical validation of discoveries
- [x] Ontology expansion with soft reset
- [x] Checkpoint/resume support

### Phase 4: Golden Path (Complete)
- [x] Prescience crisis detection
- [x] Mentat pattern mining
- [x] SietchFinder hidden variable discovery
- [x] GomJabbar statistical validation
- [x] SpiceAgony runtime integration
- [x] DiscoveryToolkit orchestration
- [x] Symbolic regression tool (PySR/gplearn/numpy)
- [x] Causal discovery tool (DoWhy/causal-learn/numpy)
- [x] Code analysis tool (Python AST)
- [x] Web research tool (arXiv, Semantic Scholar)
- [x] Wolfram tool (Wolfram Alpha API)

### Phase 5: Advanced Features (Planned)
- [ ] Blind reproduction test with separate LLM
- [ ] Adaptive attack selection based on history
- [ ] Cross-run knowledge transfer
- [ ] Multi-objective problem evolution
- [ ] Custom tool plugin system

### Phase 6: Tooling (Planned)
- [ ] Visualizer for problem evolution tree
- [ ] Discovery event timeline UI
- [ ] Phenotype space explorer
- [ ] Surprise heatmaps
- [ ] Ontology lineage visualizer
- [ ] Golden Path discovery dashboard
