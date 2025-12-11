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

  # Adversarial skeptic
  skeptic_enabled: true
  skeptic:
    num_attack_rounds: 3
    attack_timeout: 30.0
    enable_blind_reproduction: false

  # Archive enhancements
  surprise_tracking_enabled: true
  curiosity_sampling_enabled: true
  phenotype_dimensions:
    - "complexity"
    - "efficiency"

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
