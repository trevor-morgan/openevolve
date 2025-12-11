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
│   └── engine.py                # DiscoveryEngine (main integration)
│
├── examples/
│   └── discovery_mode/
│       ├── run_discovery.py     # Example runner
│       ├── initial_sort.py      # Starting program
│       ├── evaluator.py         # Multi-stage evaluator
│       ├── config.yaml          # Discovery config
│       └── README.md            # Usage guide
│
└── docs/
    └── DISCOVERY_MODE_ARCHITECTURE.md  # This document
```

---

## Implementation Phases

### Phase 1: Core Modules (Complete)
- [x] `ProblemSpace` and `ProblemEvolver`
- [x] `AdversarialSkeptic`
- [x] `EpistemicArchive` with phenotype tracking
- [x] `DiscoveryEngine` integration

### Phase 2: Deep Integration
- [ ] Modify `Controller.run()` to call DiscoveryEngine
- [ ] Add discovery config to main `Config` class
- [ ] Integrate problem context into prompt templates
- [ ] Add discovery metrics to evolution trace

### Phase 3: Advanced Features
- [ ] Blind reproduction test with separate LLM
- [ ] Adaptive attack selection based on history
- [ ] Cross-run knowledge transfer
- [ ] Multi-objective problem evolution

### Phase 4: Tooling
- [ ] Visualizer for problem evolution tree
- [ ] Discovery event timeline UI
- [ ] Phenotype space explorer
- [ ] Surprise heatmaps

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
