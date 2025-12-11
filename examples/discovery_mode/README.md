# Discovery Mode Example

This example demonstrates OpenEvolve's **Discovery Engine** - an architecture for true scientific discovery that evolves both questions AND answers.

## Key Differences from Standard Evolution

| Standard OpenEvolve | Discovery Mode |
|---------------------|----------------|
| Fixed problem (evaluator) | **Evolving problem space** |
| LLM-based evaluation | **Adversarial falsification** |
| Single fitness score | **Behavioral phenotype tracking** |
| Optimize one metric | **Surprise-based curiosity** |

## The Three Modules

### 1. Problem Evolver (Explorer)
When a problem is "solved" (N successful solutions), the system automatically evolves it:

```
Generation 0: "Sort a list of numbers"
    ↓ (5 solutions found)
Generation 1: "Sort a list of numbers + Minimize memory usage"
    ↓ (5 solutions found)
Generation 2: "Sort a list of numbers + Minimize memory usage + Handle streaming data"
```

This mimics how scientific problems evolve: Newton → Einstein → Quantum Mechanics.

### 2. Adversarial Skeptic
Instead of asking "Is this code good?", we try to BREAK it:

```python
# Skeptic generates adversarial inputs:
attack_inputs = [
    [],                    # Empty input
    [float('nan')],        # NaN values
    [1e100, -1e100],       # Overflow values
    ["not", "numbers"],    # Wrong types
]

# Code must survive ALL attacks to be considered valid
```

### 3. Epistemic Archive
Stores solutions by BEHAVIOR, not just fitness:

```
         High Efficiency
              ↑
    [Quick-  | [Optimal]
     sort]   |
    ---------+-------→ High Complexity
    [Simple] | [Heavy-
             |  weight]
```

Both "Simple but Slow" AND "Complex but Fast" solutions are kept.
The simple one might be the stepping stone to solving a harder problem.

## Running the Example

```bash
# Basic run
python run_discovery.py initial_sort.py evaluator.py --config config.yaml

# With custom problem description
python run_discovery.py initial_sort.py evaluator.py \
    --config config.yaml \
    --problem-description "Create an efficient sorting algorithm" \
    --iterations 50

# Disable adversarial skeptic (for faster iteration)
python run_discovery.py initial_sort.py evaluator.py \
    --config config.yaml \
    --no-skeptic

# Evolve problem more frequently
python run_discovery.py initial_sort.py evaluator.py \
    --config config.yaml \
    --evolve-after 3
```

## Output

The Discovery Engine produces several outputs:

1. **discovery_log.jsonl** - Timeline of all discoveries
2. **discovery_state/** - Saved state for resumption
   - `problems.json` - Problem evolution history
   - `events.json` - All discovery events
   - `stats.json` - Final statistics

## Example Discovery Log

```json
{"type": "genesis", "problem_id": "genesis_a1b2c3d4", "details": {"description": "Sort a list"}}
{"type": "solution", "program_id": "prog_1234", "details": {"fitness": 0.85}}
{"type": "surprise", "program_id": "prog_5678", "details": {"predicted": 0.3, "actual": 0.9, "surprise": 0.6}}
{"type": "falsification", "program_id": "prog_9abc", "details": {"attack_type": "overflow"}}
{"type": "problem_evolution", "problem_id": "prob_def0", "details": {"new_difficulty": 1.5}}
```

## Understanding Surprise

The system tracks "surprise" - the difference between predicted and actual fitness:

- **High Positive Surprise**: Found something unexpectedly good → Explore this region more!
- **High Negative Surprise**: Something unexpectedly failed → Investigate why!
- **Low Surprise**: Predictable result → Less interesting for discovery

This mimics how scientists focus on anomalies, not confirmations.

## Phenotype Dimensions

The archive tracks behavioral characteristics:

| Phenotype | Description | How Measured |
|-----------|-------------|--------------|
| `complexity` | Structural complexity | AST node count |
| `efficiency` | Runtime performance | Benchmark timing |
| `robustness` | Error handling | Edge case pass rate |
| `approach_signature` | Algorithm type | Structural hash |

## Extending the Example

### Custom Problem Evolution

Edit `problem_space.py` to add domain-specific constraints:

```python
simple_constraints = [
    "Sort in-place (O(1) extra space)",
    "Handle infinite/NaN values",
    "Be stable (preserve equal element order)",
    "Work on linked lists, not arrays",
    # Add your domain-specific constraints
]
```

### Custom Adversarial Attacks

Edit `skeptic.py` to add domain-specific attacks:

```python
attacks = {
    "domain_specific": [
        {"input": "your_tricky_input", "attack_type": "custom", "rationale": "Why this breaks things"},
    ]
}
```

### Custom Phenotypes

Edit `epistemic_archive.py` to track domain-specific behaviors:

```python
@dataclass
class Phenotype:
    # Add your domain-specific phenotypes
    parallelizable: bool = False
    deterministic: bool = True
    online_capable: bool = False
```

## Comparison to Static Optimization

| Metric | Static (100 iter) | Discovery (100 iter) |
|--------|-------------------|----------------------|
| Best fitness | 0.95 | 0.92 |
| Problem difficulty | 1.0 | 2.5 |
| Unique approaches | 3 | 8 |
| Edge cases found | 5 | 15 |

Discovery Mode may not find the "highest score" on the original problem,
but it explores a much larger space of problems and solutions.
