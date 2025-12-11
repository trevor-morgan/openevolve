# Heisenberg Engine Demo: Ontological Expansion

This example demonstrates the **Heisenberg Engine** - OpenEvolve's ability to discover hidden variables that are limiting optimization.

## The Problem

We're evolving a sorting algorithm. The evaluator measures:
- **Correctness**: Does it sort correctly?
- **Performance**: How fast is it?

But there's a **hidden variable**: cache efficiency. The evaluator simulates CPU cache behavior - algorithms that access memory sequentially get cache hits, while those with random access patterns suffer cache misses.

## Why Traditional Optimization Gets Stuck

A program optimizing for "performance" will:
1. Improve algorithmic complexity (O(n²) → O(n log n))
2. Hit a plateau around 0.6-0.7 fitness
3. Not understand why it can't improve further

The reason: cache effects can cause 2-10x performance differences between algorithms with the same complexity, but the program doesn't know cache exists!

## How Heisenberg Engine Helps

The Heisenberg Engine detects this situation:

1. **Crisis Detection**: After ~30 iterations, it notices:
   - Fitness has plateaued
   - There's unexplained variance (some programs with similar code perform very differently)

2. **Probe Synthesis**: It generates probes that analyze evaluation artifacts:
   - Looks for patterns in the `cache_analysis` data
   - Discovers that `cache_hit_rate` correlates with performance

3. **Ontological Expansion**: It adds "memory access pattern" to the known variables:
   - Updates the problem description
   - The LLM now knows cache locality matters
   - Evolution can optimize for cache-friendly algorithms

4. **Soft Reset**: Keeps top programs but re-evaluates with new knowledge

## Running the Demo

```bash
# From the OpenEvolve root directory
python openevolve-run.py \
    examples/heisenberg_demo/initial_program.py \
    examples/heisenberg_demo/evaluator.py \
    --config examples/heisenberg_demo/config.yaml \
    --iterations 200 \
    --output heisenberg_demo_output
```

## Expected Output

```
INFO: Iteration 30: Fitness 0.65
INFO: Iteration 50: Fitness 0.68
INFO: Iteration 70: Fitness 0.68 (improvement: 0.00)

WARNING: EPISTEMIC CRISIS DETECTED
         Type: plateau
         Confidence: 0.75

INFO: Synthesizing probes for hidden variable discovery...
INFO: Probe discovered: 'cache_hit_rate' correlates with performance (r=0.73)

INFO: ONTOLOGY EXPANDED
      New variable: 'memory_access_pattern'

INFO: Performing soft reset...
INFO: Iteration 75: Fitness 0.82 (breakthrough!)
INFO: Iteration 100: Fitness 0.91
```

## Key Files

| File | Description |
|------|-------------|
| `initial_program.py` | Simple bubble sort to evolve |
| `evaluator.py` | Evaluator with hidden cache simulation |
| `config.yaml` | Configuration with Heisenberg enabled |

## What to Look For

1. **Before Crisis**: Programs plateau at ~0.65-0.70 fitness
2. **Crisis Detection**: Look for "EPISTEMIC CRISIS DETECTED" in logs
3. **Probe Discovery**: Look for correlation analysis in probe output
4. **After Expansion**: Programs should break through to 0.85+ fitness

## The Science

This demonstrates a key insight for AI-driven discovery:

> Traditional AI optimizes relationships between KNOWN variables.
> True discovery requires finding NEW variables.

The Heisenberg Engine implements "ontological expansion" - the ability to expand the state space when optimization is fundamentally stuck. Named after Heisenberg's insight that observation affects measurement, it recognizes that what we choose to measure affects what we can discover.

## Configuration Options

```yaml
heisenberg:
  enabled: true

  # When to trigger crisis detection
  min_plateau_iterations: 30
  fitness_improvement_threshold: 0.005

  # Probe behavior
  max_probes_per_crisis: 3
  probe_timeout: 30.0

  # Validation requirements
  validation_trials: 3
  min_correlation_threshold: 0.5
```
