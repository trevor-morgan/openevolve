# Matrix Cache Demo: Ontological Expansion

This example demonstrates the **Heisenberg Engine** with a more challenging problem where the hidden variable (cache locality) truly matters.

## The Problem

Optimize matrix multiplication without using numpy.

**Hidden Variable**: Loop ordering affects cache efficiency dramatically.

| Loop Order | Access Pattern | Cache Behavior | Fitness |
|------------|---------------|----------------|---------|
| ijk (naive) | B column-wise | Cache unfriendly | ~0.5 |
| ikj | B row-wise | Cache friendly | ~0.95 |

## Why This Is Hard

1. **Both implementations are O(n³)** - same algorithmic complexity
2. **Both are correct** - produce identical results
3. **The only difference is loop order** - which seems arbitrary
4. **The performance gap is 2x** - but the reason is hidden (cache behavior)

A program optimizing for "performance" will try various algorithmic improvements but won't understand why `ikj` beats `ijk` unless it discovers that **memory access patterns matter**.

## How Heisenberg Engine Helps

1. **Crisis Detection**: After ~25 iterations of plateau at ~0.5 fitness
2. **Probe Synthesis**: Analyzes `cache_analysis` artifacts, discovers `b_access_pattern` correlates with performance
3. **Ontological Expansion**: Adds "B matrix access pattern" to known variables
4. **Soft Reset**: Updates problem description to mention cache locality

## Running the Demo

```bash
# From the OpenEvolve root directory
python openevolve-run.py \
    examples/matrix_cache_demo/initial_program.py \
    examples/matrix_cache_demo/evaluator.py \
    --config examples/matrix_cache_demo/config.yaml \
    --iterations 100 \
    --output matrix_cache_output
```

## Expected Evolution

```
Initial (ijk):     combined_score = 0.496, b_access_pattern = "column_wise"
After discovery:   combined_score = 0.958, b_access_pattern = "row_wise"
```

## The Science

The cache effect comes from how matrices are stored in memory (row-major in Python):

```
Matrix B in memory: [B[0][0], B[0][1], B[0][2], ..., B[1][0], B[1][1], ...]

ijk order: for j: for k: ... B[k][j]  # Jumps between rows - cache misses!
ikj order: for k: for j: ... B[k][j]  # Stays in same row - cache hits!
```

When you access `B[k][j]` with `j` varying fast (ikj), you traverse row k sequentially.
When you access `B[k][j]` with `k` varying fast (ijk), you jump between rows.

CPU caches load entire cache lines (64 bytes = 8 floats), so sequential access is much faster.

## Key Files

| File | Description |
|------|-------------|
| `initial_program.py` | Naive ijk matrix multiplication |
| `evaluator.py` | Evaluator with hidden cache simulation |
| `config.yaml` | Configuration with Heisenberg enabled |
