# ARC Topology Evolution

This example evolves graph topology algorithms using OpenEvolve.

## Problem Types

1. **Betti Numbers** (`betti`) - Compute connected components (β₀) and cycles (β₁)
2. **Directed Clique Counting** (`clique`) - Count k-simplices in directed graphs
3. **Integration Score** (`integration`) - Compute min-cut (simplified Φ)

## Setup

1. Start CLIProxyAPI:
   ```bash
   ~/cliproxyapi/cli-proxy-api
   ```

2. Run evolution:
   ```bash
   cd examples/arc_topology
   python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 50
   ```

## Changing Problem Type

Edit `evaluator.py` line 175:
```python
PROBLEM_TYPE = "betti"  # Change to "clique" or "integration"
```

## Expected Results

- **Betti Numbers**: Should achieve 100% accuracy with Euler characteristic formula
- **Clique Counting**: May take longer due to O(n⁴) complexity
- **Integration**: NP-hard, exponential in graph size

## Configuration

The config uses Claude Sonnet as primary model for fast iteration. Edit `config.yaml` to use different models:

```yaml
primary_model: "claude-opus-4-5-20251101"  # More capable but slower
```
