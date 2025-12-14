# OpenEvolve Configuration Files

This directory contains configuration files for OpenEvolve with examples for different use cases.

## Configuration Files

### `default_config.yaml`
The main configuration file containing all available options with sensible defaults. This file includes:
- Complete documentation for all configuration parameters
- Default values for all settings
- **Island-based evolution parameters** for proper evolutionary diversity
- **Adaptive systems** (meta‑prompting and RL selection) that learn online in both single‑process and process‑parallel runs

Use this file as a template for your own configurations.

### `island_config_example.yaml`
A practical example configuration demonstrating proper island-based evolution setup. Shows:
- Recommended island settings for most use cases
- Balanced migration parameters
- Complete working configuration

### `island_examples.yaml`
Multiple example configurations for different scenarios:
- **Maximum Diversity**: Many islands, frequent migration
- **Focused Exploration**: Few islands, rare migration
- **Balanced Approach**: Default recommended settings
- **Quick Exploration**: Small-scale rapid testing
- **Large-Scale Evolution**: Complex optimization runs

Includes guidelines for choosing parameters based on your problem characteristics.

### `cuda_optimization.yaml`
A comprehensive configuration for GPU kernel optimization that combines:
- **RL-based Adaptive Selection**: Learns when to explore vs exploit based on evolution progress
- **CUDA-specific Meta-prompting**: Uses empirically-derived strategies from GB10 analysis
- **Hardware-aware Settings**: Tuned for CUDA kernel development cycles

Key features:
- 14 CUDA optimization strategies with success rates from real evolution runs
- Contextual Thompson Sampling for both RL selection and meta-prompting
- Higher exploration rates suitable for GPU optimization (many viable approaches)
- Extended timeouts for CUDA compilation and benchmarking
- Evolution trace enabled for post-run analysis

Use this configuration for:
- GEMM kernel optimization
- Attention kernel optimization (FlashAttention-style)
- Any CUDA kernel performance improvement task

```bash
python openevolve-run.py kernel.cu evaluator.py --config configs/cuda_optimization.yaml
```

### `full_scientific_discovery.yaml`
The most comprehensive configuration enabling ALL advanced features for open-ended scientific discovery:

| Feature | Status | Purpose |
|---------|--------|---------|
| RL Adaptive Selection | Enabled | Learns when to explore vs exploit |
| Meta-Prompting | Enabled | Learns which prompt strategies work |
| Discovery Mode | Enabled | Problem evolution + surprise exploration |
| Adversarial Skeptic | Enabled | Tries to break programs (falsification) |
| Heisenberg Engine | Enabled | Discovers hidden variables when stuck |
| Evolution Trace | Enabled | Exports data for offline RL training |

Key settings:
- 2000 iterations, 7 islands, 2000 population
- High diversity weights (0.25) and novelty weights (0.15)
- Comprehensive code instrumentation for probe analysis
- Crisis detection at 75 iterations plateau with 0.65 confidence threshold

Use this for:
- Open-ended optimization where solution space is unknown
- Scientific discovery tasks
- Problems where hidden variables might affect fitness
- Long-running evolution that might get stuck

```bash
python openevolve-run.py program.py evaluator.py \
  --config configs/full_scientific_discovery.yaml \
  --iterations 2000
```

**Important**: Set `discovery.problem_description` in the config for best results!

## Island-Based Evolution Parameters

The key new parameters for proper evolutionary diversity are:

```yaml
database:
  num_islands: 5                      # Number of separate populations
  migration_interval: 50              # Migrate every N generations  
  migration_rate: 0.1                 # Fraction of top programs to migrate
```

### Parameter Guidelines

- **num_islands**: 3-10 for most problems (more = more diversity)
- **migration_interval**: 25-100 generations (higher = more independence)
- **migration_rate**: 0.05-0.2 (5%-20%, higher = faster knowledge sharing)

### When to Use What

- **Complex problems** → More islands, less frequent migration
- **Simple problems** → Fewer islands, more frequent migration
- **Long runs** → More islands to maintain diversity
- **Short runs** → Fewer islands for faster convergence

## Usage

Copy any of these files as a starting point for your configuration:

```bash
cp configs/default_config.yaml my_config.yaml
# Edit my_config.yaml for your specific needs
```

Then use with OpenEvolve:

```python
from openevolve import OpenEvolve
evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py",
    config_path="my_config.yaml"
)
```
