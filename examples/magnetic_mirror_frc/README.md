# Magnetic Mirror / FRC Configuration Optimization

Evolve magnetic field configurations for plasma confinement in alternative fusion devices.

## Scientific Background

### Why Magnetic Mirrors?

While tokamaks (ITER) dominate fusion research, alternative concepts like **magnetic mirrors** and **field-reversed configurations (FRCs)** offer compelling advantages:

| Concept | Advantages | Active Companies |
|---------|------------|-----------------|
| **Magnetic Mirror** | Simpler geometry, easier maintenance, direct energy conversion | TAE Technologies (C-2W) |
| **FRC** | Highest beta (~1), compact, natural divertor | Helion Energy, TAE |
| **Tandem Mirror** | Electrostatic plugging, potential for Q>1 | Historical: MFTF, GAMMA-10 |

### The Physics

**Magnetic mirrors** confine plasma using the conservation of magnetic moment:

```
μ = mv²⊥ / 2B = constant
```

As a particle moves into a region of stronger field (B↑), its perpendicular velocity (v⊥) must increase. If v⊥ exceeds the total velocity, the particle reflects—this is the "mirror" effect.

**Key parameters:**

- **Mirror Ratio (R)**: `R = B_max / B_min`
  - Determines loss cone angle: `sin²(θ_loss) = 1/R`
  - R=5 → 27% of particles in loss cone
  - R=10 → 18% of particles in loss cone

- **Magnetic Well Depth (U)**: `U = (B_edge - B_center) / B_center`
  - Positive U → "minimum-B" configuration → MHD stable
  - Negative U → unstable to flute/interchange modes

### Historical Configurations

```
Simple Mirror (1950s)          Tandem Mirror (1970s-80s)
    ║                              ║    ║    ║
   ╔╩╗                            ╔╩╗  ╔╩╗  ╔╩╗
───╢ ╟────────────╢ ╟───     ────╢ ╟──╢ ╟──╢ ╟────
   ╚╦╝                            ╚╦╝  ╚╦╝  ╚╦╝
    ║                              ║    ║    ║
  Mirror                        Plug  Central  Plug

Baseball Coil (minimum-B)      Yin-Yang Coils
      ┌──┐                         ╱╲
     ╱    ╲                       ╱  ╲
    │      │                     │    │
     ╲    ╱                       ╲  ╱
      └──┘                         ╲╱
```

## What OpenEvolve Will Discover

Starting from a basic two-mirror configuration, OpenEvolve can potentially discover:

1. **Optimal mirror ratios** through coil current tuning
2. **Minimum-B configurations** using quadrupole fields
3. **Tandem arrangements** with multiple mirror cells
4. **Novel coil geometries** that balance confinement and stability

## Running the Example

### Prerequisites

```bash
# Make sure OpenEvolve is installed
pip install -e ".[dev]"

# Test the initial configuration
cd examples/magnetic_mirror_frc
python initial_program.py
python evaluator.py
```

### Configure LLM Access

Edit `config.yaml` to use your LLM provider:

```yaml
# For CLIProxyAPI (Claude Max subscription):
llm:
  primary_model: "openai/claude-sonnet-4-20250514"
  api_base: "http://localhost:8317/v1"

# For direct OpenAI:
llm:
  primary_model: "gpt-4o"
  api_base: "https://api.openai.com/v1"
```

### Run Evolution

```bash
# From repository root
python openevolve-run.py \
  examples/magnetic_mirror_frc/initial_program.py \
  examples/magnetic_mirror_frc/evaluator.py \
  --config examples/magnetic_mirror_frc/config.yaml \
  --iterations 50
```

### Visualize Results

```bash
python examples/magnetic_mirror_frc/visualize.py \
  --checkpoint openevolve_output/checkpoints/checkpoint_50/
```

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Mirror Ratio | B_max / B_min | > 5 (good), > 10 (excellent) |
| Well Depth | (B_edge - B_center) / B_center | > 0 (stable), > 0.1 (robust) |
| Field Uniformity | 1 - σ(B)/μ(B) in plasma region | > 0.9 |
| Confinement Time | From particle tracing | Maximize |
| Radial Stability | dB/dr > 0 at center | True |

## Scientific Merit

This example has genuine scientific value because:

1. **Real physics**: Uses actual Biot-Savart magnetic field calculations
2. **Particle simulation**: Traces guiding-center orbits for confinement estimates
3. **Stability analysis**: Checks MHD stability criteria (magnetic well, curvature)
4. **Open problem**: Optimal mirror configurations remain an active research area
5. **Industry relevance**: TAE, Helion, and others are actively developing these concepts

### Potential Discoveries

OpenEvolve might rediscover known configurations (validation) or find novel arrangements:

- **Known**: Baseball coils achieve minimum-B through quadrupole fields
- **Known**: Tandem mirrors use electrostatic plugging
- **Novel?**: Optimal number and placement of shaping coils
- **Novel?**: Hybrid configurations combining different stabilization approaches

## References

1. Post, R.F. (1987). "The magnetic mirror approach to fusion." *Nuclear Fusion* 27(10).
2. Ryutov, D.D. (1988). "Open-ended traps." *Soviet Physics Uspekhi* 31(4).
3. Binderbauer, M.W. et al. (2015). "A high performance field-reversed configuration." *Physics of Plasmas* 22.
4. Wurden, G.A. et al. (2016). "Magneto-inertial fusion." *Journal of Fusion Energy* 35.

## Discovery Mode & Golden Path

This example demonstrates OpenEvolve's advanced discovery features:

### Heisenberg Engine (Collaborative Discovery)

When evolution plateaus, the **Heisenberg Engine** activates a team of specialized agents to discover NEW physics variables:

- **Theorist**: Proposes new physics from first principles
- **Experimentalist**: Designs tests and measurements
- **Skeptic**: Challenges ideas, finds flaws
- **Synthesizer**: Combines insights into working code

Enable in `config.yaml`:
```yaml
discovery:
  heisenberg:
    enabled: true
    collaborative_discovery_enabled: true
    max_debate_rounds: 4
```

### Golden Path (Autonomous Ontological Discovery)

The **Golden Path** uses external tools to discover hidden variables that don't have names yet:

| Tool | Discovers |
|------|-----------|
| Symbolic Regression | Mathematical formulas: `fitness ~ sqrt(n_coils)` |
| Causal Discovery | Causal relationships: `coil_count → confinement` |
| Code Analysis | Structural patterns in evolved programs |

Enable in `config.yaml`:
```yaml
discovery:
  golden_path:
    enabled: true
    prescience_short_window: 10
    min_programs_for_analysis: 15
```

### Running Discovery Tests

```bash
# Test the Golden Path toolkit
python examples/magnetic_mirror_frc/test_toolkit.py

# Test collaborative discovery
python examples/magnetic_mirror_frc/test_collaborative_discovery.py
```

### Example Discovery Output

When the Golden Path activates:
```
Phase 1: Mentat analyzing programs...
  Found 8 significant patterns

Phase 2: DiscoveryToolkit running external tools...
  symbolic_regression: fitness ~ 0.3 + 0.1*sqrt(n_coils)
  causal_discovery: n_coils → fitness (r=0.42)
  code_analysis: 12 correlated patterns

Phase 3: SietchFinder proposing hidden variables...
  7 candidates proposed

Phase 4: GomJabbar validating hypotheses...
  ✓ coil_count VALIDATED (r=0.45, incr_r2=0.15)
  ✗ random_metric failed (r=0.05)

Phase 5: SpiceAgony integrating discoveries...
  2 new variables integrated
```

## Extending This Example

Ideas for enhancement:

- Add 3D coil geometries (not just circular)
- Implement full Grad-Shafranov equilibrium solver
- Add thermal particle distributions
- Include collisional effects (Pastukhov confinement)
- Optimize for specific fusion reactions (D-T, D-He3, p-B11)
- Add custom discovery tools (e.g., plasma physics-specific analyzers)
