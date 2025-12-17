# Sonofusion Discovery
## Seeking Thermonuclear Conditions in Cavitation Bubbles

This experiment evolves simulation codes for Single-Bubble Sonoluminescence (SBSL) to explore the parameter space for "Sonofusion" (Bubble Fusion).

### Goals
1.  **Model Evolution:** Start with a standard Keller-Miksis model and evolve it to include advanced physics (Real gas EOS, dissociation, ionization, shock convergence).
2.  **Parameter Discovery:** Find the optimal drive frequency, amplitude, and liquid properties to maximize core temperature ($T_{max}$) and density ($
ho_{max}$).
3.  **Ontological Discovery:** Use `Golden Path` and `Heisenberg` engines to identify hidden variables limiting collapse intensity (e.g., shape stability vs. collapse speed).

### Running
```bash
python openevolve-run.py \
  examples/sonofusion/initial_program.py \
  examples/sonofusion/evaluator.py \
  --config examples/sonofusion/config.yaml \
  --iterations 50
```
