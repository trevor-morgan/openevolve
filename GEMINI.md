# OpenEvolve Context for Gemini

## Project Overview
OpenEvolve is an open-source implementation of AlphaEvolve, an evolutionary coding agent that uses Large Language Models (LLMs) to optimize code and discover new algorithms. It transforms static code optimization into a dynamic evolutionary process.

**Key Features:**
*   **MAP-Elites w/ Islands:** Uses Quality-Diversity evolution with isolated populations (islands) to prevent premature convergence.
*   **LLM-Driven Mutation:** Uses LLMs (OpenAI, Gemini, etc.) to mutate and improve code based on evolutionary history.
*   **Artifact Side-Channel:** Programs can return rich feedback (stderr, profiling data) to guide the next generation of evolution.
*   **Discovery Mode:** Can evolve the problem description itself to find novel challenges and solutions.

## High-Level Architecture
*   **Controller (`openevolve/controller.py`):** Orchestrates the evolution loop, managing parallel workers.
*   **Database (`openevolve/database.py`):** Implements the MAP-Elites algorithm, storing programs in a multi-dimensional feature grid.
*   **Evaluator (`openevolve/evaluator.py`):** Runs the code candidates. Supports a "cascade" pattern (fast checks -> full benchmarks).
*   **LLM Integration (`openevolve/llm/`):** Handles communication with AI models, supporting ensembles and fallbacks.

## Building and Running

### Installation
The project is a standard Python package managed with `pyproject.toml`.

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"
# Or use the Makefile
make install-dev
```

### Running Evolution
The primary entry point is `openevolve-run.py`.

```bash
# Basic run
python openevolve-run.py <initial_program.py> <evaluator.py> --config <config.yaml> --iterations 100

# Example: Function Minimization
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

### Running Tests
Tests are split into unit (fast) and integration (requires LLM/server).

```bash
# Run unit tests only (Recommended for general dev)
make test
# OR
pytest tests -v --ignore tests/integration

# Run all tests (requires configuration)
make test-all
```

### Formatting
The project uses `black` for code formatting.

```bash
make lint
# OR
python -m black openevolve examples tests scripts
```

### Visualization
An interactive visualizer is available to track evolution.

```bash
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/
```

## Development Conventions
*   **Python Version:** >= 3.10
*   **Configuration:** All evolution parameters are controlled via YAML files (see `configs/`).
    *   **Consolidated Config:** Hardcoded parameters in RL and Discovery modules have been moved to `Config` dataclasses.
*   **Architectural Interfaces:** Core P-R-A (Perception-Reasoning-Action) interfaces are defined in `openevolve/interfaces.py`.
*   **LLM Provider:** Uses `OPENAI_API_KEY` by default. Can be configured for Gemini (`api_base: ...`, `model: gemini-2.5-pro`) or local models (Ollama).
*   **Marking Code:** Use `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` to designate specific sections of code for the LLM to optimize.

## Key Directories
*   `openevolve/`: Core source code.
*   `examples/`: Rich set of examples (math, GPU kernels, sorting). **Reference these for how to structure evaluators.**
*   `tests/`: Test suite.
*   `configs/`: Example configurations.
