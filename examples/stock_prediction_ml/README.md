# Stock Prediction ML - Full Scientific Discovery Example

Evolve PyTorch neural networks for stock return prediction using GPU training on DGX Spark.

## Features

This example demonstrates **all** OpenEvolve advanced features:

| Feature | Purpose |
|---------|---------|
| **RL Adaptive Selection** | Learns when to explore new architectures vs exploit working ones |
| **Meta-Prompting** | Learns which optimization strategies work for financial ML |
| **Discovery Mode** | Evolves the problem as solutions are found |
| **Adversarial Skeptic** | Tests models against edge cases |
| **Heisenberg Engine** | Discovers hidden market variables (regimes, correlations) |

## Discovery Requirements

- The evaluator must accept `problem_context` (see `evaluator.py`). If it doesn’t, OpenEvolve auto-disables problem evolution/coevolution to prevent “fictional” problem drift.
- The skeptic treats non-finite numeric outputs (NaN/inf) from `skeptic_entrypoint` as falsification; handle NaNs/shape edge cases and return finite metrics.
- Heisenberg “soft reset” keeps only the top N programs after ontology expansion (`discovery.heisenberg.programs_to_keep_on_reset` in `config.yaml`).

## What Gets Evolved

The `initial_program.py` contains 4 evolvable sections:

1. **Feature Engineering** (`engineer_features`)
   - Technical indicators
   - Volume patterns
   - Cross-asset signals

2. **Model Architecture** (`StockPredictor`)
   - LSTM layers
   - Attention mechanisms
   - Hidden sizes, dropout

3. **Loss Function** (`compute_loss`)
   - MSE vs Huber vs quantile
   - Directional accuracy bonuses
   - Regularization

4. **Training Loop** (`train_model`)
   - Learning rate schedules
   - Optimizers
   - Early stopping

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Alpaca API

The `.env` file should contain:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
DGX_HOST=user@dgx-host
DGX_WORK_DIR=/tmp/openevolve_stocks
```

### 3. Ensure DGX Access

```bash
# Test SSH connection
ssh trevor-morgan@sparky.local "nvidia-smi"

# Ensure PyTorch is installed on DGX
ssh trevor-morgan@sparky.local "python3 -c 'import torch; print(torch.cuda.is_available())'"
```

### 4. Install PyTorch on DGX (if needed)

```bash
ssh trevor-morgan@sparky.local "pip install torch --index-url https://download.pytorch.org/whl/cu121"
```

## Usage

### Run Evolution

```bash
cd examples/stock_prediction_ml
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### Test Evaluator Locally

```bash
python evaluator.py
```

### Resume from Checkpoint

```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --checkpoint openevolve_output/checkpoints/checkpoint_100/
```

## Fitness Metrics

The evaluator combines multiple metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Direction Accuracy | 35% | Predicting up/down correctly |
| Sharpe Score | 30% | Risk-adjusted returns |
| Information Coefficient | 20% | Prediction-return correlation |
| MSE Score | 15% | Raw prediction accuracy |

## Hidden Variables (Heisenberg Engine)

The Heisenberg Engine may discover:

- **Market Regime**: Bull/bear/sideways states
- **Volatility Clustering**: GARCH-like effects
- **Sector Correlations**: Cross-asset dependencies
- **Calendar Effects**: Day-of-week, month-end patterns
- **Liquidity Patterns**: Volume-based regime shifts

When the engine detects a plateau (optimization stuck), it synthesizes probes to discover these hidden variables and expands the model's state space.

## Expected Results

| Metric | Random | After 100 iters | After 500 iters |
|--------|--------|-----------------|-----------------|
| Direction Accuracy | 50% | 52-54% | 54-58% |
| Sharpe | 0.0 | 0.2-0.5 | 0.5-1.0 |
| IC | 0.0 | 0.02-0.05 | 0.05-0.10 |

**Note**: Financial prediction is hard. Even small improvements are meaningful.

## Tips

1. **Start with fewer symbols** for faster iteration:
   ```python
   SYMBOLS = ["SPY", "QQQ"]  # Just 2 ETFs
   ```

2. **Reduce sequence length** for faster training:
   ```python
   SEQUENCE_LENGTH = 30  # 30 days instead of 60
   ```

3. **Watch for overfitting**: If train metrics improve but validation doesn't, the model is overfitting.

4. **Check the evolution trace**: `openevolve_output/evolution_trace.jsonl` contains detailed logs of what worked.
