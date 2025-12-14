# Trading ML on Sparky

Evolves feature engineering code and trains real XGBoost models on sparky.local.

## How It Works

1. **OpenEvolve** generates variations of `compute_evolved_features()` using Claude Opus / Gemini 3
2. **Evaluator** SSHs to sparky.local and runs XGBoost training
3. **Walk-forward backtest** computes real Sharpe ratios
4. **MAP-Elites** selects best feature engineering approaches

## Metrics

- **Sharpe Ratio**: Risk-adjusted returns (target > 1.0)
- **Max Drawdown**: Worst peak-to-trough (target < 10%)
- **Win Rate**: Fraction of profitable trades (target > 52%)

## Requirements

- SSH access to sparky.local (no password - key auth)
- `ml_venv` on sparky with: sklearn, xgboost, pandas, numpy

## Running

```bash
cd ~/repos/forked/openevolve
source .venv/bin/activate
python openevolve-run.py examples/trading_ml_sparky/initial_program.py \
  examples/trading_ml_sparky/evaluator.py \
  --config examples/trading_ml_sparky/config.yaml \
  --iterations 30
```

## What Gets Evolved

The `compute_evolved_features()` function creates alpha signals from:
- OHLCV price data
- Pre-computed base features (momentum, volatility, RSI, etc.)

The LLM discovers combinations like:
- Volatility-adjusted momentum
- Cross-sectional ratios
- Multi-horizon signals
- Volume-price divergences
