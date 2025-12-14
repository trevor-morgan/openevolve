# Trading Alpha Evolution

Evolves trading strategies using AFML (Advances in Financial Machine Learning) techniques.

## Metrics

- **Sharpe Ratio**: Risk-adjusted returns (target > 1.5)
- **Max Drawdown**: Worst peak-to-trough (target < 15%)
- **Win Rate**: Fraction of profitable days (target > 52%)

## Features Available

The evaluator provides these features for signal generation:

- `momentum_5`: 5-day price momentum
- `momentum_20`: 20-day price momentum
- `volatility_10`: 10-day rolling volatility
- `rsi_14`: 14-day RSI
- `ma_cross`: Moving average crossover signal

## Running

```bash
cd examples/trading_alpha
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 100
```

## AFML Concepts

The LLM is prompted to explore:

1. **Fractional Differentiation**: Balance stationarity vs memory retention
2. **Triple Barrier Labeling**: Better labels than simple returns
3. **Meta-Labeling**: Separate signal quality from direction
4. **Bet Sizing**: Kelly criterion and position management

## Expected Evolution

The strategy should evolve from simple momentum to:

1. Multi-factor combination
2. Regime-aware switching
3. Dynamic position sizing
4. Risk parity allocation
