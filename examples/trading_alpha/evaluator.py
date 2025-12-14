"""
Evaluator for trading alpha discovery using AFML techniques.

Evolves trading strategies that implement:
- Fractional differentiation for stationarity
- Triple barrier labeling
- Signal generation from features

The program must implement `generate_signals(prices, features)` that returns
buy/sell signals with proper risk management.
"""

import importlib.util
import math
import random
import traceback

from openevolve.evaluation_result import EvaluationResult

# =============================================================================
# Simulated market data generator
# =============================================================================


def generate_market_data(days: int = 252, seed: int | None = None) -> dict:
    """Generate synthetic market data for backtesting."""
    if seed is not None:
        random.seed(seed)

    # Generate price series with trend and noise
    prices = [100.0]
    returns = []

    for _ in range(days - 1):
        # Random walk with slight mean reversion
        drift = 0.0001  # Small positive drift
        volatility = 0.02
        shock = random.gauss(0, volatility)
        mean_reversion = -0.01 * (prices[-1] - 100) / 100

        ret = drift + mean_reversion + shock
        returns.append(ret)
        prices.append(prices[-1] * (1 + ret))

    # Generate features
    features = {
        "momentum_5": compute_momentum(prices, 5),
        "momentum_20": compute_momentum(prices, 20),
        "volatility_10": compute_volatility(returns, 10),
        "rsi_14": compute_rsi(prices, 14),
        "ma_cross": compute_ma_crossover(prices, 10, 50),
    }

    return {
        "prices": prices,
        "returns": returns,
        "features": features,
        "days": days,
    }


def compute_momentum(prices: list[float], window: int) -> list[float]:
    """Compute momentum indicator."""
    result = [0.0] * len(prices)
    for i in range(window, len(prices)):
        result[i] = (prices[i] - prices[i - window]) / prices[i - window]
    return result


def compute_volatility(returns: list[float], window: int) -> list[float]:
    """Compute rolling volatility."""
    result = [0.0] * (len(returns) + 1)
    for i in range(window, len(returns)):
        window_returns = returns[i - window : i]
        mean = sum(window_returns) / window
        variance = sum((r - mean) ** 2 for r in window_returns) / window
        result[i + 1] = math.sqrt(variance)
    return result


def compute_rsi(prices: list[float], window: int) -> list[float]:
    """Compute RSI indicator."""
    result = [50.0] * len(prices)

    for i in range(window + 1, len(prices)):
        gains = []
        losses = []
        for j in range(i - window, i):
            change = prices[j + 1] - prices[j]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = sum(gains) / window if gains else 0.001
        avg_loss = sum(losses) / window if losses else 0.001

        rs = avg_gain / avg_loss
        result[i] = 100 - (100 / (1 + rs))

    return result


def compute_ma_crossover(prices: list[float], fast: int, slow: int) -> list[float]:
    """Compute moving average crossover signal."""
    result = [0.0] * len(prices)

    for i in range(slow, len(prices)):
        fast_ma = sum(prices[i - fast : i]) / fast
        slow_ma = sum(prices[i - slow : i]) / slow
        result[i] = (fast_ma - slow_ma) / slow_ma

    return result


# =============================================================================
# Backtest engine
# =============================================================================


def backtest_signals(
    prices: list[float], signals: list[float], transaction_cost: float = 0.001
) -> dict:
    """
    Backtest trading signals.

    Args:
        prices: Price series
        signals: -1 (short), 0 (flat), +1 (long) for each day
        transaction_cost: Cost per trade as fraction

    Returns:
        Performance metrics
    """
    n = len(prices)
    if len(signals) != n:
        return {"error": "Signal length mismatch", "sharpe": -10.0}

    # Track portfolio
    position = 0.0
    pnl = []
    trades = 0

    for i in range(1, n):
        # PnL from position
        ret = (prices[i] - prices[i - 1]) / prices[i - 1]
        daily_pnl = position * ret

        # Position change
        new_position = signals[i]
        if new_position != position:
            # Transaction cost
            daily_pnl -= abs(new_position - position) * transaction_cost
            trades += 1
            position = new_position

        pnl.append(daily_pnl)

    # Compute metrics
    if not pnl or all(p == 0 for p in pnl):
        return {
            "sharpe": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": trades,
        }

    total_return = sum(pnl)
    mean_return = sum(pnl) / len(pnl)
    std_return = math.sqrt(sum((p - mean_return) ** 2 for p in pnl) / len(pnl))

    sharpe = (mean_return / std_return) * math.sqrt(252) if std_return > 0 else 0.0

    # Max drawdown
    cumulative = [0.0]
    for p in pnl:
        cumulative.append(cumulative[-1] + p)
    peak = cumulative[0]
    max_dd = 0.0
    for c in cumulative:
        peak = max(peak, c)
        dd = (peak - c) / (peak + 1e-6)
        max_dd = max(max_dd, dd)

    # Win rate
    wins = sum(1 for p in pnl if p > 0)
    win_rate = wins / len(pnl)

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "trades": trades,
        "mean_daily_return": round(mean_return, 6),
    }


# =============================================================================
# Evaluator functions
# =============================================================================


def evaluate(program_path: str) -> EvaluationResult:
    """
    Full evaluation with multiple market scenarios.
    """
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "generate_signals"):
            return EvaluationResult(
                metrics={
                    "sharpe": -10.0,
                    "combined_score": 0.0,
                    "error": "Missing generate_signals function",
                },
                artifacts={"error_type": "MissingFunction"},
            )

        # Test on multiple market scenarios
        scenarios = [
            ("trending_up", generate_market_data(252, seed=42)),
            ("trending_down", generate_market_data(252, seed=123)),
            ("choppy", generate_market_data(252, seed=456)),
            ("volatile", generate_market_data(252, seed=789)),
        ]

        all_sharpes = []
        all_returns = []
        all_drawdowns = []
        errors = []

        for name, data in scenarios:
            try:
                signals = program.generate_signals(data["prices"], data["features"])

                # Validate signals
                if not isinstance(signals, list):
                    errors.append(f"{name}: signals must be a list")
                    continue

                if len(signals) != len(data["prices"]):
                    errors.append(f"{name}: signal length mismatch")
                    continue

                # Normalize signals to -1, 0, +1
                normalized = []
                for s in signals:
                    if s > 0.3:
                        normalized.append(1.0)
                    elif s < -0.3:
                        normalized.append(-1.0)
                    else:
                        normalized.append(0.0)

                result = backtest_signals(data["prices"], normalized)

                all_sharpes.append(result["sharpe"])
                all_returns.append(result["total_return"])
                all_drawdowns.append(result["max_drawdown"])

            except Exception as e:
                errors.append(f"{name}: {e!s}")
                all_sharpes.append(-5.0)

        # Aggregate metrics
        avg_sharpe = sum(all_sharpes) / len(all_sharpes) if all_sharpes else -10.0
        avg_return = sum(all_returns) / len(all_returns) if all_returns else 0.0
        max_dd = max(all_drawdowns) if all_drawdowns else 1.0

        # Combined score favoring high Sharpe with risk control
        # Sharpe > 1.5 is good, > 2.0 is excellent
        # Drawdown < 10% is acceptable, < 5% is good
        sharpe_score = min(1.0, max(0.0, (avg_sharpe + 2) / 4))  # Maps [-2, 2] to [0, 1]
        dd_penalty = max(0.0, 1.0 - max_dd * 2)  # 50% DD = 0 score

        combined_score = 0.7 * sharpe_score + 0.3 * dd_penalty

        # Bonus for consistent positive returns across scenarios
        if all(s > 0 for s in all_sharpes):
            combined_score *= 1.2

        return EvaluationResult(
            metrics={
                "sharpe": round(avg_sharpe, 3),
                "total_return": round(avg_return, 4),
                "max_drawdown": round(max_dd, 4),
                "combined_score": round(combined_score, 4),
                "scenarios_passed": len(all_sharpes) - len(errors),
            },
            artifacts={
                "scenario_sharpes": all_sharpes,
                "errors": errors[:3],
            },
        )

    except Exception as e:
        return EvaluationResult(
            metrics={
                "sharpe": -10.0,
                "combined_score": 0.0,
                "error": str(e),
            },
            artifacts={
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Quick validation."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "generate_signals"):
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0},
                artifacts={"error": "Missing generate_signals function"},
            )

        # Quick test
        data = generate_market_data(50, seed=42)
        signals = program.generate_signals(data["prices"], data["features"])

        if not isinstance(signals, list) or len(signals) != 50:
            return EvaluationResult(
                metrics={"runs_successfully": 0.5, "combined_score": 0.1},
                artifacts={"error": "Invalid signal output"},
            )

        return EvaluationResult(
            metrics={"runs_successfully": 1.0, "combined_score": 0.5},
            artifacts={"stage": "1", "status": "passed"},
        )

    except Exception as e:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": 0.0}, artifacts={"error": str(e)}
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation."""
    return evaluate(program_path)
