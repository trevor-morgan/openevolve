"""
Initial trading strategy program.

Implements a basic momentum strategy that will be evolved to discover
better alpha signals using AFML techniques.
"""


def generate_signals(prices: list[float], features: dict[str, list[float]]) -> list[float]:
    """
    Generate trading signals from price and feature data.

    Args:
        prices: Historical price series
        features: Dictionary of feature arrays (momentum, volatility, rsi, etc.)

    Returns:
        List of signals: positive = long, negative = short, 0 = flat
    """
    # EVOLVE-BLOCK-START
    n = len(prices)
    signals = [0.0] * n

    # Simple momentum strategy (placeholder)
    momentum = features.get("momentum_20", [0.0] * n)
    rsi = features.get("rsi_14", [50.0] * n)

    for i in range(20, n):
        # Long if momentum is positive and RSI not overbought
        if momentum[i] > 0.02 and rsi[i] < 70:
            signals[i] = 1.0
        # Short if momentum is negative and RSI not oversold
        elif momentum[i] < -0.02 and rsi[i] > 30:
            signals[i] = -1.0
        else:
            signals[i] = 0.0

    return signals
    # EVOLVE-BLOCK-END
