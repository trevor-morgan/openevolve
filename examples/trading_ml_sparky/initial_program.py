"""
Initial feature engineering program for trading ML.

This program defines how to compute evolved features from market data.
OpenEvolve will evolve the EVOLVE-BLOCK to discover better alpha signals.

The evolved features are combined with base features and used to train
XGBoost classifiers on sparky.local.
"""

import numpy as np
import pandas as pd


def compute_evolved_features(df: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute evolved features from market data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, returns
        base_features: Pre-computed features (momentum, volatility, RSI, etc.)

    Returns:
        DataFrame with new evolved features
    """
    # EVOLVE-BLOCK-START
    features = pd.DataFrame(index=df.index)

    # Simple placeholder features - LLM will evolve these
    # Use .get() for safety with missing columns
    mom_5 = base_features.get("mom_5", pd.Series(0, index=df.index))
    mom_10 = base_features.get("mom_10", pd.Series(0, index=df.index))
    mom_20 = base_features.get("mom_20", pd.Series(0, index=df.index))
    vol_10 = base_features.get("vol_10", pd.Series(0.01, index=df.index))

    # Cross-sectional momentum
    features["momentum_ratio"] = mom_5 / (mom_20 + 1e-10)

    # Volatility-adjusted momentum
    features["vol_adj_mom"] = mom_10 / (vol_10 + 1e-10)

    # Price acceleration
    features["price_accel"] = mom_5 - mom_5.shift(5)

    return features
    # EVOLVE-BLOCK-END
