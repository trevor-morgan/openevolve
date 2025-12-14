"""
Evaluator that trains real ML models on sparky.local.

This evaluator:
1. Receives evolved feature engineering code from OpenEvolve
2. SSHs to sparky.local and trains XGBoost models
3. Returns real walk-forward Sharpe ratios

The evolved code defines feature engineering - the model architecture stays fixed.
"""

import importlib.util
import json
import subprocess
import tempfile
import traceback
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

# Sparky configuration
SPARKY_HOST = "trevor-morgan@sparky.local"
SPARKY_VENV = "source ~/ml_venv/bin/activate"
SPARKY_DATA_DIR = "~/trading_data"  # Where market data lives on sparky

# Training script template that runs on sparky
TRAIN_SCRIPT_TEMPLATE = '''
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Feature engineering code (evolved by OpenEvolve)
{feature_code}

def generate_synthetic_data(n_days=1000, seed=42):
    """Generate realistic synthetic market data."""
    np.random.seed(seed)

    # Price series with trend, mean reversion, and volatility clustering
    returns = np.zeros(n_days)
    volatility = 0.02

    for i in range(1, n_days):
        # GARCH-like volatility
        volatility = 0.01 + 0.9 * volatility + 0.1 * abs(returns[i-1])
        volatility = min(0.05, max(0.005, volatility))

        # Mean-reverting returns
        mean_reversion = -0.1 * returns[i-1] if i > 0 else 0
        returns[i] = mean_reversion + np.random.normal(0, volatility)

    prices = 100 * np.exp(np.cumsum(returns))

    # Create DataFrame with OHLCV
    df = pd.DataFrame({{
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_days),
    }})
    df['returns'] = df['close'].pct_change()

    return df

def compute_base_features(df):
    """Compute standard technical features."""
    features = pd.DataFrame(index=df.index)

    # Momentum
    for period in [5, 10, 20, 60]:
        features[f'mom_{{period}}'] = df['close'].pct_change(period)

    # Volatility
    for period in [10, 20, 60]:
        features[f'vol_{{period}}'] = df['returns'].rolling(period).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Moving average crossovers
    features['ma_cross_10_50'] = (df['close'].rolling(10).mean() / df['close'].rolling(50).mean()) - 1
    features['ma_cross_20_100'] = (df['close'].rolling(20).mean() / df['close'].rolling(100).mean()) - 1

    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Price position
    features['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)

    return features

def create_labels(df, horizon=5, threshold=0.01):
    """Create classification labels based on future returns."""
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    labels = (future_returns > threshold).astype(int)
    return labels

def walk_forward_backtest(features, labels, prices, n_splits=5):
    """Walk-forward validation with trading simulation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_returns = []

    for train_idx, test_idx in tscv.split(features):
        X_train = features.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        X_test = features.iloc[test_idx]

        # Train XGBoost
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Generate predictions
        probs = model.predict_proba(X_test)[:, 1]
        positions = (probs > 0.55).astype(float) - (probs < 0.45).astype(float)

        # Calculate returns
        test_returns = prices.iloc[test_idx].pct_change().fillna(0)
        strategy_returns = positions * test_returns.shift(-1).fillna(0)
        all_returns.extend(strategy_returns.values)

    return np.array(all_returns)

def main():
    # Generate data
    df = generate_synthetic_data(n_days=2000, seed=42)

    # Compute base features
    base_features = compute_base_features(df)

    # Apply evolved feature engineering
    try:
        evolved_features = compute_evolved_features(df, base_features)
        if evolved_features is not None and len(evolved_features.columns) > 0:
            features = pd.concat([base_features, evolved_features], axis=1)
        else:
            features = base_features
    except Exception as e:
        print(json.dumps({{"error": f"Feature engineering failed: {{str(e)}}"}}))
        return

    # Create labels
    labels = create_labels(df)

    # Drop NaN rows
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    prices = df['close'].loc[valid_idx]

    if len(features) < 500:
        print(json.dumps({{"error": "Not enough valid data points"}}))
        return

    # Run walk-forward backtest
    returns = walk_forward_backtest(features, labels, prices)

    # Compute metrics
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    total_return = np.sum(returns)
    max_drawdown = np.min(np.minimum.accumulate(np.cumsum(returns)) - np.cumsum(returns))
    win_rate = np.mean(returns > 0)

    result = {{
        "sharpe": float(sharpe),
        "total_return": float(total_return),
        "max_drawdown": float(abs(max_drawdown)),
        "win_rate": float(win_rate),
        "n_features": len(features.columns),
        "n_samples": len(features),
    }}

    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''


def run_on_sparky(feature_code: str, timeout: int = 120) -> dict:
    """
    Execute training on sparky.local via SSH.

    Args:
        feature_code: The evolved feature engineering function
        timeout: Max seconds for training

    Returns:
        Dict with metrics or error
    """
    # Create the full training script
    script = TRAIN_SCRIPT_TEMPLATE.format(feature_code=feature_code)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        temp_path = f.name

    try:
        # Copy script to sparky
        remote_script = "/tmp/openevolve_train.py"
        subprocess.run(
            ["scp", temp_path, f"{SPARKY_HOST}:{remote_script}"],
            capture_output=True,
            timeout=30,
        )

        # Run on sparky
        result = subprocess.run(
            ["ssh", SPARKY_HOST, f"{SPARKY_VENV} && python {remote_script}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return {"error": f"Training failed: {result.stderr[:500]}"}

        # Parse JSON output
        output = result.stdout.strip()
        if not output:
            return {"error": "No output from training"}

        return json.loads(output)

    except subprocess.TimeoutExpired:
        return {"error": "Training timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON output: {e}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        Path(temp_path).unlink(missing_ok=True)


def evaluate(program_path: str) -> EvaluationResult:
    """
    Full evaluation - trains model on sparky.local.
    """
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "compute_evolved_features"):
            return EvaluationResult(
                metrics={
                    "sharpe": -10.0,
                    "combined_score": 0.0,
                    "error": "Missing compute_evolved_features function",
                },
                artifacts={"error_type": "MissingFunction"},
            )

        # Read the source code
        with open(program_path) as f:
            source = f.read()

        # Extract the entire compute_evolved_features function
        # We need the full function definition, not just the EVOLVE-BLOCK
        import re

        match = re.search(
            r"(def compute_evolved_features\(.*?\):.*?)(?=\ndef |\Z)", source, re.DOTALL
        )
        if match:
            feature_code = match.group(1)
        else:
            feature_code = source

        # Run training on sparky
        result = run_on_sparky(feature_code, timeout=180)

        if "error" in result:
            return EvaluationResult(
                metrics={
                    "sharpe": -5.0,
                    "combined_score": 0.1,
                    "error": result["error"],
                },
                artifacts={"error_type": "TrainingError"},
            )

        # Compute combined score
        sharpe = result.get("sharpe", -10.0)
        max_dd = result.get("max_drawdown", 1.0)
        win_rate = result.get("win_rate", 0.5)

        # Score: prioritize Sharpe, penalize drawdown
        sharpe_score = min(1.0, max(0.0, (sharpe + 1) / 3))  # Maps [-1, 2] to [0, 1]
        dd_penalty = max(0.0, 1.0 - max_dd * 5)  # 20% DD = 0 score
        wr_bonus = (win_rate - 0.5) * 0.5  # Bonus for >50% win rate

        combined_score = 0.6 * sharpe_score + 0.3 * dd_penalty + 0.1 * (0.5 + wr_bonus)

        # Bonus for positive Sharpe
        if sharpe > 0:
            combined_score *= 1.2
        if sharpe > 1.0:
            combined_score *= 1.2

        return EvaluationResult(
            metrics={
                "sharpe": round(sharpe, 4),
                "total_return": round(result.get("total_return", 0), 4),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 4),
                "combined_score": round(combined_score, 4),
                "n_features": result.get("n_features", 0),
            },
            artifacts={
                "n_samples": result.get("n_samples", 0),
                "trained_on": "sparky.local",
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
    """Quick local validation - doesn't use sparky."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "compute_evolved_features"):
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0},
                artifacts={"error": "Missing compute_evolved_features function"},
            )

        # Quick test with dummy data
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(
            {
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 101,
                "low": np.random.randn(100) + 99,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "returns": np.random.randn(100) * 0.02,
            }
        )

        base_features = pd.DataFrame(
            {
                "mom_5": np.random.randn(100),
                "vol_10": np.random.randn(100),
            },
            index=df.index,
        )

        result = program.compute_evolved_features(df, base_features)

        if result is None or not isinstance(result, pd.DataFrame):
            return EvaluationResult(
                metrics={"runs_successfully": 0.5, "combined_score": 0.1},
                artifacts={"error": "Invalid output type"},
            )

        return EvaluationResult(
            metrics={"runs_successfully": 1.0, "combined_score": 0.3},
            artifacts={"stage": "1", "status": "passed", "n_features": len(result.columns)},
        )

    except Exception as e:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": 0.0}, artifacts={"error": str(e)}
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation on sparky."""
    return evaluate(program_path)
