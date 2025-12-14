"""
Stock Prediction ML Evaluator for OpenEvolve

Trains evolved PyTorch models on DGX Spark (GB10) using Alpaca market data.
Returns fitness based on prediction accuracy, Sharpe ratio, and information coefficient.

Hidden variables for Heisenberg Engine to discover:
- Market regime (bull/bear/sideways)
- Volatility clustering (GARCH effects)
- Sector correlations
- Time-of-day effects
- Earnings/news impact
"""

import ast
import hashlib
import json
import os
import pickle
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# Load environment variables
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DGX_HOST = os.getenv("DGX_HOST", "trevor-morgan@sparky.local")
DGX_WORK_DIR = os.getenv("DGX_WORK_DIR", "/tmp/openevolve_stocks")
DGX_PYTHON = os.getenv(
    "DGX_PYTHON", "/home/trevor-morgan/repos/dgx-local-trainer/.venv/bin/python3"
)

# Cache directory for market data
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Symbols to train on
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
SEQUENCE_LENGTH = 60  # 60 days of history
TRAIN_DAYS = 504  # ~2 years of trading days


def _parse_problem_context(problem_context: str | None) -> dict:
    """Parse ProblemSpace.to_prompt_context into a structured dict.

    This keeps coevolution non-fictional by allowing the evaluator to respond
    to evolved objectives/constraints in a minimally structured way.
    """
    if not problem_context:
        return {
            "difficulty_level": None,
            "generation": None,
            "constraints": [],
            "objectives": [],
        }

    difficulty_level = None
    generation = None
    constraints: list[str] = []
    objectives: list[str] = []

    try:
        m = re.search(r"^Difficulty Level:\\s*([0-9]+(?:\\.[0-9]+)?)\\s*$", problem_context, re.M)
        if m:
            difficulty_level = float(m.group(1))
    except Exception:
        difficulty_level = None

    try:
        m = re.search(r"^Problem Generation:\\s*(\\d+)\\s*$", problem_context, re.M)
        if m:
            generation = int(m.group(1))
    except Exception:
        generation = None

    try:
        lines = problem_context.splitlines()
        mode = None
        for raw in lines:
            line = raw.strip()
            if not line:
                if mode in ("constraints", "objectives"):
                    mode = None
                continue
            if line.startswith("### Constraints"):
                mode = "constraints"
                continue
            if line.startswith("### Objectives"):
                mode = "objectives"
                continue

            if mode == "constraints" and line.startswith("- "):
                constraints.append(line[2:].strip())
            elif mode == "objectives":
                m = re.match(r"^\\d+\\.\\s+(.*)$", line)
                if m:
                    objectives.append(m.group(1).strip())
    except Exception:
        pass

    return {
        "difficulty_level": difficulty_level,
        "generation": generation,
        "constraints": constraints,
        "objectives": objectives,
    }


def _weights_from_objectives(objectives: list[str]) -> dict[str, float] | None:
    """Map natural-language objectives onto metric weights."""
    if not objectives:
        return None

    scores = {"direction": 0.0, "sharpe": 0.0, "ic": 0.0, "mse": 0.0}
    for idx, obj in enumerate(objectives):
        if not obj:
            continue
        priority = 1.0 / float(idx + 1)
        s = obj.lower()

        if any(k in s for k in ("direction", "accuracy", "hit rate", "classification")):
            scores["direction"] += priority
        if any(k in s for k in ("sharpe", "risk", "risk-adjusted", "drawdown", "volatility")):
            scores["sharpe"] += priority
        if any(k in s for k in ("ic", "information coefficient", "correlation")):
            scores["ic"] += priority
        if any(k in s for k in ("mse", "mae", "loss", "error", "prediction error")):
            scores["mse"] += priority

    total = sum(scores.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in scores.items()}


def _compute_combined_score(metrics: dict, weights: dict[str, float] | None = None) -> float:
    """Compute the combined fitness score from raw metrics."""
    direction_acc = metrics.get("direction_accuracy", 0.5)
    sharpe = metrics.get("sharpe", 0.0)
    ic = metrics.get("ic", 0.0)
    mse = metrics.get("mse", 1.0)

    # Normalize components to [0, 1] range
    direction_score = float(direction_acc)  # Already 0-1
    sharpe_score = min(max((float(sharpe) + 1) / 3, 0), 1)  # Sharpe -1 to 2 -> 0 to 1
    ic_score = min(max((float(ic) + 0.5) / 1.0, 0), 1)  # IC -0.5 to 0.5 -> 0 to 1
    mse_score = max(0, 1 - float(mse) * 100)  # Lower MSE is better

    base_weights = {"direction": 0.35, "sharpe": 0.30, "ic": 0.20, "mse": 0.15}
    if weights:
        try:
            merged = {**base_weights, **weights}
            denom = sum(float(v) for v in merged.values())
            if denom > 0:
                base_weights = {k: float(v) / denom for k, v in merged.items()}
        except Exception:
            pass

    combined_score = (
        base_weights["direction"] * direction_score
        + base_weights["sharpe"] * sharpe_score
        + base_weights["ic"] * ic_score
        + base_weights["mse"] * mse_score
    )

    metrics["combined_score"] = combined_score
    metrics["direction_score"] = direction_score
    metrics["sharpe_score"] = sharpe_score
    metrics["ic_score"] = ic_score
    metrics["mse_score"] = mse_score
    metrics["combined_weights_direction"] = float(base_weights["direction"])
    metrics["combined_weights_sharpe"] = float(base_weights["sharpe"])
    metrics["combined_weights_ic"] = float(base_weights["ic"])
    metrics["combined_weights_mse"] = float(base_weights["mse"])

    return combined_score


def _maybe_perturb_data_for_problem(
    data: dict,
    problem_context: str | None,
) -> tuple[dict, dict]:
    """Apply deterministic perturbations based on problem context (robustness axis)."""
    ctx = _parse_problem_context(problem_context)
    difficulty = ctx.get("difficulty_level") or 1.0
    objectives = ctx.get("objectives") or []

    # Deterministic seed per problem variant.
    seed = 0
    try:
        if problem_context:
            seed = int(hashlib.md5(problem_context.encode("utf-8")).hexdigest()[:8], 16)
    except Exception:
        seed = 0

    # Multiplicative noise scale (prices/volumes) increases with difficulty,
    # and is boosted when objectives mention robustness/adversarial.
    noise_scale = 0.0
    try:
        noise_scale = max(0.0, 0.002 * max(0.0, float(difficulty) - 1.0))  # 0..+
        noise_scale = min(noise_scale, 0.01)  # cap at 1%
    except Exception:
        noise_scale = 0.0

    if any("robust" in (o or "").lower() or "adversar" in (o or "").lower() for o in objectives):
        noise_scale = max(noise_scale, 0.005)

    artifacts: dict = {
        "problem_difficulty": float(difficulty),
        "problem_generation": ctx.get("generation"),
        "problem_objectives": objectives,
        "problem_constraints": ctx.get("constraints") or [],
        "input_noise_scale": float(noise_scale),
        "input_noise_seed": int(seed),
    }

    if noise_scale <= 0.0:
        return data, artifacts

    try:
        rng = np.random.default_rng(seed)
        prices = np.array(data["prices"], copy=True)
        volumes = np.array(data["volumes"], copy=True)

        prices = prices * (1.0 + rng.normal(0.0, noise_scale, size=prices.shape))
        volumes = volumes * (1.0 + rng.normal(0.0, noise_scale, size=volumes.shape))
        volumes = np.clip(volumes, 0.0, None)

        new_data = dict(data)
        new_data["prices"] = prices
        new_data["volumes"] = volumes
        return new_data, artifacts
    except Exception:
        return data, artifacts


def _run_remote_training(
    program_code: str,
    data: dict,
    artifacts: dict,
    timeout: int,
) -> tuple[dict | None, dict]:
    """Run training on DGX and return metrics plus updated artifacts."""
    # Setup remote directory
    run_remote(f"mkdir -p {DGX_WORK_DIR}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write evolved model
        model_path = os.path.join(tmpdir, "model.py")
        with open(model_path, "w") as f:
            f.write(program_code)

        # Write training script
        train_script = create_training_script(data)
        train_path = os.path.join(tmpdir, "train.py")
        with open(train_path, "w") as f:
            f.write(train_script)

        # Copy files to DGX
        if not copy_to_remote(model_path, f"{DGX_WORK_DIR}/model.py"):
            artifacts["error"] = "Failed to copy model to DGX"
            return None, artifacts

        if not copy_to_remote(train_path, f"{DGX_WORK_DIR}/train.py"):
            artifacts["error"] = "Failed to copy training script to DGX"
            return None, artifacts

        # Run training on DGX
        ret, stdout, stderr = run_remote(
            f"cd {DGX_WORK_DIR} && {DGX_PYTHON} train.py",
            timeout=timeout,
        )

        artifacts["stdout"] = stdout[:5000]
        artifacts["stderr"] = stderr[:2000]

        if "RESULTS_START" in stdout:
            try:
                results_str = stdout.split("RESULTS_START")[1].split("RESULTS_END")[0].strip()
                metrics = json.loads(results_str)
                return metrics, artifacts
            except (json.JSONDecodeError, IndexError) as e:
                artifacts["error"] = f"Failed to parse results: {e}"
                artifacts["raw_stdout"] = stdout[:1000]
                return None, artifacts

        if "ERROR_START" in stdout:
            try:
                error_str = stdout.split("ERROR_START")[1].split("ERROR_END")[0].strip()
                error_info = json.loads(error_str)
                artifacts["error"] = error_info.get("error", "Unknown error")
                artifacts["traceback"] = error_info.get("traceback", "")
                return None, artifacts
            except Exception:
                artifacts["error"] = "Training failed"
                artifacts["raw_stdout"] = stdout[:1000]
                return None, artifacts

        artifacts["error"] = "No results found"
        artifacts["raw_stdout"] = stdout[:1000]
        return None, artifacts


def evaluate_stage1(program_path: str, problem_context: str | None = None) -> dict:
    """
    Stage 1 cascade evaluation: fast local viability checks.

    Filters out syntax errors or missing required entrypoints before remote GPU training.
    """
    artifacts: dict = {"stage": 1}
    metrics: dict = {}

    try:
        with open(program_path) as f:
            program_code = f.read()

        # Syntax check only (no imports executed)
        compile(program_code, program_path, "exec")
        artifacts["syntax_ok"] = True
    except Exception as e:
        artifacts["syntax_ok"] = False
        artifacts["error"] = str(e)
        metrics["combined_score"] = 0.0
        metrics["stage1_passed"] = 0.0
        return {"score": 0.0, "metrics": metrics, "artifacts": artifacts}

    # AST checks for required functions/classes
    tree = ast.parse(program_code)
    has_run_training = any(
        isinstance(n, ast.FunctionDef) and n.name == "run_training" for n in ast.walk(tree)
    )
    has_create_model = any(
        isinstance(n, ast.FunctionDef) and n.name == "create_model" for n in ast.walk(tree)
    )
    has_predictor_class = any(
        isinstance(n, ast.ClassDef) and n.name == "StockPredictor" for n in ast.walk(tree)
    )

    missing: list[str] = []
    viability = 0.0
    if has_run_training:
        viability += 0.5
    else:
        missing.append("run_training")
    if has_create_model or has_predictor_class:
        viability += 0.5
    else:
        missing.append("create_model/StockPredictor")

    if missing:
        artifacts["missing"] = missing

    if problem_context:
        artifacts["problem_context"] = problem_context[:500]

    metrics["combined_score"] = viability
    metrics["stage1_passed"] = 1.0 if viability >= 0.3 else 0.0

    return {"score": viability, "metrics": metrics, "artifacts": artifacts}


def evaluate_stage2(program_path: str, problem_context: str | None = None) -> dict:
    """
    Stage 2 cascade evaluation: small‑data remote training.

    Uses fewer symbols/days and a capped sample count to cheaply estimate fitness.
    """
    artifacts: dict = {"stage": 2}
    metrics: dict = {}

    with open(program_path) as f:
        program_code = f.read()

    try:
        # Smaller slice for quicker training, scaled by problem difficulty.
        ctx = _parse_problem_context(problem_context)
        difficulty = ctx.get("difficulty_level") or 1.0
        days = int(np.clip(180 + (float(difficulty) - 1.0) * 60.0, 120, 240))

        data = fetch_stock_data(SYMBOLS[:3], days=days)
        data, problem_artifacts = _maybe_perturb_data_for_problem(data, problem_context)
        max_samples = min(len(data["targets"]), 2000)
        data_small = {
            "prices": data["prices"][:max_samples],
            "volumes": data["volumes"][:max_samples],
            "targets": data["targets"][:max_samples],
            "symbols": data.get("symbols", [])[:max_samples],
        }
        artifacts["num_samples"] = max_samples
        artifacts["symbols"] = SYMBOLS[:3]
        artifacts["days"] = days
        artifacts.update(problem_artifacts)
    except Exception as e:
        artifacts["error"] = f"Failed to fetch reduced data: {e}"
        metrics["combined_score"] = 0.0
        metrics["stage2_passed"] = 0.0
        return {"score": 0.0, "metrics": metrics, "artifacts": artifacts}

    metrics_remote, artifacts = _run_remote_training(
        program_code=program_code,
        data=data_small,
        artifacts=artifacts,
        timeout=180,
    )

    if metrics_remote is None:
        metrics["combined_score"] = 0.0
        metrics["stage2_passed"] = 0.0
        return {"score": 0.0, "metrics": metrics, "artifacts": artifacts}

    objective_weights = _weights_from_objectives(
        _parse_problem_context(problem_context).get("objectives") or []
    )
    combined_score = _compute_combined_score(metrics_remote, weights=objective_weights)

    # If the evolved problem emphasizes efficiency, softly penalize long training.
    try:
        objectives = _parse_problem_context(problem_context).get("objectives") or []
        if any(
            "efficien" in (o or "").lower() or "compute" in (o or "").lower() for o in objectives
        ):
            epochs = float(metrics_remote.get("epochs_trained", 0.0) or 0.0)
            penalty = float(np.clip(0.002 * epochs, 0.0, 0.1))
            metrics_remote["efficiency_penalty"] = penalty
            combined_score = float(np.clip(combined_score - penalty, 0.0, 1.0))
            metrics_remote["combined_score"] = combined_score
    except Exception:
        pass

    metrics_remote["stage2_passed"] = 1.0 if combined_score >= 0.5 else 0.0

    return {
        "score": combined_score,
        "metrics": metrics_remote,
        "artifacts": artifacts,
    }


def evaluate_stage3(program_path: str, problem_context: str | None = None) -> dict:
    """Stage 3 cascade evaluation: full evaluation."""
    return evaluate(program_path, problem_context=problem_context)


def get_alpaca_client():
    """Get Alpaca API client."""
    try:
        from alpaca.data import StockHistoricalDataClient

        return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    except ImportError:
        raise ImportError("Please install alpaca-py: pip install alpaca-py")


def fetch_stock_data(
    symbols: list[str],
    days: int = TRAIN_DAYS,
    use_cache: bool = True,
) -> dict:
    """
    Fetch historical stock data from Alpaca.

    Returns dict with prices, volumes, and targets for each symbol.
    """
    cache_key = hashlib.md5(f"{symbols}_{days}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"stock_data_{cache_key}.pkl"

    if use_cache and cache_file.exists():
        # Check if cache is fresh (less than 1 day old)
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age < 86400:  # 24 hours
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        raise ImportError("Please install alpaca-py: pip install alpaca-py")

    client = get_alpaca_client()

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(days * 1.5))  # Extra buffer for weekends

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )

    bars = client.get_stock_bars(request)

    all_data = {"prices": [], "volumes": [], "targets": [], "symbols": []}

    for symbol in symbols:
        if symbol not in bars.data:
            continue

        symbol_bars = bars.data[symbol]
        if len(symbol_bars) < SEQUENCE_LENGTH + 1:
            continue

        closes = np.array([bar.close for bar in symbol_bars])
        volumes = np.array([bar.volume for bar in symbol_bars])

        # Create sequences
        for i in range(SEQUENCE_LENGTH, len(closes) - 1):
            price_seq = closes[i - SEQUENCE_LENGTH : i]
            vol_seq = volumes[i - SEQUENCE_LENGTH : i]

            # Target: next day return
            target = (closes[i + 1] - closes[i]) / closes[i]

            all_data["prices"].append(price_seq)
            all_data["volumes"].append(vol_seq)
            all_data["targets"].append(target)
            all_data["symbols"].append(symbol)

    # Convert to numpy
    all_data["prices"] = np.array(all_data["prices"])
    all_data["volumes"] = np.array(all_data["volumes"])
    all_data["targets"] = np.array(all_data["targets"])

    # Cache the data
    with open(cache_file, "wb") as f:
        pickle.dump(all_data, f)

    return all_data


def run_remote(cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run command on remote DGX via SSH."""
    full_cmd = f'ssh {DGX_HOST} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def copy_to_remote(local_path: str, remote_path: str) -> bool:
    """Copy file to remote DGX via SCP."""
    cmd = f"scp {local_path} {DGX_HOST}:{remote_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
    return result.returncode == 0


def create_training_script(data: dict) -> str:
    """Create the training script to run on DGX."""
    # Serialize data to base64 for embedding in script
    import base64

    data_bytes = pickle.dumps(
        {
            "prices": data["prices"],
            "volumes": data["volumes"],
            "targets": data["targets"],
        }
    )
    data_b64 = base64.b64encode(data_bytes).decode()

    return f'''#!/usr/bin/env python3
"""Training script for DGX execution."""
import sys
import json
import base64
import pickle
import traceback
import random
import numpy as np

# Deterministic seeding for stable cascade scores
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# Deserialize data
DATA_B64 = """{data_b64}"""
data = pickle.loads(base64.b64decode(DATA_B64))

prices = data["prices"]
volumes = data["volumes"]
targets = data["targets"]

# Import the evolved model
sys.path.insert(0, "{DGX_WORK_DIR}")
try:
    from model import run_training

    # Run training on GPU
    metrics = run_training(
        prices=prices,
        volumes=volumes,
        targets=targets,
        train_ratio=0.8,
        device="cuda",
    )

    # Output results as JSON
    print("RESULTS_START")
    print(json.dumps(metrics))
    print("RESULTS_END")

except Exception as e:
    print("ERROR_START")
    print(json.dumps({{
        "error": str(e),
        "traceback": traceback.format_exc(),
    }}))
    print("ERROR_END")
'''


def evaluate(program_path: str, problem_context: str | None = None) -> dict:
    """
    Evaluate an evolved stock prediction model.

    1. Fetch stock data from Alpaca
    2. Copy evolved model to DGX
    3. Run training on GPU
    4. Return fitness metrics

    Returns:
        dict with 'score', 'metrics', and 'artifacts'
    """
    artifacts: dict = {}
    metrics: dict = {}

    # Read the evolved program
    with open(program_path) as f:
        program_code = f.read()

    artifacts["program_length"] = len(program_code)

    # Fetch market data
    try:
        data = fetch_stock_data(SYMBOLS, days=TRAIN_DAYS)
        data, problem_artifacts = _maybe_perturb_data_for_problem(data, problem_context)
        artifacts["num_samples"] = len(data["targets"])
        artifacts["symbols"] = SYMBOLS
        artifacts.update(problem_artifacts)
    except Exception as e:
        return {
            "score": 0.0,
            "metrics": {"error": f"Failed to fetch data: {e}"},
            "artifacts": artifacts,
        }

    metrics_remote, artifacts = _run_remote_training(
        program_code=program_code,
        data=data,
        artifacts=artifacts,
        timeout=300,  # 5 minutes max
    )

    if metrics_remote is None:
        return {
            "score": 0.0,
            "metrics": {"error": artifacts.get("error", "Training failed")},
            "artifacts": artifacts,
        }

    metrics = metrics_remote

    # Calculate combined fitness score
    objective_weights = _weights_from_objectives(
        _parse_problem_context(problem_context).get("objectives") or []
    )
    combined_score = _compute_combined_score(metrics, weights=objective_weights)

    # If the evolved problem emphasizes efficiency, softly penalize long training.
    try:
        objectives = _parse_problem_context(problem_context).get("objectives") or []
        if any(
            "efficien" in (o or "").lower() or "compute" in (o or "").lower() for o in objectives
        ):
            epochs = float(metrics.get("epochs_trained", 0.0) or 0.0)
            penalty = float(np.clip(0.002 * epochs, 0.0, 0.1))
            metrics["efficiency_penalty"] = penalty
            combined_score = float(np.clip(combined_score - penalty, 0.0, 1.0))
            metrics["combined_score"] = combined_score
    except Exception:
        pass

    return {
        "score": combined_score,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def extract_phenotype(
    code: str,
    metrics: dict,
    artifacts: dict | None = None,
) -> dict:
    """
    Task-specific phenotype extractor for ML stock prediction.

    Returned values are used by Discovery's EpistemicArchive and can be mirrored into
    program metrics for MAP-Elites feature dimensions.

    Args:
        code: Program source (unused for now).
        metrics: Evaluation metrics from cascade/full eval.
        artifacts: Optional artifacts (timeout/error info).

    Returns:
        Dict of phenotype attributes (numeric or bool).
    """
    # Training cost: fewer epochs to converge is "better" and a useful QD axis.
    epochs = float(metrics.get("epochs_trained", 0.0) or 0.0)
    training_cost = min(int(epochs / 5.0), 9)  # 0..9 bin

    # Robustness: penalize timeouts/crashes, reward passing deeper stages.
    robustness = 0
    if artifacts and artifacts.get("timeout"):
        robustness = 0
    else:
        try:
            finite = True
            for k in ("mse", "mae", "sharpe", "ic"):
                v = metrics.get(k)
                if v is not None and not np.isfinite(float(v)):
                    finite = False
            if finite:
                if float(metrics.get("stage2_passed", 0.0) or 0.0) >= 1.0:
                    robustness = 2
                elif float(metrics.get("stage1_passed", 0.0) or 0.0) >= 1.0:
                    robustness = 1
        except Exception:
            robustness = 0

    # Risk skill axis based on Sharpe score if present.
    sharpe_score = float(metrics.get("sharpe_score", 0.0) or 0.0)
    risk_skill = int(np.clip(sharpe_score * 10.0, 0, 9))

    return {
        "training_cost": float(training_cost),
        "robustness": float(robustness),
        "risk_skill": float(risk_skill),
    }


# Test locally
if __name__ == "__main__":
    print("Testing stock prediction evaluator...")

    # Test data fetch
    print("\n1. Fetching stock data from Alpaca...")
    try:
        data = fetch_stock_data(SYMBOLS[:3], days=100)  # Smaller test
        print(f"   Fetched {len(data['targets'])} samples")
        print(f"   Price shape: {data['prices'].shape}")
        print(f"   Target range: [{data['targets'].min():.4f}, {data['targets'].max():.4f}]")
    except Exception as e:
        print(f"   Error: {e}")
        print("   (This is OK if Alpaca credentials aren't set up)")

    # Test with initial program
    print("\n2. Testing evaluation with initial program...")
    example_dir = Path(__file__).parent
    initial_program = example_dir / "initial_program.py"

    if initial_program.exists():
        result = evaluate(str(initial_program))
        print(f"   Score: {result['score']:.4f}")
        print(f"   Metrics: {json.dumps(result['metrics'], indent=2)}")
    else:
        print("   initial_program.py not found")
