"""
Stock Prediction Neural Network - Initial Architecture

This model predicts next-day returns using historical price/volume data.
OpenEvolve will evolve the architecture, feature engineering, and training logic.

Hidden variables the Heisenberg Engine might discover:
- Market regime (bull/bear/sideways)
- Volatility clustering
- Sector correlations
- Sentiment signals
- Liquidity patterns
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# EVOLVE-BLOCK-START: feature_engineering
def engineer_features(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Engineer features from raw price and volume data.

    Args:
        prices: Array of shape (num_samples, sequence_length) - closing prices
        volumes: Array of shape (num_samples, sequence_length) - volumes

    Returns:
        features: Array of shape (num_samples, sequence_length, num_features)
    """
    _num_samples, seq_len = prices.shape

    # Basic returns
    returns = np.zeros_like(prices)
    returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / (prices[:, :-1] + 1e-8)

    # Log returns
    log_returns = np.zeros_like(prices)
    log_returns[:, 1:] = np.log(prices[:, 1:] / (prices[:, :-1] + 1e-8) + 1e-8)

    # Volatility (rolling std of returns)
    volatility = np.zeros_like(prices)
    window = 5
    for i in range(window, seq_len):
        volatility[:, i] = np.std(returns[:, i - window : i], axis=1)

    # Volume change
    vol_change = np.zeros_like(volumes)
    vol_change[:, 1:] = (volumes[:, 1:] - volumes[:, :-1]) / (volumes[:, :-1] + 1e-8)

    # Normalized volume
    vol_mean = np.mean(volumes, axis=1, keepdims=True)
    vol_std = np.std(volumes, axis=1, keepdims=True) + 1e-8
    vol_normalized = (volumes - vol_mean) / vol_std

    # Simple moving averages
    sma_5 = np.zeros_like(prices)
    sma_20 = np.zeros_like(prices)
    for i in range(5, seq_len):
        sma_5[:, i] = np.mean(prices[:, i - 5 : i], axis=1)
    for i in range(20, seq_len):
        sma_20[:, i] = np.mean(prices[:, i - 20 : i], axis=1)

    # Price relative to SMAs
    price_to_sma5 = (prices - sma_5) / (sma_5 + 1e-8)
    price_to_sma20 = (prices - sma_20) / (sma_20 + 1e-8)

    # Stack features
    features = np.stack(
        [
            returns,
            log_returns,
            volatility,
            vol_change,
            vol_normalized,
            price_to_sma5,
            price_to_sma20,
        ],
        axis=-1,
    )

    return features


# EVOLVE-BLOCK-END: feature_engineering


# EVOLVE-BLOCK-START: model_architecture
class StockPredictor(nn.Module):
    """
    Neural network for stock return prediction.

    Architecture can be evolved: layer sizes, attention, normalization, etc.
    """

    def __init__(
        self,
        input_features: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Linear(input_features, hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_size)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)

        Returns:
            predictions: Tensor of shape (batch, 1) - predicted return
        """
        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)

        # LSTM
        lstm_out, (_h_n, _c_n) = self.lstm(x)

        # Attention over sequence
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attn_norm(lstm_out + attn_out)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)

        # Output MLP
        out = self.dropout(F.gelu(self.fc1(last_hidden)))
        out = self.fc2(out)

        return out


# EVOLVE-BLOCK-END: model_architecture


# EVOLVE-BLOCK-START: loss_function
def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
) -> torch.Tensor:
    """
    Compute training loss.

    Can evolve to include:
    - Different loss functions (MSE, Huber, quantile)
    - Regularization terms
    - Directional accuracy bonuses
    - Sharpe-aware losses
    """
    # Base MSE loss
    mse_loss = F.mse_loss(predictions.squeeze(), targets)

    # Directional accuracy bonus (predict sign correctly)
    pred_sign = torch.sign(predictions.squeeze())
    target_sign = torch.sign(targets)
    direction_accuracy = (pred_sign == target_sign).float().mean()
    direction_bonus = -0.1 * direction_accuracy  # Negative because we minimize

    # L2 regularization
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    l2_loss = 1e-5 * l2_reg

    total_loss = mse_loss + direction_bonus + l2_loss

    return total_loss


# EVOLVE-BLOCK-END: loss_function


# EVOLVE-BLOCK-START: training_loop
def train_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """
    Train the model and return metrics.

    Can evolve:
    - Learning rate schedules
    - Optimizers
    - Early stopping criteria
    - Data augmentation
    """
    model = model.to(device)

    # Convert to tensors
    X_train = torch.FloatTensor(train_features).to(device)
    y_train = torch.FloatTensor(train_targets).to(device)
    X_val = torch.FloatTensor(val_features).to(device)
    y_val = torch.FloatTensor(val_targets).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train))
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = compute_loss(predictions, batch_y, model)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred.squeeze(), y_val).item()

        train_losses.append(epoch_loss / num_batches)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).squeeze().cpu().numpy()
        val_true = y_val.cpu().numpy()

    # Calculate metrics
    mse = np.mean((val_pred - val_true) ** 2)
    mae = np.mean(np.abs(val_pred - val_true))

    # Directional accuracy
    direction_correct = np.mean(np.sign(val_pred) == np.sign(val_true))

    # Sharpe ratio of predictions (if we traded based on them)
    returns_if_traded = val_pred * val_true  # Positive if direction correct
    sharpe = np.mean(returns_if_traded) / (np.std(returns_if_traded) + 1e-8) * np.sqrt(252)

    # Information coefficient (correlation)
    ic = np.corrcoef(val_pred, val_true)[0, 1] if len(val_pred) > 1 else 0.0

    return {
        "mse": float(mse),
        "mae": float(mae),
        "direction_accuracy": float(direction_correct),
        "sharpe": float(sharpe),
        "ic": float(ic),
        "epochs_trained": len(train_losses),
        "best_val_loss": float(best_val_loss),
    }


# EVOLVE-BLOCK-END: training_loop


# Entry point for evaluation
def create_model(**kwargs) -> nn.Module:
    """Create model instance with given parameters."""
    return StockPredictor(**kwargs)


def skeptic_entrypoint(
    prices: np.ndarray,
    volumes: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    device: str | None = None,
) -> dict:
    """
    Lightweight robustness check used by the Discovery skeptic.

    Runs a tiny training/eval pass on potentially perturbed data and raises on
    non‑finite metrics or shape inconsistencies. This function is not evolved.
    """
    import numpy as _np

    prices = _np.asarray(prices)
    volumes = _np.asarray(volumes)
    targets = _np.asarray(targets)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D, got shape {prices.shape}")
    if volumes.shape != prices.shape:
        raise ValueError(f"volumes shape {volumes.shape} != prices shape {prices.shape}")
    if targets.ndim != 1 or targets.shape[0] != prices.shape[0]:
        raise ValueError(f"targets length {targets.shape} != num_samples {prices.shape[0]}")

    # Sanitize NaNs/Infs and clip extremes
    prices = _np.where(_np.isfinite(prices), prices, _np.nan)
    volumes = _np.where(_np.isfinite(volumes), volumes, _np.nan)
    targets = _np.where(_np.isfinite(targets), targets, _np.nan)

    if _np.isnan(prices).any():
        prices = _np.nan_to_num(prices, nan=float(_np.nanmedian(prices)))
    if _np.isnan(volumes).any():
        volumes = _np.nan_to_num(volumes, nan=0.0)
    if _np.isnan(targets).any():
        targets = _np.nan_to_num(targets, nan=0.0)

    prices = _np.clip(prices, 1e-3, 1e6)
    volumes = _np.clip(volumes, 0.0, 1e9)
    targets = _np.clip(targets, -1.0, 1.0)

    max_samples = min(prices.shape[0], 64)
    prices = prices[:max_samples]
    volumes = volumes[:max_samples]
    targets = targets[:max_samples]

    features = engineer_features(prices, volumes)
    if features.shape[0] < 2:
        raise ValueError("not enough samples for robustness check")

    n_train = int(features.shape[0] * train_ratio)
    n_train = max(1, min(n_train, features.shape[0] - 1))

    train_features = features[:n_train]
    train_targets = targets[:n_train]
    val_features = features[n_train:]
    val_targets = targets[n_train:]

    # Choose model constructor
    if "create_model" in globals() and callable(create_model):
        model = create_model(
            input_features=features.shape[-1],
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            use_attention=True,
        )
    elif "StockPredictor" in globals():
        model = StockPredictor(
            input_features=features.shape[-1],
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            use_attention=True,
        )
    else:
        raise ValueError("No model constructor found")

    if device is None:
        # Skeptic checks should be lightweight and deterministic; default to CPU
        # to avoid CUDA init overhead and GPU nondeterminism on remote hosts.
        device = "cpu"

    metrics = train_model(
        model=model,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        device=device,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
    )

    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not _np.isfinite(v):
            raise ValueError(f"non-finite metric {k}={v}")

    return metrics


def run_training(
    prices: np.ndarray,
    volumes: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    device: str = "cuda",
) -> dict:
    """
    Full training pipeline.

    Args:
        prices: Shape (num_samples, sequence_length)
        volumes: Shape (num_samples, sequence_length)
        targets: Shape (num_samples,) - next day returns
        train_ratio: Train/val split ratio
        device: Device to train on

    Returns:
        Dictionary of metrics
    """
    # Engineer features
    features = engineer_features(prices, volumes)

    # Train/val split
    n_train = int(len(features) * train_ratio)

    train_features = features[:n_train]
    train_targets = targets[:n_train]
    val_features = features[n_train:]
    val_targets = targets[n_train:]

    # Create model
    model = create_model(
        input_features=features.shape[-1],
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        use_attention=True,
    )

    # Train
    metrics = train_model(
        model=model,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        device=device,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
    )

    return metrics


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    num_samples = 1000
    seq_len = 60

    prices = 100 + np.cumsum(np.random.randn(num_samples, seq_len) * 0.02, axis=1)
    volumes = np.abs(np.random.randn(num_samples, seq_len)) * 1e6
    targets = np.random.randn(num_samples) * 0.02

    metrics = run_training(prices, volumes, targets, device="cpu")
    print("Test metrics:", metrics)
