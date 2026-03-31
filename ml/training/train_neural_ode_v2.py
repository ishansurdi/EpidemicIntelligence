import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import pandas as pd
import numpy as np

from ml.models.neural_ode_v2 import NeuralODEModel, NeuralODEConfig

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"

MAX_SAMPLES = 20_000


def _prepare_training_data(frame: pd.DataFrame, seq_len: int = 14, max_samples: int = MAX_SAMPLES) -> tuple:
    """Prepare sliding windows; subsample to max_samples for tractable ODE training."""
    frame = frame.sort_values(["country", "date"])

    X_context, y_targets = [], []
    for _, group in frame.groupby("country"):
        group = group.reset_index(drop=True)
        if len(group) < seq_len + 8:
            continue
        for i in range(seq_len, len(group) - 7):
            row = group.iloc[i]
            context = np.array(
                [
                    float(np.clip(row.get("people_fully_vaccinated_per_hundred", 0) / 100.0, 0, 1)),
                    float(np.clip((row.get("mobility_index", 0.0) + 100) / 200.0, 0, 1)),
                    float(np.clip(row.get("stringency_index", 0) / 100.0, 0, 1)),
                    float(np.clip(row.get("case_acceleration", 0) / 1000.0, -1, 1)),
                ],
                dtype=np.float32,
            )
            target = float(max(group.iloc[i + 7].get("daily_new_cases", 0), 0))
            X_context.append(context)
            y_targets.append(target)

    if not X_context:
        return np.array([]), np.array([])

    X = np.array(X_context, dtype=np.float32)
    y = np.array(y_targets, dtype=np.float32)

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    feature_path = PROCESSED_ROOT / "features_daily.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Run `python -m ml.data.build_processed` first.")

    frame = pd.read_csv(feature_path, parse_dates=["date"])
    frame = frame.dropna(subset=["daily_new_cases"])

    X_context, y_targets = _prepare_training_data(frame)
    if len(X_context) == 0:
        raise ValueError("Insufficient training data prepared.")

    print(f"Training samples: {len(X_context)}")

    # Log-normalise targets so ODE I-fraction [0,1] can directly predict them
    y_log = np.log1p(y_targets)
    y_scale = float(y_log.max()) if y_log.max() > 0 else 1.0
    y_norm = (y_log / y_scale).astype(np.float32)
    print(f"Target stats (log-norm) - mean: {y_norm.mean():.4f}, std: {y_norm.std():.4f}")

    split_idx = int(0.8 * len(X_context))
    X_train, y_train = X_context[:split_idx], y_norm[:split_idx]
    X_test,  y_test  = X_context[split_idx:], y_norm[split_idx:]
    y_test_orig = y_targets[split_idx:]

    config = NeuralODEConfig(learning_rate=1e-3, epochs=50, early_stopping_patience=10)
    model = NeuralODEModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    batch_size = 64
    t_span = torch.linspace(0, 1.0, 4, dtype=torch.float32)
    best_loss = float("inf")
    patience_counter = 0
    epochs_done = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i : i + batch_size]
            X_batch = torch.from_numpy(X_train[batch_idx])
            y_batch = torch.from_numpy(y_train[batch_idx])

            y0 = torch.zeros(len(X_batch), 4)
            y0[:, 0] = 0.99
            y0[:, 1] = 0.005
            y0[:, 2] = 0.005

            optimizer.zero_grad()
            try:
                pred, _ = model.forward_normalized(y0, t_span, X_batch)  # (B,)
                loss = criterion(pred, y_batch)
            except Exception as e:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        epochs_done = epoch + 1
        if epochs_done % 5 == 0:
            print(f"Epoch {epochs_done}/{config.epochs}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epochs_done}")
                break

    # Evaluate in original scale
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test)
        y0_test = torch.zeros(len(X_test), 4)
        y0_test[:, 0] = 0.99
        y0_test[:, 1] = 0.005
        y0_test[:, 2] = 0.005
        try:
            pred_norm, _ = model.forward_normalized(y0_test, t_span, X_test_t)
            pred_norm = pred_norm.numpy()
        except Exception:
            pred_norm = np.zeros(len(X_test), dtype=np.float32)

    pred_orig = np.expm1(pred_norm * y_scale)
    test_mae  = float(np.mean(np.abs(pred_orig - y_test_orig)))
    test_rmse = float(np.sqrt(np.mean((pred_orig - y_test_orig) ** 2)))

    print(f"Test MAE:  {test_mae:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    torch.save(model.state_dict(), ARTIFACT_ROOT / "neural_ode_model.pt")
    print(f"Saved: {ARTIFACT_ROOT / 'neural_ode_model.pt'}")

    metrics = {
        "model": "NeuralODE-SEIR",
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "train_samples": int(len(X_train)),
        "epochs_trained": epochs_done,
        "y_scale": y_scale,
    }
    with open(ARTIFACT_ROOT / "neural_ode_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")


if __name__ == "__main__":
    main()
