import torch
import torch.optim as optim
from pathlib import Path
import json

import pandas as pd
import numpy as np

from ml.models.temporal_gat_v2 import TemporalGATModel, TemporalGATConfig

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


FEATURE_COLUMNS = [
    "mobility_index",
    "stringency_index",
    "people_fully_vaccinated_per_hundred",
    "case_acceleration",
]


def _build_spatial_graph(graph_df: pd.DataFrame, countries: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    country_to_idx = {country: idx for idx, country in enumerate(countries)}
    edges: list[list[int]] = []
    weights: list[float] = []

    for row in graph_df.itertuples(index=False):
        if row.source not in country_to_idx or row.target not in country_to_idx:
            continue
        edges.append([country_to_idx[row.source], country_to_idx[row.target]])
        weights.append(float(row.weight))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight, country_to_idx


def _normalize_by_country(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for col in columns:
        group_mean = normalized.groupby("country")[col].transform("mean")
        group_std = normalized.groupby("country")[col].transform("std").replace(0.0, 1.0).fillna(1.0)
        normalized[col] = ((normalized[col].fillna(0.0) - group_mean.fillna(0.0)) / group_std).clip(-5.0, 5.0)
    return normalized


def _prepare_temporal_graph_data(frame: pd.DataFrame, countries: list[str], seq_len: int = 14, horizon: int = 7) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    frame = frame.sort_values(["country", "date"]).copy()
    normalized = _normalize_by_country(frame, FEATURE_COLUMNS)

    pivot_features: dict[str, pd.DataFrame] = {}
    for col in FEATURE_COLUMNS:
        pivot_features[col] = normalized.pivot(index="date", columns="country", values=col).reindex(columns=countries).fillna(0.0)
    target_pivot = frame.pivot(index="date", columns="country", values="daily_new_cases").reindex(columns=countries).fillna(0.0)

    dates = target_pivot.index.to_list()
    if len(dates) < seq_len + horizon:
        return [], []

    windows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for end_idx in range(seq_len, len(dates) - horizon + 1):
        start_idx = end_idx - seq_len
        feature_slices = []
        for col in FEATURE_COLUMNS:
            values = pivot_features[col].iloc[start_idx:end_idx].to_numpy(dtype=np.float32).T
            feature_slices.append(values)
        window = np.stack(feature_slices, axis=-1)
        future_target = target_pivot.iloc[end_idx + horizon - 1].to_numpy(dtype=np.float32)
        future_target = np.log1p(np.clip(future_target, 0.0, None))
        windows.append(torch.tensor(window, dtype=torch.float32))
        targets.append(torch.tensor(future_target, dtype=torch.float32))

    return windows, targets


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    
    feature_path = PROCESSED_ROOT / "features_daily.csv"
    graph_path = PROCESSED_ROOT / "graph_snapshot.csv"
    
    if not feature_path.exists():
        raise FileNotFoundError("Run `python -m ml.data.build_processed` first.")
    if not graph_path.exists():
        raise FileNotFoundError("Missing graph_snapshot.csv. Run `python -m ml.data.build_processed` first.")
    
    frame = pd.read_csv(feature_path, parse_dates=["date"])
    frame = frame.dropna(subset=["daily_new_cases"])
    graph_df = pd.read_csv(graph_path)
    print(f"Loaded graph snapshot: {len(graph_df)} edges")
    
    print(f"Training data: {len(frame)} rows, {frame['country'].nunique()} countries")

    countries = sorted(set(frame["country"].unique()) & (set(graph_df["source"]) | set(graph_df["target"])))
    windows, targets = _prepare_temporal_graph_data(frame, countries)
    edge_index, edge_weight, country_to_idx = _build_spatial_graph(graph_df, countries)

    if len(windows) == 0:
        raise ValueError("Insufficient training sequences prepared.")

    # Stride windows to reduce training time (every 3rd window keeps temporal diversity)
    STRIDE = 3
    windows = windows[::STRIDE]
    targets = targets[::STRIDE]

    print(f"Temporal windows: {len(windows)} (strided by {STRIDE})")
    print(f"Graph: {len(country_to_idx)} nodes, {edge_index.shape[1]} edges")

    split_idx = int(0.8 * len(windows))
    train_windows = windows[:split_idx]
    train_targets = targets[:split_idx]
    test_windows = windows[split_idx:]
    test_targets = targets[split_idx:]

    config = TemporalGATConfig(
        num_nodes=len(country_to_idx),
        temporal_dim=14,
        feature_dim=4,
        num_heads=4,
        hidden_dim=32,
        gat_layers=2,
        learning_rate=1e-3,
        epochs=30,
        early_stopping_patience=8,
    )

    model = TemporalGATModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    forecast_criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        batch_losses: list[float] = []

        for x_batch, y_batch in zip(train_windows, train_targets):
            optimizer.zero_grad()
            output = model(x_batch, edge_index, edge_weight)
            forecast_pred = output["forecast"].squeeze(-1)
            forecast_loss = forecast_criterion(forecast_pred, y_batch)
            forecast_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(float(forecast_loss.item()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.eval()
    mae_values: list[float] = []
    rmse_values: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in zip(test_windows, test_targets):
            output_test = model(x_batch, edge_index, edge_weight)
            forecast_test = torch.expm1(output_test["forecast"].squeeze(-1)).cpu().numpy()
            target_test = torch.expm1(y_batch).cpu().numpy()
            mae_values.append(float(np.mean(np.abs(forecast_test - target_test))))
            rmse_values.append(float(np.sqrt(np.mean((forecast_test - target_test) ** 2))))

    test_mae = float(np.mean(mae_values)) if mae_values else None
    test_rmse = float(np.mean(rmse_values)) if rmse_values else None

    if test_mae is not None and test_rmse is not None:
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")

    torch.save(model.state_dict(), ARTIFACT_ROOT / "temporal_gat_model.pt")
    print(f"Saved: {ARTIFACT_ROOT / 'temporal_gat_model.pt'}")

    metrics = {
        "model": "TemporalGAT",
        "config": {
            "num_nodes": config.num_nodes,
            "temporal_dim": config.temporal_dim,
            "feature_dim": config.feature_dim,
            "num_heads": config.num_heads,
            "hidden_dim": config.hidden_dim,
            "gat_layers": config.gat_layers,
        },
        "train_windows": len(train_windows),
        "test_windows": len(test_windows),
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }

    with open(ARTIFACT_ROOT / "temporal_gat_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {ARTIFACT_ROOT / 'temporal_gat_metrics.json'}")


if __name__ == "__main__":
    main()
