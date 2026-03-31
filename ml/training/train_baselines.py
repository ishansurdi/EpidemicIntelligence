import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Rank-based AUC (Mann-Whitney U relation).
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg))


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.2, 0.8, 31):
        pred = (y_prob >= t).astype(int)
        f1 = _f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    feature_path = PROCESSED_ROOT / "features_daily.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            "Missing processed feature file. Run `python -m ml.data.build_processed` first."
        )

    frame = pd.read_csv(feature_path, parse_dates=["date"])
    frame = frame.dropna(subset=["daily_new_cases", "rolling_7d_cases", "outbreak_label"])
    frame = frame.sort_values(["country", "date"])

    split_date = frame["date"].quantile(0.8)
    train = frame[frame["date"] <= split_date].copy()
    test = frame[frame["date"] > split_date].copy()

    # Forecast baseline: one-step persistence of rolling mean.
    test["forecast_pred"] = test.groupby("country")["rolling_7d_cases"].shift(1)
    test["forecast_pred"] = test["forecast_pred"].fillna(test["rolling_7d_cases"])

    y_true_forecast = test["daily_new_cases"].to_numpy(dtype=float)
    y_pred_forecast = test["forecast_pred"].to_numpy(dtype=float)

    forecast_metrics = {
        "mae": round(_mae(y_true_forecast, y_pred_forecast), 6),
        "rmse": round(_rmse(y_true_forecast, y_pred_forecast), 6),
        "mape": round(_mape(y_true_forecast, y_pred_forecast), 6),
    }

    risk_features = [
        "recent_growth_14d",
        "rolling_7d_cases",
        "case_acceleration",
        "mobility_index",
        "stringency_index",
        "people_fully_vaccinated_per_hundred",
    ]
    available = [col for col in risk_features if col in train.columns]

    train_matrix = train[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    means = train_matrix.mean(axis=0)
    stds = train_matrix.std(axis=0)
    stds[stds == 0.0] = 1.0

    weights = np.array([0.70, 0.35, 0.25, 0.20, -0.10, -0.20], dtype=float)[: len(available)]
    train_z = (train_matrix - means) / stds
    train_score = train_z @ weights
    train_prob = 1.0 / (1.0 + np.exp(-train_score))
    y_train = train["outbreak_label"].to_numpy(dtype=int)
    threshold, train_f1 = _best_threshold(y_train, train_prob)

    test_matrix = test[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    test_z = (test_matrix - means) / stds
    test_prob = 1.0 / (1.0 + np.exp(-(test_z @ weights)))
    y_test = test["outbreak_label"].to_numpy(dtype=int)
    y_test_pred = (test_prob >= threshold).astype(int)

    outbreak_metrics = {
        "auc_roc": round(_roc_auc(y_test, test_prob), 6),
        "f1": round(_f1_score(y_test, y_test_pred), 6),
        "train_f1": round(train_f1, 6),
        "threshold": round(float(threshold), 6),
    }

    metrics_payload = {
        "split_date": split_date.strftime("%Y-%m-%d"),
        "forecast": forecast_metrics,
        "outbreak": outbreak_metrics,
        "rows": {
            "train": int(len(train)),
            "test": int(len(test)),
        },
    }

    risk_artifact = {
        "features": available,
        "weights": [float(x) for x in weights],
        "means": [float(x) for x in means],
        "stds": [float(x) for x in stds],
        "threshold": float(threshold),
    }

    forecast_artifact = {
        "name": "rolling_7d_persistence",
        "description": "Predict next-day new cases using previous rolling_7d_cases signal.",
    }

    (ARTIFACT_ROOT / "phase2_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (ARTIFACT_ROOT / "outbreak_baseline.json").write_text(json.dumps(risk_artifact, indent=2), encoding="utf-8")
    (ARTIFACT_ROOT / "forecast_baseline.json").write_text(json.dumps(forecast_artifact, indent=2), encoding="utf-8")

    print("Saved baseline artifacts:")
    print(f"- {ARTIFACT_ROOT / 'phase2_metrics.json'}")
    print(f"- {ARTIFACT_ROOT / 'outbreak_baseline.json'}")
    print(f"- {ARTIFACT_ROOT / 'forecast_baseline.json'}")
    print(f"Forecast metrics: {forecast_metrics}")
    print(f"Outbreak metrics: {outbreak_metrics}")


if __name__ == "__main__":
    main()