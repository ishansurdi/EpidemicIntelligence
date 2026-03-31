import json
from pathlib import Path


def main() -> None:
    artifact_path = Path(__file__).resolve().parents[1] / "artifacts" / "phase2_metrics.json"
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Phase 2 metrics not found. Run `python -m ml.training.train_baselines` first."
        )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    print("Evaluation summary")
    print(f"Split date: {payload['split_date']}")
    print(f"Forecast MAE: {payload['forecast']['mae']:.6f}")
    print(f"Forecast RMSE: {payload['forecast']['rmse']:.6f}")
    print(f"Forecast MAPE: {payload['forecast']['mape']:.6f}")
    print(f"Outbreak AUC-ROC: {payload['outbreak']['auc_roc']:.6f}")
    print(f"Outbreak F1: {payload['outbreak']['f1']:.6f}")
    print(f"Rows train/test: {payload['rows']['train']}/{payload['rows']['test']}")


if __name__ == "__main__":
    main()
