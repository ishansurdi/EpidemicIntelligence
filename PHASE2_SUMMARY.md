# Phase 2 Completion Checklist

## ETL & Data Pipeline ✓

- [x] Designed feature engineering with outbreak labels, growth features, and temporal aggregates
- [x] Implemented `ml/data/build_processed.py` to merge JHU, OWID, and mobility data
- [x] Generated processed datasets in `data/processed/` directory
  - `timeseries_daily.csv`: Base confirmed cases and daily new cases
  - `features_daily.csv`: Complete feature matrix with epidemiological and mobility indicators
  - `graph_snapshot.csv`: Country connectivity graph derived from feature similarity
- [x] Updated `backend/app/services/data_service.py` to prefer processed outputs

## Baseline Models ✓

- [x] Implemented `ml/training/train_baselines.py` with:
  - Forecast baseline using rolling-7d persistence
  - Outbreak risk baseline using normalized features +weighted scoring
  - Automatic threshold search and evaluation
- [x] Generated artifact files in `ml/artifacts/`:
  - `phase2_metrics.json`: Split date, MAE/RMSE/MAPE, AUC-ROC/F1, row counts
  - `outbreak_baseline.json`: Feature weights, means, stds, threshold for inference
  - `forecast_baseline.json`: Model metadata
- [x] Updated evaluation script to read and report metrics

## API Integration ✓

- [x] Backend now checks for processed datasets before falling back to raw files
- [x] Graph endpoint now returns full computed edges instead of placeholder
- [x] All endpoints remain typed and stable (Phase 1 contracts preserved)
- [x] Created `run_phase2_test.py` to verify API endpoint integration

## Config Updates ✓

- [x] Updated `ml/configs/experiment.yaml` to reference processed/artifacts paths
- [x] Added baseline strategy and feature requirements to YAML

## Current Metrics (Phase 2 Baseline)

- **Forecast** (rolling-7d persistence):
  - MAE: 1328.87
  - RMSE: 7397.56
  - MAPE: 36276.66
- **Outbreak Classification** (normalized features + logistic):
  - AUC-ROC: 0.519
  - F1: 0.041
  - Threshold: 0.5
- **Data split**: 80/20 at 2022-07-24
- **Training rows**: 183,915 | **Test rows**: 45,828

## What This Enables

1. Real feature pipelines for Phase 3 model training (Neural ODE, T-GAT)
2. Reproducible baseline to benchmark improvements against
3. Live API integration with processed datasets for judges
4. Outbreak label generation for supervised classification
5. Graph topology learned from data similarity (not placeholder)

## Next Steps (Phase 3)

- Implement PyTorch Neural ODE with torchdiffeq
- Implement PyTorch Geometric Temporal GAT
- Add ensemble weighted combining
- Add interpretability (SHAP, attention visualization)
- Add scenario simulation with graph perturbations
- Upgrade frontend to React + deck.gl globe

## Running Phase 2

```bash
# Rebuild processed data and artifacts
python -m ml.data.build_processed
python -m ml.training.train_baselines
python -m ml.training.evaluate

# Start API
uvicorn backend.app.main:app --reload

# Test all endpoints
python run_phase2_test.py
```
