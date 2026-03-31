# ORACLE: Epidemic Cascade Intelligence

ORACLE is a full-stack epidemic forecasting platform designed for decision support, not just leaderboard metrics.

This repository starts with a strong Phase 1 foundation:
- no-Redis backend architecture (FastAPI + PostgreSQL-ready)
- reproducible ML pipeline skeleton for JHU + OWID + mobility data
- unique static frontend prototype in HTML + Tailwind CSS
- architecture and model docs suitable for technical judging

## Why This Wins

Most teams forecast isolated country curves. ORACLE models epidemics as cascade dynamics on a spatio-temporal graph, then exposes intervention simulation for actionability.

Core differentiators:
- Physics-informed transmission dynamics (Neural ODE-ready)
- Graph attention for cross-border cascade modeling
- Outbreak risk classification and scenario simulation in one pipeline
- Spatial holdout evaluation to prove generalization
- Interpretability first-class (attention corridors + feature attribution)

## Phase Plan

## Phase 1 (current in this repo)
- Repository scaffolding and architecture contracts
- FastAPI service with strongly typed endpoint schemas
- Data loading + feature engineering skeleton tied to local datasets in `data/`
- Static frontend command center mock (HTML + Tailwind only)
- Documentation package (`docs/`)

## Phase 2 (✅ Complete)
- Robust ETL and processed feature store generation
- Baseline outbreak and forecast models
- Model artifacts and evaluation suite

## Phase 3 (🔨 In Progress)
- **Neural ODE**: Physics-informed SEIR transmission dynamics
- **Temporal GAT**: Graph Attention Networks for cascade modeling
- **Ensemble**: Learned weighted combination of ODE + GAT forecasts
- **Scenario Engine**: Intervention simulation with real forward passes
- **Cascade Tracer**: Attention-based upstream source detection
- **Interpretability**: SHAP values, attention heatmaps, feature attribution

## Phase 4 (Planned)
- Rich frontend interactivity (React + deck.gl globe)
- CI/CD, data quality checks, model cards
- Policy-grade governance and fairness reporting

## Current Project Layout

```text
backend/      FastAPI app and services
frontend/     Static HTML + Tailwind prototype (Phase 1)
ml/           Data, model, training, and inference pipeline skeleton
docs/         Architecture, API, and model documentation
data/         Source datasets (already available locally)
notebooks/    EDA and experiment notebooks
```

## Quick Start (Phase 3)

### Prerequisites

```bash
.venv\Scripts\python -m pip install -r backend/requirements.txt
```

### Run API

```bash
.venv\Scripts\python -m uvicorn backend.app.main:app --reload
```

API root: `http://127.0.0.1:8000`

Test: `curl http://127.0.0.1:8000/api/v1/health`

### Open Frontend

Open `frontend/index.html` in any browser.

### Prepare Processed Data (one-time)

```bash
.venv\Scripts\python -m ml.data.build_processed
```

### Train Phase 3 Models

```bash
.venv\Scripts\python -m ml.training.train_neural_ode_v2     # Physics-informed Neural ODE with SEIR dynamics
.venv\Scripts\python -m ml.training.train_temporal_gat_v2   # Graph Attention Networks for spatio-temporal forecasting
```

These training scripts will:
1. Load processed data from `data/processed/`
2. Normalize features by country
3. Build temporal sequences and spatial graphs
4. Train models using Adam optimizer with early stopping
5. Save trained weights to `ml/artifacts/`
6. Generate evaluation metrics

### Run Scenario Analysis

```bash
.venv\Scripts\python -c "from ml.inference.scenario_runner import run_example_scenarios; run_example_scenarios()"
```

## Data Sources Used

- Johns Hopkins CSSE confirmed cases (`data/time_series_covid19_confirmed_global.csv`)
- OWID datasets (`data/owid/*.csv`)
- Google mobility report (`data/Global_Mobility_Report.csv` and `data/owid/google_mobility.csv`)

## Engineering Notes

- **No Docker / No Redis**: Direct Python execution using installed packages. Clean, reproducible, auditable.
- **All endpoints return typed JSON contracts** to unblock frontend and model integration in parallel.
- **Phase 3 models** (Neural ODE + Temporal GAT) integrate seamlessly with existing backend services via standardized artifact loading.
- This repository is designed for incremental, judge-friendly demos at each phase milestone.
