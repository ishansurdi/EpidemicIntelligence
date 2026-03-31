# Architecture

## Objective

Build an epidemic intelligence system that predicts spread and outbreak risk while supporting intervention simulation and explainability.

## Design Principles

- Mechanistic + data-driven: blend epidemiological constraints with deep learning
- Forecast + classify: case forecasting and outbreak probability in one platform
- Explainable by default: every score should have a transmission and feature rationale
- Policy-friendly: support what-if intervention analysis, not only passive prediction

## System Components

## 1) Data Layer

Inputs:
- JHU global confirmed time series
- OWID epidemiological indicators (testing, vaccination, policy, hospital)
- Mobility signals (Google and OWID mobility products)

Responsibilities:
- align by country ISO and date
- derive temporal epidemiological features
- create graph-ready node and edge tensors

## 2) ML Layer

Modules:
- `Neural ODE` (physics-informed transmission dynamics)
- `Temporal GAT` (spatial cascade and temporal wave modeling)
- `Ensemble` (stability and robustness)
- `Outbreak risk head` (binary risk with calibrated probabilities)

Outputs:
- horizon forecasts with uncertainty bounds
- outbreak probability and risk tier
- attention corridors and feature importance payloads

## 3) Inference Layer

- prediction orchestration service
- scenario engine that perturbs edge/node conditions
- cascade tracer to estimate probable upstream influence chains

## 4) API Layer (FastAPI)

- `/api/v1/data/*` data access and engineered features
- `/api/v1/predict/*` forecasting, risk, and scenario endpoints
- `/api/v1/cascade/*` causal trace surface
- `/api/v1/interpret/*` explainability payloads

## 5) Frontend Layer (Phase 1)

A static command-center style UI in HTML + Tailwind to validate interaction architecture and panel semantics before React implementation.

## Storage and Infra (Current)

- No Redis in this phase
- Local files for datasets and artifacts
- PostgreSQL integration planned for job metadata and persistent serving data

## Evaluation Strategy

- Forecast metrics: MAE, RMSE, MAPE
- Outbreak metrics: AUC-ROC, F1
- Spatial holdout: train on subset of countries, evaluate on unseen countries
- Ablation: mobility edges, policy signals, vaccination inputs, attention regularization

## Governance and Trust Signals

To align with public-health-grade expectations:
- confidence intervals and calibration reporting
- model card with intended-use boundaries
- data freshness and provenance logs
- transparent assumptions and known failure modes
