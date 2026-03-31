from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from ..models.schemas import ForecastRequest, ForecastResponse, ForecastSeries
from ml.models.neural_ode_v2 import NeuralODEModel, NeuralODEConfig
from ml.models.temporal_gat_v2 import TemporalGATModel, TemporalGATConfig

_PROCESSED_ROOT = Path(__file__).resolve().parents[3] / "data" / "processed"
_FEATURES_CACHE: pd.DataFrame | None = None


def _load_features() -> pd.DataFrame:
    global _FEATURES_CACHE
    if _FEATURES_CACHE is None:
        path = _PROCESSED_ROOT / "features_daily.csv"
        if path.exists():
            _FEATURES_CACHE = pd.read_csv(path, parse_dates=["date"])
        else:
            _FEATURES_CACHE = pd.DataFrame()
    return _FEATURES_CACHE


_FALLBACK_LAST_DATE = date(2023, 3, 9)  # JHU data ended March 10 2023


def _get_country_base(region: str) -> tuple[float, float, date]:
    """Return (recent_avg_cases, trend_rate, last_data_date) for the region."""
    features = _load_features()
    if features.empty:
        return 1000.0, 1.002, _FALLBACK_LAST_DATE
    rows = features[features["country"].str.lower() == region.lower()]
    if rows.empty:
        return 1000.0, 1.002, _FALLBACK_LAST_DATE
    rows = rows.sort_values("date").tail(30)
    last_data_date = rows["date"].iloc[-1].date()
    cases = rows["daily_new_cases"].clip(lower=0).fillna(0).to_numpy(dtype=float)
    if len(cases) == 0 or cases.mean() == 0:
        return 500.0, 1.001, last_data_date
    base = float(np.mean(cases[-7:]) if len(cases) >= 7 else np.mean(cases))
    # trend: growth rate from past 14d -> 7d
    if len(cases) >= 14:
        old = np.mean(cases[-14:-7]) + 1.0
        recent = np.mean(cases[-7:]) + 1.0
        trend = float((recent / old) ** (1 / 7))
    else:
        trend = 1.001
    trend = float(np.clip(trend, 0.97, 1.04))
    return max(base, 10.0), trend, last_data_date


def _load_trained_models() -> tuple:
    """Load preprocessed Phase 3 models if available."""
    artifact_root = Path(__file__).resolve().parents[3] / "ml" / "artifacts"
    
    neural_ode_model = None
    temporal_gat_model = None
    
    try:
        ode_path = artifact_root / "neural_ode_model.pt"
        if ode_path.exists():
            config = NeuralODEConfig()
            neural_ode_model = NeuralODEModel(config)
            neural_ode_model.load_state_dict(torch.load(ode_path, map_location="cpu"))
            neural_ode_model.eval()
            print(f"[INFO] Loaded Neural ODE model from {ode_path}")
    except Exception as e:
        print(f"[WARN] Could not load Neural ODE model: {e}")
    
    try:
        gat_path = artifact_root / "temporal_gat_model.pt"
        if gat_path.exists():
            config = TemporalGATConfig(num_nodes=201)
            temporal_gat_model = TemporalGATModel(config)
            temporal_gat_model.load_state_dict(torch.load(gat_path, map_location="cpu"))
            temporal_gat_model.eval()
            print(f"[INFO] Loaded Temporal GAT model from {gat_path}")
    except Exception as e:
        print(f"[WARN] Could not load Temporal GAT model: {e}")
    
    return neural_ode_model, temporal_gat_model


_NEURAL_ODE_MODEL, _TEMPORAL_GAT_MODEL = _load_trained_models()


def build_forecast(payload: ForecastRequest, scenario_scale: float = 1.0) -> ForecastResponse:
    forecasts: list[ForecastSeries] = []

    for region in payload.region_ids:
        dates: list[date] = []
        predictions: list[float] = []
        ci_lower: list[float] = []
        ci_upper: list[float] = []

        base_cases, trend_rate, last_data_date = _get_country_base(region)

        for day in range(payload.horizon):
            # Anchor forecast to the last date in the country's actual data,
            # not today — prevents a gap when the dataset predates the current date.
            dates.append(last_data_date + timedelta(days=day + 1))

            if _NEURAL_ODE_MODEL is not None and _TEMPORAL_GAT_MODEL is not None:
                try:
                    # Contextualise ODE with normalised data-driven features
                    features = _load_features()
                    ctx_vals = [0.6, 0.8, 0.3, 0.1]
                    if not features.empty:
                        rows = features[features["country"].str.lower() == region.lower()]
                        if not rows.empty:
                            r = rows.sort_values("date").iloc[-1]
                            mob = float(r.get("mobility_index", 0.0) or 0.0)
                            str_idx = float(r.get("stringency_index", 50.0) or 50.0) / 100.0
                            vax = float(r.get("people_fully_vaccinated_per_hundred", 0.0) or 0.0) / 100.0
                            accel = float(r.get("case_acceleration", 0.0) or 0.0)
                            accel_norm = float(np.clip(accel / 5000.0, -1.0, 1.0)) * 0.5 + 0.5
                            ctx_vals = [mob / 100.0 + 0.5, str_idx, vax, accel_norm]

                    context = torch.tensor([ctx_vals], dtype=torch.float32)
                    y0 = torch.tensor([[0.99, 0.005, 0.005, 0.0]], dtype=torch.float32)
                    t_span = torch.linspace(0, float(day + 1) / 30.0, 4, dtype=torch.float32)

                    with torch.no_grad():
                        ode_solution = _NEURAL_ODE_MODEL(y0, t_span, context)

                    # I (infected) compartment scaled to population base
                    ode_I = float(ode_solution[-1, 0, 2].numpy())
                    ode_pred = max(ode_I * base_cases * 10.0, base_cases * 0.5)

                    # GAT uses real node count; build minimal random context for speed
                    num_nodes = _TEMPORAL_GAT_MODEL.config.num_nodes
                    gat_context = torch.zeros(num_nodes, 14, 4)
                    edge_index = torch.randint(0, num_nodes, (2, min(300, num_nodes * 2)))
                    edge_weight = torch.ones(edge_index.shape[1])

                    with torch.no_grad():
                        gat_out = _TEMPORAL_GAT_MODEL(gat_context, edge_index, edge_weight)

                    gat_raw = float(gat_out["forecast"][0, 0].numpy())
                    gat_pred = float(np.expm1(max(gat_raw, 0.0)) * base_cases / 1000.0 + base_cases * trend_rate ** day)

                    ensemble_pred = 0.45 * ode_pred + 0.55 * gat_pred
                    value = max(ensemble_pred * scenario_scale, 0.0)
                except Exception as e:
                    print(f"[WARN] Model inference failed for {region}: {e}, using data baseline")
                    value = max(base_cases * (trend_rate ** day) * scenario_scale, 0.0)
            else:
                # Data-driven baseline: actual recent average + trend extrapolation
                value = max(base_cases * (trend_rate ** day) * scenario_scale, 0.0)

            # Widen CI as horizon grows
            spread_pct = 0.10 + 0.005 * day
            spread = value * spread_pct
            predictions.append(float(round(value, 2)))
            ci_lower.append(float(round(max(value - spread, 0.0), 2)))
            ci_upper.append(float(round(value + spread, 2)))

        forecasts.append(
            ForecastSeries(
                region=region,
                dates=dates,
                predicted_cases=predictions,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return ForecastResponse(forecasts=forecasts)
