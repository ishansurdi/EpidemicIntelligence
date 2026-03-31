from pathlib import Path
import numpy as np

from ..models.schemas import (
    ForecastRequest,
    ScenarioRegionDelta,
    ScenarioRequest,
    ScenarioResponse,
)
from .forecast_service import build_forecast
from ml.inference.scenario_runner import ScenarioRunner


def _get_scenario_runner() -> ScenarioRunner | None:
    """Load scenario runner with trained models."""
    try:
        artifact_root = Path(__file__).resolve().parents[3] / "ml" / "artifacts"
        return ScenarioRunner.from_artifacts(artifact_root)
    except Exception as e:
        print(f"[WARN] Could not initialize ScenarioRunner: {e}")
        return None


_SCENARIO_RUNNER = _get_scenario_runner()


def _intervention_scale(request: ScenarioRequest) -> float:
    scale = 1.0
    for intervention in request.interventions:
        if intervention.type == "mobility_reduction":
            scale *= 1.0 - (0.25 * intervention.magnitude)
        elif intervention.type == "vaccination_acceleration":
            scale *= 1.0 - (0.20 * intervention.magnitude)
        elif intervention.type == "border_closure":
            scale *= 1.0 - (0.15 * abs(intervention.magnitude))
        elif intervention.type == "policy_shift":
            scale *= 1.0 - (0.10 * intervention.magnitude)
    return max(scale, 0.55)


def run_scenario(payload: ScenarioRequest) -> ScenarioResponse:
    baseline_req = ForecastRequest(
        region_ids=payload.region_ids,
        horizon=payload.horizon,
        confidence=True,
    )
    baseline = build_forecast(baseline_req, scenario_scale=1.0)

    scenario_scale = _intervention_scale(payload)
    intervention = build_forecast(baseline_req, scenario_scale=scenario_scale)

    deltas: list[ScenarioRegionDelta] = []
    affected: list[str] = []

    for base_series, scenario_series in zip(baseline.forecasts, intervention.forecasts):
        base_sum = sum(base_series.predicted_cases)
        scenario_sum = sum(scenario_series.predicted_cases)
        delta = scenario_sum - base_sum
        delta_pct = (delta / base_sum * 100.0) if base_sum > 0 else 0.0
        deltas.append(
            ScenarioRegionDelta(
                region=base_series.region,
                delta_cases=round(delta, 2),
                delta_percent=round(delta_pct, 2),
            )
        )
        if abs(delta_pct) > 2.0:
            affected.append(base_series.region)

    return ScenarioResponse(
        baseline_forecast=baseline,
        intervention_forecast=intervention,
        delta=deltas,
        regions_affected=affected,
    )
