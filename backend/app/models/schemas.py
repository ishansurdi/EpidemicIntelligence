from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    region_ids: list[str] = Field(min_length=1)
    horizon: int = Field(default=28, ge=1, le=60)
    confidence: bool = True


class ForecastSeries(BaseModel):
    region: str
    dates: list[date]
    predicted_cases: list[float]
    ci_lower: list[float]
    ci_upper: list[float]


class ForecastResponse(BaseModel):
    forecasts: list[ForecastSeries]


class OutbreakRiskRequest(BaseModel):
    region_ids: list[str] = Field(min_length=1)


class RegionRisk(BaseModel):
    region: str
    outbreak_probability: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"]
    contributing_factors: dict[str, float]


class OutbreakRiskResponse(BaseModel):
    risks: list[RegionRisk]


class Intervention(BaseModel):
    type: Literal[
        "mobility_reduction",
        "vaccination_acceleration",
        "border_closure",
        "policy_shift",
    ]
    region_pair: list[str] | None = None
    region: str | None = None
    magnitude: float = Field(default=0.0, ge=-1.0, le=1.0)


class ScenarioRequest(BaseModel):
    interventions: list[Intervention]
    region_ids: list[str] = Field(min_length=1)
    horizon: int = Field(default=28, ge=1, le=60)


class ScenarioRegionDelta(BaseModel):
    region: str
    delta_cases: float
    delta_percent: float


class ScenarioResponse(BaseModel):
    baseline_forecast: ForecastResponse
    intervention_forecast: ForecastResponse
    delta: list[ScenarioRegionDelta]
    regions_affected: list[str]


class CascadeStep(BaseModel):
    region: str
    lag_days: int
    attention_weight: float


class CascadeTraceResponse(BaseModel):
    origin_chain: list[CascadeStep]
    cascade_tree: dict[str, list[str]]


class AttentionEdge(BaseModel):
    source: str
    target: str
    weight: float


class AttentionMapResponse(BaseModel):
    edges: list[AttentionEdge]


class FeatureImportanceResponse(BaseModel):
    region: str
    shap_values: dict[str, float]
