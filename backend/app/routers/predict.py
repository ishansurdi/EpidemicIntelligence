from fastapi import APIRouter, Query

from ..models.schemas import (
    ForecastRequest,
    ForecastResponse,
    OutbreakRiskRequest,
    OutbreakRiskResponse,
    ScenarioRequest,
    ScenarioResponse,
)
from ..services.forecast_service import build_forecast
from ..services.risk_service import build_outbreak_risk
from ..services.scenario_service import run_scenario

router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest) -> ForecastResponse:
    return build_forecast(payload)


@router.get("/forecast")
def forecast_get(
    country: str = Query(default="India"),
    horizon: int = Query(default=28, ge=1, le=60),
) -> ForecastResponse:
    payload = ForecastRequest(region_ids=[country], horizon=horizon)
    return build_forecast(payload)


@router.post("/outbreak-risk", response_model=OutbreakRiskResponse)
def outbreak_risk(payload: OutbreakRiskRequest) -> OutbreakRiskResponse:
    return build_outbreak_risk(payload)


@router.get("/outbreak-risk")
def outbreak_risk_get(country: str = Query(default="India")) -> OutbreakRiskResponse:
    payload = OutbreakRiskRequest(region_ids=[country])
    return build_outbreak_risk(payload)


@router.post("/scenario", response_model=ScenarioResponse)
def scenario(payload: ScenarioRequest) -> ScenarioResponse:
    return run_scenario(payload)
