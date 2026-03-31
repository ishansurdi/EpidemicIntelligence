from pathlib import Path

import pandas as pd

from ..models.schemas import OutbreakRiskRequest, OutbreakRiskResponse, RegionRisk

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


def _compute_risk(row: pd.Series) -> tuple[float, dict[str, float]]:
    vax_gap = 1.0 - float(row.get("people_fully_vaccinated_per_hundred", 0) or 0) / 100.0
    mobility_raw = float(row.get("mobility_index", 0.0) or 0.0)
    mobility_norm = min(max((mobility_raw + 100) / 125.0, 0.0), 1.0)
    stringency = float(row.get("stringency_index", 50) or 50) / 100.0
    accel = float(row.get("case_acceleration", 0) or 0)
    accel_norm = min(max(accel / 5000.0, -1.0), 1.0) * 0.5 + 0.5

    factors = {
        "vaccination_gap": round(vax_gap * 0.30, 4),
        "mobility_connectivity": round(mobility_norm * 0.25, 4),
        "low_policy_stringency": round((1.0 - stringency) * 0.20, 4),
        "case_acceleration": round(accel_norm * 0.25, 4),
    }
    score = sum(factors.values())
    return round(min(max(score, 0.0), 1.0), 4), factors


def _risk_label(probability: float) -> str:
    if probability >= 0.75:
        return "critical"
    if probability >= 0.55:
        return "high"
    if probability >= 0.35:
        return "medium"
    return "low"


def build_outbreak_risk(payload: OutbreakRiskRequest) -> OutbreakRiskResponse:
    features = _load_features()
    risks: list[RegionRisk] = []

    for region in payload.region_ids:
        if not features.empty:
            country_rows = features[features["country"].str.lower() == region.lower()]
            if not country_rows.empty:
                row = country_rows.sort_values("date").iloc[-1]
                probability, factors = _compute_risk(row)
            else:
                # Fallback: neutral score
                probability = 0.50
                factors = {"vaccination_gap": 0.15, "mobility_connectivity": 0.12, "low_policy_stringency": 0.10, "case_acceleration": 0.13}
        else:
            probability = 0.50
            factors = {"vaccination_gap": 0.15, "mobility_connectivity": 0.12, "low_policy_stringency": 0.10, "case_acceleration": 0.13}

        risks.append(
            RegionRisk(
                region=region,
                outbreak_probability=probability,
                risk_level=_risk_label(probability),
                contributing_factors=factors,
            )
        )
    return OutbreakRiskResponse(risks=risks)
