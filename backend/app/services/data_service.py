from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"

# Module-level caches — load each CSV exactly once per process lifetime
_TIMESERIES_DF: pd.DataFrame | None = None
_FEATURES_DF: pd.DataFrame | None = None
_RISK_MAP_CACHE: dict | None = None


def _get_timeseries_df() -> pd.DataFrame:
    global _TIMESERIES_DF
    if _TIMESERIES_DF is None:
        processed_path = PROCESSED_ROOT / "timeseries_daily.csv"
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            _TIMESERIES_DF = df.sort_values(["country", "date"])
        else:
            _TIMESERIES_DF = pd.DataFrame()
    return _TIMESERIES_DF


def _get_features_df() -> pd.DataFrame:
    global _FEATURES_DF
    if _FEATURES_DF is None:
        processed_path = PROCESSED_ROOT / "features_daily.csv"
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            _FEATURES_DF = df.sort_values(["country", "date"])
        else:
            _FEATURES_DF = pd.DataFrame()
    return _FEATURES_DF


def load_jhu_timeseries(country: str | None = None) -> pd.DataFrame:
    grouped = _get_timeseries_df()
    if grouped.empty:
        # Fallback: parse raw JHU CSV (first call only for this path)
        file_path = DATA_ROOT / "time_series_covid19_confirmed_global.csv"
        frame = pd.read_csv(file_path)
        value_columns = [
            col for col in frame.columns
            if col not in ["Province/State", "Country/Region", "Lat", "Long"]
        ]
        melted = frame.melt(
            id_vars=["Province/State", "Country/Region", "Lat", "Long"],
            value_vars=value_columns,
            var_name="date", value_name="confirmed_cases",
        )
        melted["date"] = pd.to_datetime(melted["date"], errors="coerce")
        grouped = (
            melted.groupby(["Country/Region", "date"], as_index=False)["confirmed_cases"]
            .sum()
            .rename(columns={"Country/Region": "country"})
        )
        grouped = grouped.sort_values(["country", "date"])
    if country:
        return grouped[grouped["country"].str.lower() == country.lower()]
    return grouped

    file_path = DATA_ROOT / "time_series_covid19_confirmed_global.csv"
    frame = pd.read_csv(file_path)

    value_columns = [
        col
        for col in frame.columns
        if col not in ["Province/State", "Country/Region", "Lat", "Long"]
    ]

    melted = frame.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        value_vars=value_columns,
        var_name="date",
        value_name="confirmed_cases",
    )
    melted["date"] = pd.to_datetime(melted["date"], errors="coerce")

    grouped = (
        melted.groupby(["Country/Region", "date"], as_index=False)["confirmed_cases"]
        .sum()
        .rename(columns={"Country/Region": "country"})
    )

    if country:
        grouped = grouped[grouped["country"].str.lower() == country.lower()]

    return grouped.sort_values(["country", "date"])


def build_feature_frame(country: str | None = None) -> pd.DataFrame:
    frame = _get_features_df()
    if not frame.empty:
        if country:
            return frame[frame["country"].str.lower() == country.lower()]
        return frame

    timeseries = load_jhu_timeseries(country)
    if timeseries.empty:
        return timeseries

    timeseries["daily_new_cases"] = timeseries.groupby("country")["confirmed_cases"].diff().fillna(0.0)
    timeseries["case_velocity"] = timeseries.groupby("country")["daily_new_cases"].diff().fillna(0.0)
    timeseries["case_acceleration"] = timeseries.groupby("country")["case_velocity"].diff().fillna(0.0)
    timeseries["rolling_7d_cases"] = (
        timeseries.groupby("country")["daily_new_cases"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return timeseries


def build_graph_snapshot() -> dict[str, list[dict[str, str | float]]]:
    graph_path = PROCESSED_ROOT / "graph_snapshot.csv"
    if graph_path.exists():
        graph_frame = pd.read_csv(graph_path)
        node_ids = sorted(set(graph_frame["source"].tolist()) | set(graph_frame["target"].tolist()))
        nodes = [{"id": node, "label": node} for node in node_ids]
        edges = graph_frame[["source", "target", "weight"]].to_dict(orient="records")
        return {"nodes": nodes, "edges": edges}

    return {
        "nodes": [
            {"id": "IND", "label": "India"},
            {"id": "ARE", "label": "United Arab Emirates"},
            {"id": "BRA", "label": "Brazil"},
            {"id": "USA", "label": "United States"},
        ],
        "edges": [
            {"source": "IND", "target": "ARE", "weight": 0.72},
            {"source": "USA", "target": "BRA", "weight": 0.41},
            {"source": "BRA", "target": "USA", "weight": 0.53},
        ],
    }


def build_risk_map() -> dict[str, list[dict]]:
    """Compute data-driven outbreak risk scores from the most recent features per country."""
    global _RISK_MAP_CACHE
    if _RISK_MAP_CACHE is not None:
        return _RISK_MAP_CACHE

    features_path = PROCESSED_ROOT / "features_daily.csv"
    if not features_path.exists():
        return {"countries": []}

    frame = _get_features_df()
    if frame.empty:
        return {"countries": []}
    frame = frame.dropna(subset=["country"])

    # Take the latest row per country
    latest = frame.sort_values("date").groupby("country", as_index=False).last()

    def _risk_score(row: pd.Series) -> float:
        vax_gap = 1.0 - float(row.get("people_fully_vaccinated_per_hundred", 0) or 0) / 100.0
        # mobility: higher mobility → higher spread risk (normalise -100..+25 → 0..1)
        mobility_raw = float(row.get("mobility_index", 0.0) or 0.0)
        mobility_norm = min(max((mobility_raw + 100) / 125.0, 0.0), 1.0)
        # stringency: higher stringency → lower risk
        stringency = float(row.get("stringency_index", 50) or 50) / 100.0
        # case acceleration: clip and normalise
        accel = float(row.get("case_acceleration", 0) or 0)
        accel_norm = min(max(accel / 5000.0, -1.0), 1.0) * 0.5 + 0.5
        score = (
            0.30 * vax_gap
            + 0.25 * mobility_norm
            + 0.20 * (1.0 - stringency)
            + 0.25 * accel_norm
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def _risk_level(score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.55:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    result = []
    for _, row in latest.iterrows():
        score = _risk_score(row)
        result.append({
            "country": str(row["country"]),
            "risk_score": score,
            "risk_level": _risk_level(score),
            "vax_coverage": round(float(row.get("people_fully_vaccinated_per_hundred", 0) or 0), 1),
            "stringency": round(float(row.get("stringency_index", 0) or 0), 1),
            "daily_cases": max(round(float(row.get("daily_new_cases", 0) or 0), 0), 0),
            "rolling_7d": max(round(float(row.get("rolling_7d_cases", 0) or 0), 0), 0),
        })

    _RISK_MAP_CACHE = {"countries": result}
    return _RISK_MAP_CACHE
