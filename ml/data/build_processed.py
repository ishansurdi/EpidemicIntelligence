from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.feature_engine import add_temporal_features, build_outbreak_label, melt_jhu_confirmed
from ml.data.loaders import load_google_mobility, load_jhu_confirmed, load_owid_table

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"


def _prepare_owid_feature_table() -> pd.DataFrame:
    cases = load_owid_table("cases_deaths")
    vacc = load_owid_table("vaccinations_global")
    testing = load_owid_table("testing")
    policy = load_owid_table("oxcgrt_policy")

    cases = cases[["country", "date", "new_deaths", "total_deaths", "new_cases_per_million"]].copy()
    vacc = vacc[
        [
            "country",
            "date",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
        ]
    ].copy()
    testing = testing[["country", "date", "new_tests_per_thousand_7day_smoothed"]].copy()
    policy = policy[["country", "date", "stringency_index"]].copy()

    merged = cases.merge(vacc, how="left", on=["country", "date"])
    merged = merged.merge(testing, how="left", on=["country", "date"])
    merged = merged.merge(policy, how="left", on=["country", "date"])
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged


def _prepare_mobility_table() -> pd.DataFrame:
    mobility = load_google_mobility().copy()
    if not {"country", "date", "place", "trend"}.issubset(mobility.columns):
        return pd.DataFrame(columns=["country", "date", "mobility_index"])

    mobility["trend"] = pd.to_numeric(mobility["trend"], errors="coerce")
    pivot = (
        mobility.pivot_table(index=["country", "date"], columns="place", values="trend", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    place_cols = [col for col in pivot.columns if col not in ["country", "date"]]
    pivot["mobility_index"] = pivot[place_cols].mean(axis=1, skipna=True)
    pivot["date"] = pd.to_datetime(pivot["date"], errors="coerce")
    return pivot[["country", "date", "mobility_index"]]


def _fill_countrywise(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.sort_values(["country", "date"]).copy()
    for col in columns:
        out[col] = out.groupby("country")[col].transform(lambda s: s.ffill().bfill())
        out[col] = out[col].fillna(0.0)
    return out


def _build_graph_snapshot(features: pd.DataFrame, k_neighbors: int = 5) -> pd.DataFrame:
    latest = (
        features.sort_values("date")
        .groupby("country", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    graph_features = latest[
        [
            "country",
            "rolling_7d_cases",
            "new_cases_per_million",
            "mobility_index",
            "people_fully_vaccinated_per_hundred",
            "stringency_index",
        ]
    ].copy()
    graph_features = graph_features.fillna(0.0)

    values = graph_features.drop(columns=["country"]).to_numpy(dtype=float)
    countries = graph_features["country"].tolist()

    edges: list[dict[str, object]] = []
    for idx, source in enumerate(countries):
        row = values[idx]
        distances = np.sqrt(((values - row) ** 2).sum(axis=1))
        neighbor_indices = np.argsort(distances)
        picked = 0
        for ni in neighbor_indices:
            if ni == idx:
                continue
            weight = float(np.exp(-distances[ni] / 50.0))
            edges.append(
                {
                    "source": source,
                    "target": countries[ni],
                    "weight": round(weight, 6),
                    "snapshot_date": pd.Timestamp.now("UTC").date().isoformat(),
                }
            )
            picked += 1
            if picked >= k_neighbors:
                break

    return pd.DataFrame(edges)


def build_processed_outputs() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    jhu = load_jhu_confirmed()
    timeseries = add_temporal_features(melt_jhu_confirmed(jhu))
    timeseries = build_outbreak_label(timeseries)

    owid = _prepare_owid_feature_table()
    mobility = _prepare_mobility_table()

    features = timeseries.merge(owid, how="left", on=["country", "date"])
    features = features.merge(mobility, how="left", on=["country", "date"])

    numeric_cols = [
        "new_deaths",
        "total_deaths",
        "new_cases_per_million",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "new_tests_per_thousand_7day_smoothed",
        "stringency_index",
        "mobility_index",
    ]
    for col in numeric_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    features = _fill_countrywise(features, [col for col in numeric_cols if col in features.columns])

    graph = _build_graph_snapshot(features)

    base_timeseries = features[
        ["country", "date", "confirmed_cases", "daily_new_cases"]
    ].copy()
    base_timeseries["date"] = base_timeseries["date"].dt.strftime("%Y-%m-%d")

    features["date"] = features["date"].dt.strftime("%Y-%m-%d")

    base_timeseries.to_csv(PROCESSED_ROOT / "timeseries_daily.csv", index=False)
    features.to_csv(PROCESSED_ROOT / "features_daily.csv", index=False)
    graph.to_csv(PROCESSED_ROOT / "graph_snapshot.csv", index=False)

    print(f"Saved: {PROCESSED_ROOT / 'timeseries_daily.csv'} ({len(base_timeseries)} rows)")
    print(f"Saved: {PROCESSED_ROOT / 'features_daily.csv'} ({len(features)} rows)")
    print(f"Saved: {PROCESSED_ROOT / 'graph_snapshot.csv'} ({len(graph)} rows)")


if __name__ == "__main__":
    build_processed_outputs()