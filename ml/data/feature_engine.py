import pandas as pd


def melt_jhu_confirmed(frame: pd.DataFrame) -> pd.DataFrame:
    fixed_cols = ["Province/State", "Country/Region", "Lat", "Long"]
    date_cols = [col for col in frame.columns if col not in fixed_cols]

    melted = frame.melt(
        id_vars=fixed_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="confirmed_cases",
    )
    melted["date"] = pd.to_datetime(melted["date"], format="%m/%d/%y", errors="coerce")

    grouped = (
        melted.groupby(["Country/Region", "date"], as_index=False)["confirmed_cases"]
        .sum()
        .rename(columns={"Country/Region": "country"})
    )
    return grouped.sort_values(["country", "date"])


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["daily_new_cases"] = out.groupby("country")["confirmed_cases"].diff().fillna(0.0)
    out["case_velocity"] = out.groupby("country")["daily_new_cases"].diff().fillna(0.0)
    out["case_acceleration"] = out.groupby("country")["case_velocity"].diff().fillna(0.0)
    out["case_jerk"] = out.groupby("country")["case_acceleration"].diff().fillna(0.0)
    out["rolling_7d_cases"] = (
        out.groupby("country")["daily_new_cases"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["dow"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    return out


def build_outbreak_label(frame: pd.DataFrame, growth_threshold: float = 0.5) -> pd.DataFrame:
    out = frame.sort_values(["country", "date"]).copy()
    out["past_14d_cases"] = (
        out.groupby("country")["daily_new_cases"]
        .rolling(14, min_periods=7)
        .sum()
        .reset_index(level=0, drop=True)
    )
    out["past_prev_14d_cases"] = (
        out.groupby("country")["daily_new_cases"]
        .transform(lambda s: s.shift(14).rolling(14, min_periods=7).sum())
    )
    out["recent_growth_14d"] = ((out["past_14d_cases"] + 1.0) / (out["past_prev_14d_cases"] + 1.0)) - 1.0
    out["future_14d_cases"] = (
        out.groupby("country")["daily_new_cases"]
        .transform(lambda s: s.shift(-1).rolling(14, min_periods=7).sum())
    )
    out["future_growth_14d"] = ((out["future_14d_cases"] + 1.0) / (out["past_14d_cases"] + 1.0)) - 1.0
    out["recent_growth_14d"] = out["recent_growth_14d"].replace([float("inf"), float("-inf")], pd.NA)
    out["future_growth_14d"] = out["future_growth_14d"].replace([float("inf"), float("-inf")], pd.NA)
    out["outbreak_label"] = (out["future_growth_14d"] >= growth_threshold).astype(int)
    return out
