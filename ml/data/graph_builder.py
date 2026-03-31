import pandas as pd


def build_country_graph(mobility: pd.DataFrame) -> pd.DataFrame:
    expected_cols = {"country", "date", "trend"}
    if not expected_cols.issubset(set(mobility.columns)):
        return pd.DataFrame(columns=["source", "target", "weight", "date"])

    scored = (
        mobility.dropna(subset=["country", "date", "trend"])
        .groupby(["country", "date"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )

    # Phase 1 placeholder: self-loop edges from available mobility activity volume.
    graph = scored.rename(columns={"country": "source"})
    graph["target"] = graph["source"]
    return graph[["source", "target", "weight", "date"]]
