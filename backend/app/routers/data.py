from fastapi import APIRouter, Query

from ..services.data_service import build_feature_frame, build_graph_snapshot, load_jhu_timeseries, build_risk_map

router = APIRouter()


@router.get("/timeseries")
def timeseries(
    country: str | None = Query(default=None),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> dict[str, list[dict[str, str | float]]]:
    frame = load_jhu_timeseries(country=country)

    if start_date:
        frame = frame[frame["date"] >= start_date]
    if end_date:
        frame = frame[frame["date"] <= end_date]

    payload = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    return {"rows": payload}


@router.get("/features")
def features(country: str | None = Query(default=None), date: str | None = Query(default=None)) -> dict[str, list[dict[str, str | float]]]:
    frame = build_feature_frame(country=country)
    if date:
        frame = frame[frame["date"] == date]

    frame = frame.tail(1000)
    payload = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    return {"rows": payload}


@router.get("/graph")
def graph(snapshot_date: str | None = Query(default=None)) -> dict[str, object]:
    graph = build_graph_snapshot()
    graph["snapshot_date"] = snapshot_date
    return graph


@router.get("/countries")
def countries() -> dict[str, list[str]]:
    frame = build_feature_frame()
    names = sorted(frame["country"].dropna().unique().tolist())
    return {"countries": names}


@router.get("/risk-map")
def risk_map() -> dict[str, list[dict]]:
    return build_risk_map()
