from fastapi import APIRouter

from ..models.schemas import CascadeTraceResponse
from ..services.interpret_service import build_cascade_trace

router = APIRouter()


@router.get("/trace/{region_id}", response_model=CascadeTraceResponse)
def trace(region_id: str) -> CascadeTraceResponse:
    return build_cascade_trace(region_id)
