from fastapi import APIRouter

from ..models.schemas import AttentionMapResponse, FeatureImportanceResponse
from ..services.interpret_service import build_attention_map, build_feature_importance

router = APIRouter()


@router.get("/attention-map", response_model=AttentionMapResponse)
def attention_map() -> AttentionMapResponse:
    return build_attention_map()


@router.get("/feature-importance/{region_id}", response_model=FeatureImportanceResponse)
def feature_importance(region_id: str) -> FeatureImportanceResponse:
    return build_feature_importance(region_id)
