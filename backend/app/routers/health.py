from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "oracle-epidemic-api",
        "phase": "phase-1-foundation",
    }
