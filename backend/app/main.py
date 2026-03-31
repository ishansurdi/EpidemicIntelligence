from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, data, predict, cascade, interpret

app = FastAPI(
    title="ORACLE Epidemic Intelligence API",
    version="0.1.0",
    description="Prediction and outbreak-risk API for epidemic cascade intelligence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(cascade.router, prefix="/api/v1/cascade", tags=["cascade"])
app.include_router(interpret.router, prefix="/api/v1/interpret", tags=["interpret"])
