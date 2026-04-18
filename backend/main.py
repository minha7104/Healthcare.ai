from __future__ import annotations

from fastapi import FastAPI

from routes.cancer import get_router as cancer_router
from routes.cardio import get_router as cardio_router
from routes.diabetes import get_router as diabetes_router
from utils.predictor_service import PredictorService

app = FastAPI(title="Healthcare AI API", version="1.0.0")
predictor = PredictorService()
app.include_router(cancer_router(predictor))
app.include_router(diabetes_router(predictor))
app.include_router(cardio_router(predictor))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
