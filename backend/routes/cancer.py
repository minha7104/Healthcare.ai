from __future__ import annotations

from fastapi import APIRouter

from schemas import CancerRequest, PredictionResponse
from utils.predictor_service import PredictorService


def get_router(predictor: PredictorService) -> APIRouter:
    router = APIRouter(prefix="/predict", tags=["cancer"])

    @router.post("/cancer", response_model=PredictionResponse)
    def predict_cancer(request: CancerRequest) -> PredictionResponse:
        payload = request.model_dump(by_alias=True)
        prediction, probability, top_features = predictor.predict_cancer(payload)
        return PredictionResponse(
            prediction=prediction, probability=probability, top_features=top_features
        )

    return router
