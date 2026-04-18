from __future__ import annotations

from fastapi import APIRouter

from schemas import CardioRequest, PredictionResponse
from utils.predictor_service import PredictorService


def get_router(predictor: PredictorService) -> APIRouter:
    router = APIRouter(prefix="/predict", tags=["cardio"])

    @router.post("/cardio", response_model=PredictionResponse)
    def predict_cardio(request: CardioRequest) -> PredictionResponse:
        payload = request.model_dump(by_alias=True)
        prediction, probability, top_features = predictor.predict_cardio(payload)
        return PredictionResponse(
            prediction=prediction, probability=probability, top_features=top_features
        )

    return router
