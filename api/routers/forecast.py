from fastapi import APIRouter, Query
from api.services.prediction_service import PredictionService
from api.models.schemas import ForecastResponse

router = APIRouter(prefix="/forecast", tags=["Forecast"])
prediction_service = PredictionService()

from typing import List

@router.get("/{symbol}", response_model=List[ForecastResponse])
def get_forecast(symbol: str, days: int = 7):
    # Note: Currently supports next 24h prediction only due to model nature
    return prediction_service.forecast(symbol, days)
