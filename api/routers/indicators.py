from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional
from api.services.market_service import MarketService
from api.services.indicator_service import IndicatorService
from api.models.schemas import IndicatorResponse, IndicatorType

router = APIRouter(prefix="/indicators", tags=["Indicators"])
market_service = MarketService()
indicator_service = IndicatorService()

@router.get("/rsi/{symbol}", response_model=IndicatorResponse)
def get_rsi(
    symbol: str, 
    interval: str = "5m", 
    period: int = 14,
    start_time: Optional[int] = None, 
    end_time: Optional[int] = None
):
    # Always fetch historical dataframe to support resampling (interval)
    df = market_service.get_historical_dataframe(symbol, interval, start_time, end_time)
    
    data = indicator_service.get_indicator(df, IndicatorType.RSI, period)
    
    # Filter result data by time range
    filtered_data = []
    for d in data:
        if start_time and d.time < start_time:
            continue
        if end_time and d.time > end_time:
            continue
        filtered_data.append(d)

    return IndicatorResponse(
        indicator="RSI",
        period=period,
        data=filtered_data
    )

@router.get("/{type}/{symbol}", response_model=IndicatorResponse)
def get_indicator_generic(
    symbol: str,
    type: str = Path(..., description="Indicator type: macd, bollinger, ema, sma"),
    interval: str = "5m",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
):
    try:
        indicator_type = IndicatorType(type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid indicator type: {type}")

    # Always fetch historical dataframe to support resampling (interval)
    df = market_service.get_historical_dataframe(symbol, interval, start_time, end_time)
    
    # Defaults
    period = 14
    if indicator_type == IndicatorType.EMA or indicator_type == IndicatorType.SMA:
        period = 20 # Default
    
    data = indicator_service.get_indicator(df, indicator_type, period)
    
    filtered_data = []
    for d in data:
        if start_time and d.time < start_time:
            continue
        if end_time and d.time > end_time:
            continue
        filtered_data.append(d)

    return IndicatorResponse(
        indicator=indicator_type.value,
        data=filtered_data
    )
