from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from api.services.market_service import MarketService
from api.models.schemas import Tickermetadata, TickerSummary, HistoryResponse

router = APIRouter(prefix="/market", tags=["Market"])
market_service = MarketService()

@router.get("/tickers", response_model=List[Tickermetadata])
def get_tickers(keyword: Optional[str] = None):
    return market_service.search_tickers(keyword)

@router.get("/summary/{symbol}", response_model=TickerSummary)
def get_summary(symbol: str):
    summary = market_service.get_ticker_summary(symbol)
    if not summary:
        raise HTTPException(status_code=404, detail="Ticker not found")
    return summary

@router.get("/history/{symbol}", response_model=HistoryResponse)
def get_history(
    symbol: str, 
    interval: str = "5m", 
    start_time: Optional[int] = None, 
    end_time: Optional[int] = None
):
    data = market_service.get_historical_data(symbol, interval, start_time, end_time)
    # data is List[List[float, ...]]
    return HistoryResponse(
        symbol=symbol,
        interval=interval,
        data=data
    )
