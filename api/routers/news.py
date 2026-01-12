from fastapi import APIRouter, Query
from typing import List, Optional
from api.services.news_service import NewsService
from api.models.schemas import NewsItem

router = APIRouter(prefix="/news", tags=["News"])
news_service = NewsService()

@router.get("/{symbol}", response_model=List[NewsItem])
def get_news(
    symbol: str, 
    start_time: Optional[int] = None, 
    end_time: Optional[int] = None, 
    limit: int = 20
):
    return news_service.get_news(symbol, start_time, end_time, limit)
