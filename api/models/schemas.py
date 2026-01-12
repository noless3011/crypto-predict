from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
from enum import Enum

# --- Enums ---
class TickerType(str, Enum):
    CRYPTO = "CRYPTO"
    STOCK = "STOCK"

class MarketDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"

class IndicatorType(str, Enum):
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER = "BOLLINGER"
    EMA = "EMA"
    SMA = "SMA"

# --- Shared Models ---
class Tickermetadata(BaseModel):
    symbol: str
    name: str
    type: TickerType
    exchange: str

class TickerSummary(BaseModel):
    symbol: str
    current_price: float
    change_24h: float
    volume_24h: float
    updated_at: int

class OHLCVPoint(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class HistoryResponse(BaseModel):
    symbol: str
    interval: str
    data: List[List[Union[int, float]]]  # [Time, Open, High, Low, Close, Volume]

# --- Indicator Models ---
class IndicatorDataPoint(BaseModel):
    time: int
    value: Optional[float] = None
    # For MACD
    macd: Optional[float] = None
    signal: Optional[float] = None
    histogram: Optional[float] = None
    # For Bollinger
    upper: Optional[float] = None
    middle: Optional[float] = None
    lower: Optional[float] = None

class IndicatorResponse(BaseModel):
    indicator: str
    period: Optional[int] = None
    data: List[IndicatorDataPoint]

# --- News Models ---
class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: int
    sentiment: Optional[str] = None

# --- Forecast Models ---
class ForecastPoint(BaseModel):
    time: int
    direction: MarketDirection
    confidence: float

class ForecastResponse(BaseModel):
    model_name: str
    forecast_data: List[ForecastPoint]
