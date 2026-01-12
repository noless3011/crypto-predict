# Crypto Prediction API

## Overview
This API provides market data, technical indicators, news, and AI-powered price forecasts for cryptocurrency tickers. It connects to a ClickHouse database for historical data and uses a pre-trained LSTM model for predictions.

## Setup

1.  **Dependencies**: Ensure you have the required Python packages installed.
    ```bash
    pip install fastapi uvicorn clickhouse-connect pandas numpy torch scikit-learn
    ```

2.  **Configuration**: 
    - Database credentials are configured in `api/config.py`.
    - The API expects `best_crypto_lstm.pth` to be present in the project root (`d:\ProgrammingProjects\data_science\BTL\`).

## Running the API

Run the following command from the project root (`d:\ProgrammingProjects\data_science\BTL\`):

```bash
python -m api.main
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload
```

## Documentation

Once running, access the interactive API docs at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Endpoints

### Market Data
- `GET /api/v1/market/tickers`: Search/List tickers.
- `GET /api/v1/market/summary/{symbol}`: Get 24h summary.
- `GET /api/v1/market/history/{symbol}`: Get OHLCV history.

### Indicators
- `GET /api/v1/indicators/rsi/{symbol}`: Get RSI.
- `GET /api/v1/indicators/{type}/{symbol}`: Get generic indicators (MACD, Bollinger, EMA, SMA).

### News
- `GET /api/v1/news/{symbol}`: Get news (currently global/filtered).

### Forecast
- `GET /api/v1/forecast/{symbol}`: Get AI prediction for the next 24 hours.
