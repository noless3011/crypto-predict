import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


class IdentityScaler:
    def fit(self, x):
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def convert_numpy_types(data):
    """
    Convert numpy types to native Python types for JSON serialization.
    Handles list of dicts (DataFrame records) or single dict.
    Also normalizes array-like fields that may come as strings.
    """
    # Fields that should be arrays
    array_fields = {"categories", "tags", "authors"}

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        item[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        item[key] = float(value)
                    elif isinstance(value, np.bool_):
                        item[key] = bool(value)
                    elif pd.isna(value):
                        item[key] = None
                    # Normalize array fields
                    elif key in array_fields and isinstance(value, str):
                        # Try to parse as array or split by common delimiters
                        if value.startswith("[") and value.endswith("]"):
                            try:
                                item[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                item[key] = [
                                    v.strip()
                                    for v in value.strip("[]").split(",")
                                    if v.strip()
                                ]
                        else:
                            # Split by common delimiters
                            item[key] = [
                                v.strip() for v in value.split(",") if v.strip()
                            ]
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                data[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                data[key] = float(value)
            elif isinstance(value, np.bool_):
                data[key] = bool(value)
            elif pd.isna(value):
                data[key] = None
            # Normalize array fields
            elif key in array_fields and isinstance(value, str):
                if value.startswith("[") and value.endswith("]"):
                    try:
                        data[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        data[key] = [
                            v.strip() for v in value.strip("[]").split(",") if v.strip()
                        ]
                else:
                    data[key] = [v.strip() for v in value.split(",") if v.strip()]
    return data


# Add parent directory to path so we can import data.data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml_service import ml_service
from data.data import ClickhouseHelper

app = FastAPI(title="Crypto Prediction API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_TICKER = "BTCUSDT"


def get_ohlcv_data(
    ticker: str = DEFAULT_TICKER,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    limit: int | None = None,
):
    """
    Fetch OHLCV data from the database
    If limit is specified, fetches the latest N rows (more efficient)
    Otherwise fetches data in the given time range
    """
    from data.data import Interval

    # Use get_latest_data if limit is specified for efficiency
    if limit is not None:
        df = ClickhouseHelper.get_latest_data(
            ticker=ticker,
            limit=limit,
            time_end=time_end,
            interval=Interval.FIVE_MINUTES,
            verbose=False,
        )
    else:
        df = ClickhouseHelper.get_data_between(
            ticker=ticker,
            time_start=time_start,
            time_end=time_end,
            interval=Interval.FIVE_MINUTES,
            verbose=False,
        )

    if df.empty:
        return pd.DataFrame()

    # Rename columns to match backend/frontend expectations
    df = df.rename(
        columns={
            "openTime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Ensure Date is datetime (handle ms timestamp)
    if "Date" in df.columns:
        # Check if it looks like a timestamp (numeric)
        if pd.api.types.is_numeric_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        else:
            df["Date"] = pd.to_datetime(df["Date"])

    # Convert all numeric columns from Decimal to float for numpy compatibility
    # ClickHouse returns Decimal types which don't work with numpy operations
    for col in df.columns:
        if col != "Date" and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@app.get("/api/history")
def get_history(
    start: Optional[str] = None, end: Optional[str] = None, ticker: str = DEFAULT_TICKER
):
    try:
        # Parse start/end dates if provided
        time_start = None
        time_end = None

        if start:
            time_start = pd.to_datetime(start)
            if time_start.tz is not None:
                time_start = time_start.tz_localize(None)

        if end:
            time_end = pd.to_datetime(end)
            if time_end.tz is not None:
                time_end = time_end.tz_localize(None)

        df = get_ohlcv_data(ticker=ticker, time_start=time_start, time_end=time_end)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="Data not found in database.",
            )

        # Convert to dictionary/records
        # lightweight-charts expects: time (unix timestamp or yyyy-mm-dd), open, high, low, close
        # JSON compliance: replace NaN/Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        result = df.where(pd.notnull(df), None).to_dict(orient="records")

        # Convert numpy types to native Python types for JSON serialization
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/meta")
def get_history_meta(ticker: str = DEFAULT_TICKER):
    try:
        df = get_ohlcv_data(ticker=ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail="Data not found")

        # Ensure sorted
        # df = df.sort_values('Date') # Assuming CSV is sorted or get_ohlcv_data handles it?
        # get_ohlcv_data doesn't explicit sort, but usually CSV is time order.
        # Let's rely on min/max

        start_date = df["Date"].min()
        end_date = df["Date"].max()

        return {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "count": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indicators")
def get_indicators(
    start: Optional[str] = None, end: Optional[str] = None, ticker: str = DEFAULT_TICKER
):
    try:
        # Parse start/end dates if provided
        time_start = None
        time_end = None

        if start:
            time_start = pd.to_datetime(start)
            if time_start.tz is not None:
                time_start = time_start.tz_localize(None)

        if end:
            time_end = pd.to_datetime(end)
            if time_end.tz is not None:
                time_end = time_end.tz_localize(None)

        df = get_ohlcv_data(ticker=ticker, time_start=time_start, time_end=time_end)
        if df.empty:
            raise HTTPException(status_code=404, detail="Data not found")
        # Calculate indicators using ML Service logic
        # Note: This adds columns like RSI, MACD, etc.
        df_ind = ml_service.add_technical_indicators(df)
        # Ensure datetime
        df_ind["Date"] = pd.to_datetime(df_ind["Date"])

        # Sort strictly by time
        df_ind = df_ind.sort_values("Date")

        # DROP duplicate timestamps (required for charts)
        df_ind = df_ind.drop_duplicates(subset="Date", keep="last")

        # Reset index
        df_ind = df_ind.reset_index(drop=True)

        # Convert NaN to None for JSON
        df_ind = df_ind.where(pd.notnull(df_ind), None)

        result = df_ind.to_dict(orient="records")

        # Convert numpy types to native Python types for JSON serialization
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
def get_news(
    days: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
):
    try:
        # Determine time range
        # If start/end are provided, use them. Otherwise fall back to days parameter
        if start or end:
            # Use explicit date range
            end_time = datetime.now()
            start_time = None

            if end:
                try:
                    ts_end = pd.to_datetime(end)
                    if ts_end.tz is not None:
                        ts_end = ts_end.tz_localize(None)
                    end_time = ts_end
                except Exception as e:
                    print(f"Error parsing end date: {e}")
                    pass

            if start:
                try:
                    ts_start = pd.to_datetime(start)
                    if ts_start.tz is not None:
                        ts_start = ts_start.tz_localize(None)
                    start_time = ts_start
                except Exception as e:
                    print(f"Error parsing start date: {e}")
                    pass

            # If start wasn't provided, fall back to days from end
            if start_time is None:
                days_to_use = days if days is not None else 7
                start_time = end_time - timedelta(days=days_to_use)
        else:
            # No explicit dates, use days parameter
            days_to_use = days if days is not None else 7
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_to_use)

        print(
            f"Fetching news between {start_time} (ts: {start_time.timestamp()}) and {end_time} (ts: {end_time.timestamp()})"
        )

        # Get total count first for pagination metadata
        total = ClickhouseHelper.get_news_count(
            time_start=start_time, time_end=end_time, verbose=True
        )
        print(f"Total news items: {total}")

        if total == 0:
            return {
                "data": [],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": 0,
                    "totalPages": 0,
                },
            }

        # Calculate pagination
        total_pages = (total + limit - 1) // limit
        offset = (page - 1) * limit

        # Fetch only the page we need from database
        df = ClickhouseHelper.get_news_between(
            time_start=start_time,
            time_end=end_time,
            limit=limit,
            offset=offset,
            verbose=True,
        )
        print(f"Fetched {len(df)} news items for page {page}")

        # Convert DataFrame to dict and handle numpy types for JSON serialization
        news_data = df.to_dict(orient="records")
        convert_numpy_types(news_data)

        return {
            "data": news_data,
            "pagination": {
                "page": int(page),
                "limit": int(limit),
                "total": int(total),
                "totalPages": int(total_pages),
            },
        }
    except Exception as e:
        print(f"News Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "data": [],
            "pagination": {"page": page, "limit": limit, "total": 0, "totalPages": 0},
        }


@app.get("/api/tickers")
def list_tickers():
    """
    Get list of available tickers from the database
    """
    try:
        from data.data import Interval

        df = ClickhouseHelper.list_ticker(interval=Interval.FIVE_MINUTES, verbose=False)
        if df.empty:
            return []

        tickers = df["ticker"].tolist()
        return convert_numpy_types(tickers)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def list_models():
    return ml_service.get_available_models()


@app.post("/api/predict")
def predict(
    model_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prediction_hours: int = 5,
    ticker: str = DEFAULT_TICKER,
):
    try:
        # Validate prediction_hours is divisible by 5
        if prediction_hours <= 0 or prediction_hours % 5 != 0:
            raise HTTPException(
                status_code=400,
                detail="prediction_hours must be a positive integer divisible by 5",
            )

        # Parse end date if provided
        time_end = None
        if end:
            time_end = pd.to_datetime(end)
            if time_end.tz is not None:
                time_end = time_end.tz_localize(None)

        # For predictions, we always need enough historical data (5000 rows)
        # for preprocessing (indicators + lookback window)
        # Use limit parameter to efficiently fetch only the last 5000 rows
        df_recent = get_ohlcv_data(ticker=ticker, time_end=time_end, limit=5000)

        if df_recent.empty:
            raise HTTPException(
                status_code=404, detail="No data available for prediction"
            )

        predictions = ml_service.make_prediction(
            model_name, df_recent, prediction_hours=prediction_hours
        )

        # Return predictions with future timestamps
        last_date = df_recent.iloc[-1]["Date"]
        response = []
        for i, price in enumerate(predictions):
            future_date = last_date + timedelta(hours=i + 1)  # Assuming hourly

            # Handle Inf/Nan for JSON serialization
            val = float(price)
            if np.isinf(val) or np.isnan(val):
                val = None

            response.append({"time": future_date.isoformat(), "value": val})

        return response
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Enable dummy mode to load models with custom Lambda layers",
    )
    # Parse known args to handle cases where other args might be passed (though unlikely here)
    args, unknown = parser.parse_known_args()

    if args.dummy:
        ml_service.enable_dummy_mode()

    uvicorn.run(app, host="0.0.0.0", port=8000)
