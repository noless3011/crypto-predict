import math
import os
import time
import warnings
from datetime import datetime
from enum import Enum

import clickhouse_connect
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# --- PARTITION CONFIGURATION ---
PART_ID = 2  # 0 = Test Mode (2 tickers). 1 to TOTAL_PARTS = Actual Data Split.
TOTAL_PARTS = 8  # Split the 670 tickers into this many chunks to fit Kaggle 20GB limit.

OUTPUT_DIR = "/kaggle/working/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKER_CSV_PATH = "/kaggle/input/tickers/ticker.csv"

# Database Connection
client = clickhouse_connect.get_client(
    host="128.199.197.160",
    port=32014,
    username="readonly_user",
    password="grA5SfKxOQ",
    database="default",
)


# ==========================================
# 2. FEATURE ENGINEERING LOGIC
# ==========================================
def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int = 20
) -> pd.Series:
    const = 1.0 / (4.0 * np.log(2.0))
    log_hl = np.log(high / low) ** 2
    return np.sqrt(const * log_hl.rolling(window=window).mean())


def add_features_and_clean(
    df: pd.DataFrame, news_df: pd.DataFrame = None
) -> pd.DataFrame:
    if df.empty:
        return df

    # Sort and Time Setup
    df = df.sort_values("openTime").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["openTime"], unit="ms")

    # --- A. Technical Features ---
    # 1. Log Returns
    for n in [1, 3, 6, 12, 24]:
        df[f"log_ret_{n}"] = np.log(df["close"] / df["close"].shift(n))

    # 2. Rolling Z-Score (Normalization)
    window_z = 100
    r_mean = df["close"].rolling(window=window_z).mean()
    r_std = df["close"].rolling(window=window_z).std()
    df["price_zscore"] = (df["close"] - r_mean) / (r_std + 1e-8)

    # 3. Volume / Liquidity
    df["log_volume"] = np.log(df["volume"] + 1)
    df["rvol"] = df["volume"] / (df["volume"].rolling(50).mean() + 1e-8)

    # VWAP Distance
    # Note: Using Typical Price * Volume approximation since quoteAssetVolume is removed
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(288).sum() / (
        df["volume"].rolling(288).sum() + 1e-8
    )
    df["dist_vwap"] = (df["close"] - vwap) / vwap

    # 4. Volatility & Shadows
    df["vol_parkinson"] = _calculate_parkinson_volatility(df["high"], df["low"])
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range

    # 5. Oscillators
    df["rsi_14"] = _calculate_rsi(df["close"], 14) / 100.0

    # 6. Time Embeddings (24/7 Crypto market = 1440 mins)
    minutes = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    df["time_cos"] = np.cos(2 * np.pi * minutes / 1440)

    # --- B. Merge News Sentiment ---
    if news_df is not None and not news_df.empty:
        df = pd.merge_asof(
            df,
            news_df,
            left_on="datetime",
            right_on="datetime",
            direction="backward",
            tolerance=pd.Timedelta("4h"),
        )
        df["global_sentiment_score"] = df["global_sentiment_score"].fillna(0)
        df["global_sentiment_ewma"] = df["global_sentiment_ewma"].fillna(0)

    # --- C. Cleanup ---
    # Dropping columns that are no longer needed
    cols_to_drop = ["closeTime", "numOfTrades", "expected_openTime", "tp"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Drop rows with NaNs caused by lagging
    df = df.dropna().reset_index(drop=True)

    return df


# ==========================================
# 3. NEWS PROCESSING (Global Sentiment)
# ==========================================
def process_global_news():
    print("Fetching and processing Global News...")
    query = "SELECT publishedOn, sentiment FROM news ORDER BY publishedOn"
    try:
        result = client.query(query)
        news_df = pd.DataFrame(result.result_rows, columns=result.column_names)

        news_df["datetime"] = pd.to_datetime(news_df["publishedOn"], unit="s")
        sentiment_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        news_df["val"] = news_df["sentiment"].map(sentiment_map).fillna(0)

        global_news = (
            news_df.set_index("datetime").resample("5min")["val"].mean().reset_index()
        )
        global_news["global_sentiment_ewma"] = global_news["val"].ewm(span=12).mean()
        global_news.rename(columns={"val": "global_sentiment_score"}, inplace=True)

        print(f"Processed {len(global_news)} global news intervals.")
        return global_news
    except Exception as e:
        print(f"Warning: Could not fetch/process news. {e}")
        return pd.DataFrame()


# ==========================================
# 4. DATA FETCHING (Optimized & Pruned)
# ==========================================
def get_clean_ticker_data(ticker_name: str, verbose=False, max_retries=5):
    conditions = [f"ticker = '{ticker_name}'"]

    query = f"""
    SELECT openTime, open, high, low, close, volume, closeTime, numOfTrades
    FROM future_kline_5m
    WHERE {" AND ".join(conditions)}
    ORDER BY openTime
    """

    attempt = 0
    base_wait_seconds = 300  # 5 minutes

    while attempt <= max_retries:
        try:
            # Execute Query
            result = client.query(query)
            df = pd.DataFrame(result.result_rows, columns=result.column_names)

            if df.empty:
                return pd.DataFrame()

            # --- Data Cleaning ---
            cols = ["open", "high", "low", "close", "volume", "numOfTrades"]
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

            # Forward Fill
            df = df.replace(0, np.nan).ffill().fillna(0)

            return df

        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                print(
                    f"❌ Failed to fetch {ticker_name} after {max_retries} attempts. Error: {e}"
                )
                return pd.DataFrame()

            # Calculate wait time: 5m, 10m, 15m...
            wait_time = base_wait_seconds * attempt
            print(f"⚠️ Query failed for {ticker_name}. Error: {e}")
            print(
                f"   Pausing for {wait_time / 60:.0f} minutes before retry {attempt}/{max_retries}..."
            )
            time.sleep(wait_time)


# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Load Tickers
    if not os.path.exists(TICKER_CSV_PATH):
        print(f"Error: {TICKER_CSV_PATH} not found.")
        exit()

    ticker_df = pd.read_csv(TICKER_CSV_PATH)
    all_tickers = ticker_df["ticker"].astype(str).unique().tolist()
    all_tickers.sort()  # Ensure deterministic order for partitioning

    total_tickers = len(all_tickers)
    print(f"Total Tickers found: {total_tickers}")

    # 2. Partition Logic
    tickers_to_process = []

    if PART_ID == 0:
        print(f"\n--- [TEST MODE] Running on first 2 tickers only ---")
        tickers_to_process = all_tickers[:2]
    else:
        if PART_ID > TOTAL_PARTS:
            print(
                f"Error: PART_ID ({PART_ID}) cannot be greater than TOTAL_PARTS ({TOTAL_PARTS})"
            )
            exit()

        chunk_size = math.ceil(total_tickers / TOTAL_PARTS)
        start_idx = (PART_ID - 1) * chunk_size
        end_idx = min(start_idx + chunk_size, total_tickers)

        tickers_to_process = all_tickers[start_idx:end_idx]
        print(f"\n--- [PRODUCTION MODE] Processing Part {PART_ID}/{TOTAL_PARTS} ---")
        print(
            f"Processing index {start_idx} to {end_idx} ({len(tickers_to_process)} tickers)"
        )

    # 3. Process Global News
    global_news_df = process_global_news()

    # 4. Processing Loop
    success_count = 0

    for ticker in tqdm(tickers_to_process, desc="Processing Tickers"):
        try:
            raw_df = get_clean_ticker_data(ticker)
            if raw_df.empty or len(raw_df) < 500:
                continue

            processed_df = add_features_and_clean(raw_df, global_news_df)
            processed_df["ticker"] = ticker

            save_path = os.path.join(OUTPUT_DIR, f"{ticker}.parquet")
            processed_df.to_parquet(save_path, compression="snappy")

            success_count += 1

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    print(
        f"\nProcessing Complete! {success_count}/{len(tickers_to_process)} tickers saved to {OUTPUT_DIR}"
    )
