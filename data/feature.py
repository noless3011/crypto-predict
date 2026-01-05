import os
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Assuming df_clean exists in your project directory
try:
    from df_clean import process_ticker
except ImportError:
    print("Warning: df_clean module not found. Please ensure it exists.")

    def process_ticker(ticker, verbose=False):
        return None


# --- CONFIGURATION ---
WINDOW_SIZE = 60
PREDICT_STEPS = 1


class FeatureType(str, Enum):
    PRICE = "price"  # Returns, Ranges (No raw prices)
    VOLUME = "volume"  # Log-transformed volume features
    ORDER_FLOW = "order_flow"
    TECHNICAL = "technical"  # RSI, Normalized MACD, BB Position
    MOMENTUM = "momentum"
    VOLUME_PRICE = "volume_price"
    MICROSTRUCTURE = "microstructure"
    TIME = "time"
    STATISTICAL = "statistical"  # Skew, Kurtosis


# --- HELPER: DATA CLEANING ---
def ensure_numeric_data(df):
    """Ensures basic OHLCV columns are numeric floats."""
    cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quoteAssetVolume",
        "numOfTrades",
        "takerBuyBaseAssetVolume",
    ]

    existing_cols = [c for c in cols if c in df.columns]
    for c in existing_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_infinite_values(df):
    """Replaces infinity with NaN to prevent model crashes."""
    return df.replace([np.inf, -np.inf], np.nan)


# --- FEATURE FUNCTIONS ---
def calculate_price_features(df):
    """Stationary price-based features (Returns & Ranges)"""
    # Normalized High-Low Range (Volatility proxy)
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Close-Open Return (Intraday momentum)
    df["co_return"] = (df["close"] - df["open"]) / df["open"]

    for window in [12, 36, 144]:
        # Rolling Mean/Std of RETURNS (not prices)
        df[f"rolling_mean_return_{window}"] = (
            df["log_return"].rolling(window=window).mean()
        )
        df[f"rolling_volatility_{window}"] = (
            df["log_return"].rolling(window=window).std()
        )

    return df


def calculate_volume_features(df):
    """Volume features (Log-transformed to fix skew)"""
    # Log transform is crucial for Volume to squash outliers
    df["volume_log"] = np.log1p(df["volume"])

    # Volume Change (Stationary)
    df["volume_pct_change"] = df["volume"].pct_change()

    # Relative Volume (Z-Score like)
    vol_mean = df["volume"].rolling(window=144).mean()
    df["volume_rel_ratio"] = df["volume"] / (vol_mean + 1e-8)

    return df


def calculate_order_flow_features(df):
    """Order flow ratios"""
    # Taker Buy Ratio (Buying Pressure) - naturally bounded 0-1
    df["taker_buy_ratio"] = df["takerBuyBaseAssetVolume"] / (
        df["quoteAssetVolume"] + 1e-8
    )
    return df


def calculate_technical_indicators(df):
    """RSI, Normalized MACD, Bollinger Position"""
    # RSI (Bounded 0-100)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # Normalize RSI to 0-1 for Neural Networks
    df["rsi_14"] = df["rsi_14"] / 100.0

    # NORMALIZED MACD (Crucial fix: Divide by Close to make it stationary)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_raw = ema_12 - ema_26
    df["macd_pct"] = macd_raw / df["close"]  # Now it's a percentage

    # Signal line on the normalized MACD
    df["macd_signal_pct"] = df["macd_pct"].ewm(span=9, adjust=False).mean()
    df["macd_hist_pct"] = df["macd_pct"] - df["macd_signal_pct"]

    # Bollinger Band Position (Bounded 0-1 approx)
    for window in [20, 60]:
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()

        # (Price - Lower) / (Upper - Lower)
        upper = rolling_mean + (2 * rolling_std)
        lower = rolling_mean - (2 * rolling_std)
        df[f"bb_position_{window}"] = (df["close"] - lower) / ((upper - lower) + 1e-8)

    return df


def calculate_momentum_features(df):
    """Momentum based on returns, not raw prices"""
    # ROC (Rate of Change)
    for period in [12, 36, 144]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    # Price position relative to MA (Percentage)
    for window in [12, 36, 144]:
        ma = df["close"].rolling(window=window).mean()
        df[f"dist_to_ma_{window}"] = (df["close"] - ma) / (ma + 1e-8)

    return df


def calculate_volume_price_features(df):
    """Volume-Weighted features"""
    # VWAP Deviation
    for window in [12, 36]:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vol_sum = df["volume"].rolling(window=window).sum() + 1e-8
        vp_sum = (typical_price * df["volume"]).rolling(window=window).sum()
        vwap = vp_sum / vol_sum

        # Percentage distance from VWAP
        df[f"vwap_dev_{window}"] = (df["close"] - vwap) / (vwap + 1e-8)

    return df


def calculate_microstructure_features(df):
    """Liquidity and Trade Count (Log transformed)"""
    # Log of Trade Count
    df["trades_log"] = np.log1p(df["numOfTrades"])

    # Average Trade Size (Quote Volume / Trades)
    avg_trade_val = df["quoteAssetVolume"] / (df["numOfTrades"] + 1e-8)
    df["avg_trade_val_log"] = np.log1p(avg_trade_val)

    return df


def calculate_time_features(df):
    """Cyclical time encoding (Sine/Cosine)"""
    if not np.issubdtype(df["openTime"].dtype, np.datetime64):
        # Assuming ms timestamp if not datetime
        df["datetime"] = pd.to_datetime(df["openTime"], unit="ms")
    else:
        df["datetime"] = df["openTime"]

    # Normalize Time to -1 to 1
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7)

    df.drop(columns=["datetime"], inplace=True)
    return df


def calculate_statistical_features(df):
    """Higher order moments"""
    for window in [36]:
        df[f"return_skew_{window}"] = df["log_return"].rolling(window=window).skew()
        # Kurtosis often has huge outliers, consider clipping or log
        df[f"return_kurt_{window}"] = df["log_return"].rolling(window=window).kurt()

    return df


# --- HELPER: CONTEXT LOADING ---
def load_btc_context() -> Optional[pd.DataFrame]:
    """Loads BTC returns to use as a global market factor."""
    try:
        btc_df = process_ticker("BTCUSDT", verbose=False)
        if btc_df is not None and not btc_df.empty:
            btc_df = ensure_numeric_data(btc_df)
            # Only need log return for correlation context
            btc_df["btc_log_return"] = np.log(
                btc_df["close"] / btc_df["close"].shift(1)
            )

            return btc_df[["openTime", "btc_log_return"]]
    except Exception as e:
        print(f"Could not load BTC context: {e}")
    return None


# --- MAIN GENERATOR ---
def generate_features(
    ticker: str,
    selected_features: Optional[List[FeatureType]] = None,
    btc_context: Optional[pd.DataFrame] = None,
):
    if selected_features is None:
        selected_features = list(FeatureType)

    # 1. Load Data
    df = process_ticker(ticker, verbose=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # 2. Basic Preprocessing
    df = ensure_numeric_data(df)
    # Global Log Return (Essential for stationarity)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 3. Feature Engineering
    if FeatureType.PRICE in selected_features:
        df = calculate_price_features(df)
    if FeatureType.VOLUME in selected_features:
        df = calculate_volume_features(df)
    if FeatureType.ORDER_FLOW in selected_features:
        df = calculate_order_flow_features(df)
    if FeatureType.TECHNICAL in selected_features:
        df = calculate_technical_indicators(df)
    if FeatureType.MOMENTUM in selected_features:
        df = calculate_momentum_features(df)
    if FeatureType.VOLUME_PRICE in selected_features:
        df = calculate_volume_price_features(df)
    if FeatureType.MICROSTRUCTURE in selected_features:
        df = calculate_microstructure_features(df)
    if FeatureType.TIME in selected_features:
        df = calculate_time_features(df)
    if FeatureType.STATISTICAL in selected_features:
        df = calculate_statistical_features(df)

    # 4. Merge BTC Context
    if btc_context is not None:
        if df["openTime"].dtype != btc_context["openTime"].dtype:
            df["openTime"] = df["openTime"].astype(btc_context["openTime"].dtype)
        df = df.merge(btc_context, on="openTime", how="left")
        df["btc_log_return"] = df["btc_log_return"].fillna(0)

    # 5. Create Targets
    # Regression Target
    df["TARGET_next_return"] = df["log_return"].shift(-PREDICT_STEPS)
    # Classification Target (1 if up, 0 if down) - Often easier for LSTM
    df["TARGET_class"] = (df["TARGET_next_return"] > 0).astype(int)

    # 6. Cleaning & Filtering
    df = clean_infinite_values(df)
    df = df.dropna()

    # 7. DROP RAW COLUMNS (Critical for LSTM Stationarity)
    # We remove the non-stationary raw price/volume columns
    raw_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quoteAssetVolume",
        "numOfTrades",
        "takerBuyBaseAssetVolume",
        "openTime",
        "closeTime",
        "ignore",
    ]
    # Keep only columns that are NOT in raw_cols
    final_cols = [c for c in df.columns if c not in raw_cols]

    # Ensure targets are at the end for easier splitting
    final_cols = [c for c in final_cols if "TARGET" not in c] + [
        c for c in df.columns if "TARGET" in c
    ]

    return df[final_cols]


if __name__ == "__main__":
    print("Loading BTC Context...")
    btc_ctx = load_btc_context()

    ticker = "BTCUSDT"
    print(f"Processing {ticker}...")

    # Using ALL features for demonstration
    result_df = generate_features(ticker, btc_context=btc_ctx)

    print(f"\nProcessing Complete. Final Shape: {result_df.shape}")
    if not result_df.empty:
        print("\nFirst 5 rows (Stationary Features Only):")
        print(result_df.head())
        print("\nColumns retained:")
        print(result_df.columns.tolist())

        # Validation Check
        if "close" in result_df.columns:
            print("❌ Error: Raw 'close' price still in dataset.")
        else:
            print("✅ Success: Raw prices removed.")

        result_df.to_csv("feature_output_test.csv", index=False)
