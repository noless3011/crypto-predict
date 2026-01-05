import os

import pandas as pd

from data import ClickhouseHelper, Interval

# Set pandas option to avoid downcasting warnings
pd.set_option("future.no_silent_downcasting", True)


# Process each ticker
def process_ticker(ticker: str, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing ticker {ticker}")
        print(f"{'=' * 60}")

    try:
        # Columns: ticker, openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume
        df = ClickhouseHelper.get_data_between(
            ticker=ticker,
            time_start=None,
            time_end=None,
            interval=Interval.FIVE_MINUTES,
            verbose=verbose,
            chunk_size=100000,
        )
        if verbose:
            print("\nDataFrame columns and types:")
            print(df.dtypes)
        if df.empty:
            if verbose:
                print(f"No data available for {ticker}, skipping...")

        # Check if closeTime - openTime equals 5 minutes in milliseconds
        expected_duration_ms = 300000
        expected_duration_inclusive_ms = 299999
        differences = df["closeTime"] - df["openTime"]
        unique_diffs = differences.unique()
        if verbose:
            print("Unique time differences (ms):", unique_diffs)
        if (differences == expected_duration_inclusive_ms).all():  # type: ignore
            if verbose:
                print("All rows are exactly 5 minutes (inclusive close: 299,999 ms).")
        elif (differences == expected_duration_ms).all():  # type: ignore
            if verbose:
                print("All rows are exactly 5 minutes (exclusive close: 300,000 ms).")
        else:
            if verbose:
                print("Warning: Inconsistent time intervals found.")

        # Sort by openTime and check for duplicates
        df = df.sort_values("openTime")
        duplicates = df[df.duplicated(subset=["openTime"], keep=False)]
        if not duplicates.empty:
            if verbose:
                print("Duplicate openTime entries found:")
                print(duplicates)
        else:
            if verbose:
                print("No duplicate openTime entries found.")

        # Check for missing intervals
        df = df.reset_index(drop=True)
        df["expected_openTime"] = df["openTime"].shift(1) + expected_duration_ms
        missing_intervals = df[df["openTime"] != df["expected_openTime"]]
        if not missing_intervals.empty:
            if verbose:
                print("Missing intervals detected:")
                print(missing_intervals[["openTime", "expected_openTime"]])
        else:
            if verbose:
                print("No missing intervals detected.")

        # Forward-fill price and volume (replace zeros with NaN first, then forward-fill)
        df["open"] = df["open"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        df["high"] = df["high"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        df["low"] = df["low"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        df["close"] = df["close"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        df["volume"] = df["volume"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        df["numOfTrades"] = (
            df["numOfTrades"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        )
        df["quoteAssetVolume"] = (
            df["quoteAssetVolume"].replace(0, pd.NA).ffill().infer_objects(copy=False)
        )
        df["takerBuyBaseAssetVolume"] = (
            df["takerBuyBaseAssetVolume"]
            .replace(0, pd.NA)
            .ffill()
            .infer_objects(copy=False)
        )

        # Drop temporary and unnecessary columns (keep openTime for feature engineering)
        df = df.drop(
            columns=["expected_openTime", "ticker", "closeTime"],
            errors="ignore",
        )
        if verbose:
            print("Data cleaning completed.")
        return df

    except Exception as e:
        e = e.with_traceback(None)
        print(f"Error processing {ticker}: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    process_ticker("btcusdt", verbose=False)
