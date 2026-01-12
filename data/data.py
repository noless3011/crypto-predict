from datetime import datetime
from enum import Enum
from re import I

import clickhouse_connect
import pandas as pd
from iterfzf import iterfzf
from tqdm import tqdm

client = clickhouse_connect.get_client(
    host="128.199.197.160",
    port=32014,
    username="readonly_user",
    password="grA5SfKxOQ",
    database="default",
)


class Interval(Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"


class ClickhouseHelper:
    @staticmethod
    def run_to_df(query: str, verbose: bool = False):
        if verbose:
            print(query)
        result = client.query(query)
        columns = result.column_names
        rows = result.result_rows
        return pd.DataFrame(rows, columns=columns)  # type: ignore

    @staticmethod
    def find_ticker(
        ticker: str, interval: Interval = Interval.FIVE_MINUTES, verbose: bool = False
    ):
        try:
            # Query database for list of tickers
            df = ClickhouseHelper.list_ticker(interval=interval, verbose=False)
            if df.empty:
                print("No tickers found in database.")
                return None

            tickers = df["ticker"].astype(str).tolist()

            # exact match
            exact_matches = [t for t in tickers if t == ticker]
            if len(exact_matches) == 1:
                if verbose:
                    print(f"Found exact match: {exact_matches[0]}")
                return exact_matches[0]

            # let fzf rank, limit to top 5 results
            selected = iterfzf(
                tickers,
                query=ticker,
                sort=True,
                exact=False,
                __extra__=[
                    "--height=7",
                    "--tiebreak=length,index",
                    "--scheme=default",
                ],
            )

            if verbose and selected:
                print(f"Selected ticker: {selected}")

            return selected or None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def list_ticker(interval: Interval = Interval.FIVE_MINUTES, verbose: bool = False):
        query = f"SELECT DISTINCT ticker FROM future_kline_{interval.value}"
        return ClickhouseHelper.run_to_df(query, verbose=verbose)

    @staticmethod
    def get_latest_data(
        ticker: str,
        limit: int = 5000,
        time_end: datetime | None = None,
        interval: Interval = Interval.FIVE_MINUTES,
        verbose: bool = False,
    ):
        """
        Get the latest N rows of OHLCV data for a ticker, optionally up to a specific end time
        This is more efficient than get_data_between when you just need recent data
        """
        conditions = [
            f"ticker = '{ClickhouseHelper.find_ticker(ticker, interval=interval, verbose=verbose)}'"
        ]

        if time_end:
            conditions.append(f"openTime <= {int(time_end.timestamp() * 1000)}")

        query = f"""
        SELECT openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numOfTrades, takerBuyBaseAssetVolume
        FROM future_kline_{interval.value}
        WHERE {" AND ".join(conditions)}
        ORDER BY openTime DESC
        LIMIT {limit}
        """

        if verbose:
            print(f"Fetching latest {limit} rows")
            print(query)

        df = ClickhouseHelper.run_to_df(query, verbose=verbose)

        # Reverse to get chronological order (oldest to newest)
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)

            # Convert Decimal types to float for numpy compatibility
            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quoteAssetVolume",
                "takerBuyBaseAssetVolume",
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def get_data_between(
        ticker: str,
        time_start: datetime | None,
        time_end: datetime | None,
        interval: Interval = Interval.FIVE_MINUTES,
        verbose: bool = False,
        chunk_size: int = 100000,  # Number of rows per chunk
    ):
        """
        Get OHLCV data between a time range using chunked queries
        """
        conditions = [
            f"ticker = '{ClickhouseHelper.find_ticker(ticker, interval=interval, verbose=verbose)}'"
        ]
        if time_start:
            conditions.append(f"openTime >= {int(time_start.timestamp() * 1000)}")

        if time_end:
            conditions.append(f"openTime <= {int(time_end.timestamp() * 1000)}")

        # Fetch data in chunks
        all_data = []
        offset = 0

        with tqdm(desc="Fetching data", unit=" rows") as pbar:
            while True:
                query = f"""
                SELECT openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numOfTrades, takerBuyBaseAssetVolume
                FROM future_kline_{interval.value}
                WHERE {" AND ".join(conditions)}
                ORDER BY openTime
                LIMIT {chunk_size} OFFSET {offset}
                """
                if verbose:
                    print(f"Fetching chunk at offset {offset}")

                chunk_df = ClickhouseHelper.run_to_df(query, verbose=verbose)

                if chunk_df.empty:
                    break

                all_data.append(chunk_df)
                offset += chunk_size
                pbar.update(len(chunk_df))

                if len(chunk_df) < chunk_size:
                    break

        # Combine all chunks
        if all_data:
            df = pd.concat(all_data, ignore_index=True)

            # Convert Decimal types to float for numpy compatibility
            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quoteAssetVolume",
                "takerBuyBaseAssetVolume",
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df
        else:
            return pd.DataFrame()

    @staticmethod
    def get_news_between(
        time_start: datetime | None,
        time_end: datetime | None,
        limit: int | None = None,
        offset: int | None = None,
        verbose: bool = False,
        include_body: bool = False,  # Optional flag to include full article text
    ):
        """
        Get news data between a time range with optional pagination
        """
        # Select only necessary columns to reduce data transfer
        columns = [
            "id",
            "publishedOn",
            "title",
            "subtitle",
            "sentiment",
            "sourceName",
            "categories",
            "url",
            "imageUrl",
            "authors",
            "keywords",
            "guid",
        ]

        if include_body:
            columns.append("rawBody")

        conditions = []
        if time_start:
            conditions.append(f"publishedOn >= {int(time_start.timestamp())}")

        if time_end:
            conditions.append(f"publishedOn <= {int(time_end.timestamp())}")

        query = f"""
        SELECT {", ".join(columns)}
        FROM news
        {"WHERE " + " AND ".join(conditions) if conditions else ""}
        ORDER BY publishedOn DESC
        """

        if limit is not None:
            query += f" LIMIT {limit}"

        if offset is not None:
            query += f" OFFSET {offset}"

        return ClickhouseHelper.run_to_df(query, verbose=verbose)

    @staticmethod
    def get_news_count(
        time_start: datetime | None,
        time_end: datetime | None,
        verbose: bool = False,
    ):
        """
        Get count of news items between a time range
        """
        conditions = []
        if time_start:
            conditions.append(f"publishedOn >= {int(time_start.timestamp())}")

        if time_end:
            conditions.append(f"publishedOn <= {int(time_end.timestamp())}")

        query = f"""
        SELECT COUNT(*) as count
        FROM news
        {"WHERE " + " AND ".join(conditions) if conditions else ""}
        """

        df = ClickhouseHelper.run_to_df(query, verbose=verbose)
        if not df.empty:
            count = df.iloc[0]["count"]
            # Convert numpy type to native Python int for JSON serialization
            return int(count) if count is not None else 0
        return 0


if __name__ == "__main__":
    # Example usage
    # df = ClickhouseHelper.get_data_between(
    #     ticker="ETCUSDT",
    #     time_start=None,
    #     time_end=None,
    #     interval=Interval.FIVE_MINUTES,
    #     verbose=True,
    # )
    # df.to_csv("etc_usdt_5m.csv", index=False)

    news_df = ClickhouseHelper.get_news_between(
        time_start=datetime(2024, 1, 1),
        time_end=datetime(2024, 1, 2),
        verbose=True,
    )
    news_df.to_csv("news.csv", index=False)
