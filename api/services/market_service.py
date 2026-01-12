from datetime import datetime
import pandas as pd
import clickhouse_connect
from api.config import settings
from typing import List, Optional
from api.models.schemas import TickerSummary, OHLCVPoint, Tickermetadata, TickerType

class MarketService:
    def __init__(self):
        pass # Client is created per request
        
    def _get_client(self):
        return clickhouse_connect.get_client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            username=settings.CLICKHOUSE_USER,
            password=settings.CLICKHOUSE_PASSWORD,
            database=settings.CLICKHOUSE_DB,
        )

    def get_all_tickers(self) -> List[str]:
        with self._get_client() as client:
            query = "SELECT DISTINCT ticker FROM future_kline_5m"
            result = client.query(query)
            return [row[0] for row in result.result_rows]

    def _resample_dataframe(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample 5m data to target interval"""
        # Map frontend interval strings to pandas offset aliases
        # 15m -> 15min, 1h -> 1h, 4h -> 4h, 1d -> 1D
        interval_map = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W"
        }
        
        # If interval is not in map or is 5m, return original
        if interval.lower() not in interval_map or interval.lower() == "5m":
            return df
            
        pandas_interval = interval_map[interval.lower()]
            
        if df.empty:
            return df

        # Convert openTime (ms) to datetime
        df["datetime"] = pd.to_datetime(df["openTime"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        # Resample logic
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        
        # Resample
        df_res = df.resample(pandas_interval).agg(agg_dict)
        
        # Drop NaN rows (e.g. gaps)
        df_res = df_res.dropna()
        
        # Reset index and convert back to openTime (ms)
        df_res.reset_index(inplace=True)
        df_res["openTime"] = df_res["datetime"].astype(int) // 10**6 # Convert ns to ms
        
        # Return sorted by openTime with original columns
        df_res = df_res.sort_values("openTime")[["openTime", "open", "high", "low", "close", "volume"]]
        return df_res

    def search_tickers(self, keyword: Optional[str] = None) -> List[Tickermetadata]:
        # Ideally, we would have a table with full ticker names.
        # For now, we fetch all unique tickers and filter in python.
        tickers = self.get_all_tickers()
        
        filtered = tickers
        if keyword:
            keyword_lower = keyword.lower()
            filtered = [t for t in tickers if keyword_lower in t.lower()]
        
        # Mapping to metadata model (Mocking name/exchange as we only have symbol)
        return [
            Tickermetadata(
                symbol=t,
                name=t, # Placeholder as we don't have mapping
                type=TickerType.CRYPTO,
                exchange="BINANCE"
            )
            for t in filtered[:50] # Limit to 50
        ]

    def get_ticker_summary(self, symbol: str) -> Optional[TickerSummary]:
        with self._get_client() as client:
            # Get latest data point
            query = f"""
            SELECT close, volume, openTime 
            FROM future_kline_5m 
            WHERE ticker = '{symbol}' 
            ORDER BY openTime DESC 
            LIMIT 1
            """
            result = client.query(query)
            if not result.result_rows:
                return None
            
            latest_price, latest_volume, latest_time = result.result_rows[0]
            
            # Calculate 24h change
            # 24h ago = latest_time - 24*60*60*1000
            time_24h_ago = latest_time - 24 * 60 * 60 * 1000
            
            query_24h = f"""
            SELECT close
            FROM future_kline_5m
            WHERE ticker = '{symbol}' AND openTime <= {time_24h_ago}
            ORDER BY openTime DESC
            LIMIT 1
            """
            result_24h = client.query(query_24h)
            
            change_24h = 0.0
            if result_24h.result_rows:
                price_24h_ago = result_24h.result_rows[0][0]
                if price_24h_ago != 0:
                    change_24h = ((latest_price - price_24h_ago) / price_24h_ago) * 100

            return TickerSummary(
                symbol=symbol,
                current_price=float(latest_price),
                change_24h=round(float(change_24h), 2),
                volume_24h=float(latest_volume), # This is 5m volume, ideally we sum up 24h volume
                updated_at=int(latest_time)
            )

    def get_historical_data(self, symbol: str, interval: str, start_time: Optional[int], end_time: Optional[int]) -> List[List[float]]:
        # Reuse get_historical_dataframe to ensure consistency and resampling
        df = self.get_historical_dataframe(symbol, interval, start_time, end_time)
        
        if df.empty:
            return []
            
        data = df.values.tolist()
        return data

    def get_raw_dataframe(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        with self._get_client() as client:
            query = f"""
            SELECT openTime, open, high, low, close, volume
            FROM future_kline_5m
            WHERE ticker = '{symbol}'
            ORDER BY openTime DESC
            LIMIT {limit}
            """
            result = client.query(query)
            if not result.result_rows:
                return pd.DataFrame()
                
            df = pd.DataFrame(result.result_rows, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert numeric columns to float to avoid Decimal type issues with numpy
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df = df.sort_values('openTime').reset_index(drop=True)
            
            # Remove potential duplicates
            df = df.drop_duplicates(subset=['openTime'], keep='last').reset_index(drop=True)
            
            return df

    def get_historical_dataframe(self, symbol: str, interval: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        with self._get_client() as client:
            table_name = "future_kline_5m"
            conditions = [f"ticker = '{symbol}'"]
            
            # Add buffer to start_time for indicator warmup if needed
            # For simplicity, we just use the provided start_time, 
            # but ideally we should fetch a bit earlier. 
            # The caller handles 'warmup' or we just accept the first few points are unstable.
            # Actually, to get accurate RSI at 'start_time', we need PREVIOUS data.
            # Let's subtract a buffer (e.g. 100 * 5 mins) if start_time is present.
            query_start_time = start_time
            if start_time:
                # Buffer: 200 candles * 5 minutes * 60 seconds
                buffer_sec = 200 * 5 * 60
                query_start_time = start_time - buffer_sec
                conditions.append(f"openTime >= {query_start_time}")
            
            if end_time:
                conditions.append(f"openTime <= {end_time}")
            
            query = f"""
            SELECT openTime, open, high, low, close, volume
            FROM {table_name}
            WHERE {" AND ".join(conditions)}
            ORDER BY openTime ASC
            LIMIT 50000
            """
            
            result = client.query(query)
            if not result.result_rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(result.result_rows, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
             # Convert numeric columns to float
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Dedup just in case
            df = df.drop_duplicates(subset=['openTime'], keep='last').reset_index(drop=True)
            
            # Resample if needed
            if interval != "5m":
                df = self._resample_dataframe(df, interval)
            
            return df
