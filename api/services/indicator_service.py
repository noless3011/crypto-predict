import pandas as pd
import numpy as np
from typing import List, Dict, Any
from api.models.schemas import IndicatorDataPoint, IndicatorType

class IndicatorService:
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return pd.DataFrame({'macd': macd, 'signal': signal_line, 'histogram': histogram})

    @staticmethod
    def calculate_bollinger(series: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    def get_indicator(self, df: pd.DataFrame, type: IndicatorType, period: int = 14) -> List[IndicatorDataPoint]:
        if df.empty:
            return []
            
        close = df['close']
        timestamps = df['openTime']
        
        result = []
        
        if type == IndicatorType.RSI:
            rsi_values = self.calculate_rsi(close, period)
            for t, val in zip(timestamps, rsi_values):
                if pd.notna(val):
                    result.append(IndicatorDataPoint(time=t, value=val))
                    
        elif type == IndicatorType.MACD:
            # Standard MACD 12, 26, 9. Can parameterize later if needed.
            macd_df = self.calculate_macd(close)
            for t, row in macd_df.iterrows():
                # t is index if series, but here we iterate zip
                pass 
            
            # Correct iteration
            for i in range(len(timestamps)):
                t = timestamps[i]
                m = macd_df['macd'].iloc[i]
                s = macd_df['signal'].iloc[i]
                h = macd_df['histogram'].iloc[i]
                
                if pd.notna(m):
                    result.append(IndicatorDataPoint(time=t, macd=m, signal=s, histogram=h))

        elif type == IndicatorType.BOLLINGER:
            bb_df = self.calculate_bollinger(close, period=period)
            for i in range(len(timestamps)):
                t = timestamps[i]
                u = bb_df['upper'].iloc[i]
                m = bb_df['middle'].iloc[i]
                l = bb_df['lower'].iloc[i]
                
                if pd.notna(u):
                    result.append(IndicatorDataPoint(time=t, upper=u, middle=m, lower=l))
                    
        elif type == IndicatorType.EMA:
            ema = self.calculate_ema(close, period)
            for i in range(len(timestamps)):
                t = timestamps[i]
                val = ema.iloc[i]
                if pd.notna(val):
                    result.append(IndicatorDataPoint(time=t, value=val))
                    
        elif type == IndicatorType.SMA:
            sma = self.calculate_sma(close, period)
            for i in range(len(timestamps)):
                t = timestamps[i]
                val = sma.iloc[i]
                if pd.notna(val):
                    result.append(IndicatorDataPoint(time=t, value=val))
                    
        # Ensure result is sorted by time for frontend charts
        result.sort(key=lambda x: x.time)
        return result
