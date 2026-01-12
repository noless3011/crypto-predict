import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from api.models.lstm import CryptoLSTM
from api.models.transformer import CryptoTransformer
from api.services.market_service import MarketService
from api.services.indicator_service import IndicatorService
from api.config import settings
from api.models.schemas import ForecastPoint, ForecastResponse, MarketDirection
from sklearn.preprocessing import StandardScaler
import os
from typing import List

class PredictionService:
    def __init__(self):
        self.market_service = MarketService()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Architecture Configs ---
        # LSTM
        self.lstm_config = {
            "input_dim": 8,
            "hidden_dim": 128,
            "num_layers": 3,
            "output_dim": 3,
            "dropout": 0.3,
            "seq_len": 48
        }
        
        # Transformer
        self.transformer_config = {
            "input_dim": 8,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 3,
            "output_dim": 3,
            "dropout": 0.3,
            "seq_len": 72
        }

        # --- Load Models ---
        self.models = {}
        
        # 1. LSTM
        lstm_path = "best_crypto_lstm.pth"
        if os.path.exists(lstm_path):
            try:
                lstm = CryptoLSTM(
                    input_dim=self.lstm_config["input_dim"],
                    hidden_dim=self.lstm_config["hidden_dim"],
                    num_layers=self.lstm_config["num_layers"],
                    output_dim=self.lstm_config["output_dim"],
                    dropout=self.lstm_config["dropout"]
                ).to(self.device)
                lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
                lstm.eval()
                self.models["LSTM"] = lstm
            except Exception as e:
                print(f"Error loading LSTM: {e}")

        # 2. Transformer
        trans_path = "best_crypto_transformer.pth"
        if os.path.exists(trans_path):
            try:
                transformer = CryptoTransformer(
                    input_dim=self.transformer_config["input_dim"],
                    d_model=self.transformer_config["d_model"],
                    nhead=self.transformer_config["nhead"],
                    num_layers=self.transformer_config["num_layers"],
                    output_dim=self.transformer_config["output_dim"],
                    dropout=self.transformer_config["dropout"]
                ).to(self.device)
                transformer.load_state_dict(torch.load(trans_path, map_location=self.device))
                transformer.eval()
                self.models["Transformer"] = transformer
            except Exception as e:
                print(f"Error loading Transformer: {e}")
        
        # 3. LightGBM
        lgb_path = "best_crypto_model_gpu.txt"
        if os.path.exists(lgb_path):
            try:
                self.models["LightGBM"] = lgb.Booster(model_file=lgb_path)
            except Exception as e:
                print(f"Error loading LightGBM: {e}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize features as per training logic
        df['datetime'] = pd.to_datetime(df['openTime'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # Resample to 1H provided sufficient data
        df_1h = df.resample("1h").agg(agg_dict).dropna()
        
        if df_1h.empty:
            return pd.DataFrame()

        # Indicators
        # RSI 14
        df_1h["rsi_14"] = IndicatorService.calculate_rsi(df_1h["close"], 14) / 100.0 
        
        # Parkinson Volatility
        high = df_1h["high"]
        low = df_1h["low"]
        const = 1.0 / (4.0 * np.log(2.0))
        log_hl = np.log(high / low) ** 2
        df_1h["vol_parkinson"] = np.sqrt(const * log_hl.rolling(window=20).mean())
        
        # Features
        df_1h["ret_1h"] = df_1h["close"].pct_change()
        df_1h["std_24h"] = df_1h["ret_1h"].rolling(24).std()
        ma_24 = df_1h["close"].rolling(24).mean()
        df_1h["dist_ma_24"] = (df_1h["close"] - ma_24) / ma_24
        
        df_1h["ret_lag_1"] = df_1h["ret_1h"].shift(1)
        df_1h["ret_lag_2"] = df_1h["ret_1h"].shift(2)
        df_1h["ret_lag_6"] = df_1h["ret_1h"].shift(6)
        
        feature_cols = [
            "ret_1h", "ret_lag_1", "ret_lag_2", "ret_lag_6",
            "std_24h", "dist_ma_24", "rsi_14", "vol_parkinson"
        ]
        
        return df_1h[feature_cols].dropna()

    def forecast(self, symbol: str, days: int = 7) -> List[ForecastResponse]:
        # 1. Get Data (Enough for features + sequence)
        # Fetching 2000 5m candles ~ 166 hours ~ 7 days of data
        # Need enough for 1H resampling + lags + sequence
        raw_df = self.market_service.get_raw_dataframe(symbol, limit=10000)
        
        responses = []
        if raw_df.empty:
            return responses
            
        # 2. Prepare Features
        features_df = self._prepare_features(raw_df)
        if features_df.empty:
            return responses

        # Determine next time
        latest_time = features_df.index[-1]
        next_time_ms = int(latest_time.timestamp() * 1000) + (24 * 60 * 60 * 1000)
        
        # Common Feature Scaling for DL models
        scaler = StandardScaler()
        # Note: Fitting on just the requested window might shift distribution if not careful, 
        # but typical for inference if no saved scaler.
        features_scaled = scaler.fit_transform(features_df.values)

        # Iterate through loaded models
        for name, model in self.models.items():
            direction = MarketDirection.FLAT
            confidence = 0.0
            
            try:
                if name == "LightGBM":
                    # LightGBM uses unscale features (or as trained), but 'evaluate.py' didn't scale for LGBM.
                    # It uses features_df values directly.
                    # Features: "ret_1h", ...
                    # Just take the last row?
                    # LightGBM predicts on single row input usually? 
                    # Yes, current state.
                    last_row = features_df.iloc[[-1]]
                    pred_prob = model.predict(last_row) # shape (1, 3)
                    
                    pred_idx = np.argmax(pred_prob, axis=1)[0]
                    confidence = pred_prob[0][pred_idx]
                    
                elif name == "LSTM":
                    seq_len = self.lstm_config["seq_len"]
                    if len(features_scaled) < seq_len:
                        continue
                    
                    last_sequence = features_scaled[-seq_len:]
                    input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][pred_idx].item()

                elif name == "Transformer":
                    seq_len = self.transformer_config["seq_len"]
                    if len(features_scaled) < seq_len:
                        continue
                        
                    last_sequence = features_scaled[-seq_len:]
                    input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][pred_idx].item()
                
                # Map prediction
                direction_map = {0: MarketDirection.FLAT, 1: MarketDirection.UP, 2: MarketDirection.DOWN}
                direction = direction_map.get(pred_idx, MarketDirection.FLAT)
                
                responses.append(ForecastResponse(
                    model_name=name,
                    forecast_data=[
                        ForecastPoint(
                            time=next_time_ms,
                            direction=direction,
                            confidence=round(float(confidence), 2)
                        )
                    ]
                ))

            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                
        return responses
