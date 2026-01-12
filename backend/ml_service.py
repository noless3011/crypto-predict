import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model, load_model

# Define custom objects for loading models (especially custom layers/metrics if any,
# though basic Keras layers usually load fine without this if strictly standard)
# However, if we defined custom models via function in notebook, we might need to recreate architecture
# if we were building from weights. But assuming we saved the full model (.keras).


class MLService:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}

    def get_available_models(self):
        try:
            files = os.listdir(self.models_dir)
            model_files = [f for f in files if f.endswith(".keras")]
            return sorted(model_files)
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def load_model_and_scaler(self, model_filename):
        if model_filename in self.models:
            return self.models[model_filename], self.scalers.get(model_filename)

        model_path = os.path.join(self.models_dir, model_filename)
        scaler_filename = model_filename.replace(".keras", "_scaler.pkl")
        scaler_path = os.path.join(self.models_dir, scaler_filename)

        print(f"Loading model from {model_path}...")
        try:
            model = load_model(model_path)
            self.models[model_filename] = model

            scaler = None
            if os.path.exists(scaler_path):
                print(f"Loading scaler from {scaler_path}...")
                scaler = joblib.load(scaler_path)
                self.scalers[model_filename] = scaler
            else:
                print(f"Warning: No scaler found at {scaler_path}")

            return model, scaler
            return model, scaler
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            raise e

    def enable_dummy_mode(self):
        """
        Registers custom objects required for dummy models.
        """
        tf.keras.utils.get_custom_objects()["price_with_noise"] = price_with_noise
        print("Dummy mode enabled: 'price_with_noise' registered.")

    # ==========================================
    # Data Processing Functions (Replicated)
    # ==========================================
    def resample_to_hourly(self, df):
        """
        Resample 5m data to Hourly data matching train_notebook.py
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            else:
                raise ValueError("DataFrame must have Date column or DatetimeIndex")

        # Resample logic from train_notebook.py
        # df_hourly = df.resample('h').agg({...})
        # Note: 'h' is deprecated in newer pandas, 'H' or 'h' works but 'h' is preferred in recent versions?
        # Actually 'h' is usually fine.

        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }

        # Only aggregate columns that exist
        final_agg = {k: v for k, v in agg_dict.items() if k in df.columns}

        if not final_agg:
            return df  # Nothing to aggregate

        df_hourly = df.resample("h").agg(final_agg).dropna()
        return df_hourly

    def add_technical_indicators(self, df):
        """
        Add technical indicators to dataframe
        Requires: Open, High, Low, Close columns
        """
        # Ensure we work on a copy to avoid SettingWithCopy warnings on the original df
        df = df.copy()

        # 1. RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # 2. MACD
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # 3. Moving Averages
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()

        # 4. Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

        # 5. ATR (Average True Range)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()

        # 6. Price Rate of Change
        df["ROC"] = (
            (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
        ) * 100

        # Drop NaN rows created by indicators
        df = df.dropna()

        return df

    def convert_to_log_returns(self, data):
        """
        Convert price data to log returns
        data: numpy array where column 0 is Close prices
        Returns: log returns (loses first row)
        """
        # Ensure data is float64 to avoid Decimal type issues
        data = np.array(data, dtype=np.float64)
        log_data = data.copy()
        # Convert Close price (column 0) to log returns
        # data is numpy array here
        # Log return = ln(Pt / Pt-1)
        # Note: We lose one data point at the beginning
        log_data[1:, 0] = np.log(data[1:, 0] / data[:-1, 0])
        return log_data[1:]

    def data_transform(self, data, anti=False, scaler=None):
        """
        Normalization
        """
        if not anti:
            # When implementing for prediction, we usually expect the scaler to be provided
            # (loaded from training). If not, we can't really transform consistently
            # with training data.
            pass  # Not implemented for 'fit' here, assuming inference only with loaded scaler
        else:
            # Inverse transform
            # scaler is a dict of {col_index: scaler_object}
            if data.ndim == 3:
                data = data.squeeze(axis=2)
            restored = np.zeros_like(data)

            # The scaler dict keys are integers (column indices)
            for i in range(data.shape[1]):
                if i in scaler:
                    column_scaler = scaler[i]
                    col = data[:, i].reshape(-1, 1)
                    restored[:, i] = column_scaler.inverse_transform(col).ravel()
                else:
                    # If for some reason scaler is missing for a column, keep as is
                    restored[:, i] = data[:, i]
            return restored

    def prepare_inference_data(self, df, n_steps_in=36):
        """
        Prepare data for inference.
        1. Resample to Hourly
        2. Add Indicators
        3. Select Features
        4. Log Transform
        5. Normalize (using loaded scaler)
        6. Create Sequence
        """
        # 1. Resample/Clean (Assuming df is already clean from CSV, but let's ensure hourly)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        # Resample to hourly to match training data
        df = self.resample_to_hourly(df)

        # 2. Add Indicators
        df = self.add_technical_indicators(df)

        # 3. Feature Selection (Multi-variate)
        # Columns must match training: ['Close'] + feature_cols
        # Exclude: 'Date', 'Date_Str', 'Date_Only', 'Close', 'Volume' from features list
        exclude_cols = ["Date", "Date_Str", "Date_Only", "Close", "Volume"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        final_cols = ["Close"] + feature_cols

        data = df[final_cols].values.astype(np.float64)
        dates = df.index.values  # keep dates for reference

        # 4. Log Transform
        # Need original close for inverse transform later
        original_close = data[:, 0].copy()  # Close prices
        log_data = self.convert_to_log_returns(data)

        # Adjust dates and original_close to match log_data length (removed 1st row)
        # log_data starts from index 1 of original data
        dates = dates[1:]
        original_close = original_close[1:]

        return log_data, dates, original_close, final_cols

    def normalize_for_inference(self, data, scaler_dict):
        """
        Normalize input data using the saved scaler dictionary
        """
        if scaler_dict is None:
            print("WARNING: scaler_dict is None, returning unnormalized data")
            return data

        # Ensure data is float64 to avoid Decimal type issues
        data = np.array(data, dtype=np.float64)

        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            if i in scaler_dict:
                col = data[:, i].reshape(-1, 1)
                normalized_data[:, i] = scaler_dict[i].transform(col).ravel()
            else:
                normalized_data[:, i] = data[:, i]
        return normalized_data

    def make_prediction(
        self,
        model_filename,
        df_recent,
        n_steps_in=36,
        n_steps_out=5,
        prediction_hours=5,
    ):
        """
        Main prediction pipeline with iterative prediction support.
        df_recent: DataFrame containing enough recent history (at least n_steps_in + indicator warmup)
        prediction_hours: Total hours to predict (must be divisible by n_steps_out, default 5)
        """
        model, scaler_dict = self.load_model_and_scaler(model_filename)
        if model is None:
            raise Exception("Model not found")

        print(f"DEBUG: Scaler dict loaded: {scaler_dict is not None}")
        if scaler_dict:
            print(f"DEBUG: Scaler dict keys: {scaler_dict.keys()}")
            print(f"DEBUG: Scaler[0] type: {type(scaler_dict.get(0))}")
            s0 = scaler_dict.get(0)
            if hasattr(s0, "center_") and hasattr(s0, "scale_"):
                print(f"DEBUG: Scaler[0] center={s0.center_}, scale={s0.scale_}")

        # Validate prediction_hours
        if prediction_hours % n_steps_out != 0:
            raise Exception(f"prediction_hours must be divisible by {n_steps_out}")

        # Prepare Data
        print(f"DEBUG: Input df shape: {df_recent.shape}")
        print(f"DEBUG: Input df columns: {df_recent.columns.tolist()}")
        print(
            f"DEBUG: Input df Close stats: min={df_recent['Close'].min()}, max={df_recent['Close'].max()}, mean={df_recent['Close'].mean()}"
        )

        log_data, dates, original_close, feature_names = self.prepare_inference_data(
            df_recent
        )

        print(f"DEBUG: log_data shape: {log_data.shape}")
        print(
            f"DEBUG: log_data[:,0] (Close log returns) stats: min={log_data[:, 0].min()}, max={log_data[:, 0].max()}, mean={log_data[:, 0].mean()}, std={log_data[:, 0].std()}"
        )
        print(f"DEBUG: original_close[-10:]: {original_close[-10:]}")
        print(f"DEBUG: last_close_price: {original_close[-1]}")

        # We need the LAST n_steps_in data points to predict the NEXT steps
        if len(log_data) < n_steps_in:
            raise Exception(
                f"Not enough data points after processing. Need {n_steps_in}, got {len(log_data)}"
            )

        # Calculate how many prediction iterations we need
        num_iterations = prediction_hours // n_steps_out

        # Keep track of all predictions
        all_pred_prices = []

        # Start with the last n_steps_in data points
        current_log_data = log_data.copy()
        last_close_price = original_close[-1]
        curr_price = last_close_price

        for iteration in range(num_iterations):
            # Get the last n_steps_in points for this iteration
            input_seq = current_log_data[-n_steps_in:]  # (n_steps_in, n_features)

            if iteration == 0:
                print(f"DEBUG iter {iteration}: input_seq shape: {input_seq.shape}")
                print(
                    f"DEBUG iter {iteration}: input_seq[:,0] stats: min={input_seq[:, 0].min()}, max={input_seq[:, 0].max()}, mean={input_seq[:, 0].mean()}"
                )

            # Normalize
            norm_seq = self.normalize_for_inference(input_seq, scaler_dict)

            if iteration == 0:
                print(
                    f"DEBUG iter {iteration}: norm_seq[:,0] stats: min={norm_seq[:, 0].min()}, max={norm_seq[:, 0].max()}, mean={norm_seq[:, 0].mean()}"
                )

            # Reshape for model (1, n_steps_in, n_features) - excluding Close from Input if Multi
            # In train_notebook.py:
            # if dim_type == 'Multi':
            #    seq_x = sequence[i:end_ix, 1:]  # Exclude Close (col 0)
            #    seq_y = sequence[end_ix:out_end_ix, 0]

            # So we must drop column 0 for X
            X_input = norm_seq[:, 1:]  # Drop Close column
            X_input = X_input.reshape((1, n_steps_in, X_input.shape[1]))

            # Predict
            # model output shape: (1, n_steps_out) -> Log returns of Close
            pred_log_returns = model.predict(X_input, verbose=0)

            # Inverse Transform
            # The model predicts Scaled Log Returns
            # y corresponds to column 0 (Close) of normalized data.
            # So we need to inverse transform the PREDICTION using scaler for column 0.

            pred_scaled = pred_log_returns.flatten()  # (n_steps_out, )

            print(f"DEBUG iter {iteration}: pred_scaled = {pred_scaled}")

            # Manually inverse transform using scaler[0]
            if scaler_dict and 0 in scaler_dict:
                pred_inv_log = (
                    scaler_dict[0]
                    .inverse_transform(pred_scaled.reshape(-1, 1))
                    .flatten()
                )
                print(
                    f"DEBUG iter {iteration}: After inverse transform = {pred_inv_log}"
                )
            else:
                pred_inv_log = pred_scaled
                print(f"DEBUG iter {iteration}: No scaler, using raw = {pred_inv_log}")

            # Convert Log Returns back to Price for this iteration
            iteration_prices = []
            iteration_log_returns = []

            for idx, log_ret in enumerate(pred_inv_log):
                # CLAMPING to prevent overflow (e.g., max 50% move in an hour)
                # np.exp(700) overflows. np.exp(0.5) is ~1.65.
                log_ret_orig = log_ret
                log_ret = np.clip(log_ret, -1.0, 1.0)

                if idx == 0:
                    print(
                        f"DEBUG: log_ret before clip={log_ret_orig}, after clip={log_ret}, curr_price={curr_price}"
                    )

                next_price = curr_price * np.exp(log_ret)
                iteration_prices.append(next_price)
                iteration_log_returns.append(log_ret)
                curr_price = next_price

                if idx == 0:
                    print(f"DEBUG: next_price={next_price}")

            # Add this iteration's prices to overall predictions
            all_pred_prices.extend(iteration_prices)

            # If we need more iterations, append predicted data to current_log_data
            if iteration < num_iterations - 1:
                # Create new log data rows for the predicted steps
                # We need to simulate the feature values for future steps
                # For simplicity, we'll use the last known feature values (excluding Close)
                # and append the predicted log returns for Close

                last_features = current_log_data[
                    -1, 1:
                ]  # Get last row's features (excluding Close)

                for pred_log_ret in iteration_log_returns:
                    # Create new row: [pred_log_ret, last_features...]
                    new_row = np.concatenate([[pred_log_ret], last_features])
                    current_log_data = np.vstack([current_log_data, new_row])

        return all_pred_prices


def price_with_noise(x):
    import tensorflow as tf
    # Output simulated Log Returns (not Price).
    # Small random values, expecting scaler[0] to be IdentityScaler for dummy.
    # Shape should match expected output (batch, 5)

    batch_size = tf.shape(x)[0]

    # Random log returns in range [-0.5%, +0.5%]
    dummy_log_returns = tf.random.uniform(
        shape=(batch_size, 5), minval=-0.005, maxval=0.005
    )

    return dummy_log_returns


ml_service = MLService()
