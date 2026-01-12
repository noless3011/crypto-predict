import math
import os
import warnings
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize

# Import from existing modules
from data.data import ClickhouseHelper, Interval
from data.feature import add_features_and_clean

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEBUG = True  # Enable debug output
TICKER = "BTCUSDT"  # Ticker to evaluate
TIME_START = datetime(2023, 1, 1)  # Evaluation period start
TIME_END = datetime(2023, 6, 30)  # Evaluation period end
INTERVAL = Interval.FIVE_MINUTES

# Model paths
LIGHTGBM_MODEL_PATH = "best_crypto_model_gpu.txt"
LSTM_MODEL_PATH = "best_crypto_lstm.pth"
TRANSFORMER_MODEL_PATH = "best_crypto_transformer.pth"

# Output directory for results
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device for PyTorch models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# LSTM MODEL ARCHITECTURE (Must match training)
# ==============================================================================
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # out shape: (batch_size, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])
        return out


# ==============================================================================
# TRANSFORMER MODEL ARCHITECTURE (Must match training)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe shape: [max_len, 1, d_model] -> batch_first=True in logic below
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, D_Model]
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CryptoTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        output_dim,
        dropout=0.1,
        max_len=500,
    ):
        super(CryptoTransformer, self).__init__()

        # 1. Input Projection: Map features to d_model size
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # 3. Transformer Encoder
        # batch_first=True means input is (Batch, Seq, Feature)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. Output Head
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src: (Batch, Seq_Len, Features)

        # Embed and add position info
        src = self.input_embedding(src)
        src = self.pos_encoder(src)

        # Pass through Transformer
        # output: (Batch, Seq_Len, d_model)
        output = self.transformer_encoder(src)

        # We take the LAST time step for prediction
        # output[:, -1, :] shape: (Batch, d_model)
        return self.decoder(output[:, -1, :])


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_evaluation_data(ticker, time_start, time_end, interval=Interval.FIVE_MINUTES):
    """Load data from database and apply feature engineering"""
    print(f"Loading data for {ticker} from {time_start} to {time_end}...")

    # Load OHLCV data
    df = ClickhouseHelper.get_data_between(
        ticker=ticker,
        time_start=time_start,
        time_end=time_end,
        interval=interval,
        verbose=True,
    )

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    if DEBUG:
        print(f"\nRaw data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst few rows:\n{df.head()}")

    # Convert all numeric columns to proper types
    print("Converting data types...")
    numeric_cols = ["open", "high", "low", "close", "volume", "openTime", "closeTime"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove any rows with NaN after conversion
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Replace zeros with forward fill
    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]]
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
    )

    print(f"After cleaning: {len(df)} rows")

    if DEBUG:
        print(f"\nCleaned data dtypes:\n{df.dtypes}")
        print(
            f"Sample values:\n{df[['open', 'high', 'low', 'close', 'volume']].head()}"
        )

    # Load news data
    news_df = ClickhouseHelper.get_news_between(
        time_start=time_start,
        time_end=time_end,
        verbose=True,
    )

    # Process news if available
    if not news_df.empty:
        news_df["datetime"] = pd.to_datetime(news_df["publishedOn"], unit="s")
        sentiment_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        news_df["val"] = news_df["sentiment"].map(sentiment_map).fillna(0)

        global_news = (
            news_df.set_index("datetime").resample("5min")["val"].mean().reset_index()
        )
        global_news["global_sentiment_ewma"] = global_news["val"].ewm(span=12).mean()
        global_news.rename(columns={"val": "global_sentiment_score"}, inplace=True)
    else:
        global_news = pd.DataFrame()

    # Apply feature engineering (import from data.feature)
    print("Applying feature engineering...")

    if DEBUG:
        print(f"\nBefore feature engineering:")
        print(f"  Shape: {df.shape}")
        print(
            f"  Key column types: {df[['open', 'high', 'low', 'close']].dtypes.to_dict()}"
        )

    try:
        df = add_features_and_clean(df, global_news)
        print(f"After feature engineering: {len(df)} rows, {len(df.columns)} columns")

        # Verify we have the expected columns
        if df.empty:
            raise ValueError("DataFrame is empty after feature engineering")

        # Check for any remaining non-numeric values
        numeric_check_cols = ["close", "open", "high", "low", "volume"]
        for col in numeric_check_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(f"Warning: Column {col} is not numeric type: {df[col].dtype}")
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        print(f"DataFrame info before error:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Data types:\n{df.dtypes}")
        raise


def prepare_lightgbm_features(df):
    """Prepare features for LightGBM model"""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Resample to 1h
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "vol_parkinson": "mean",
        "rsi_14": "last",
    }
    valid_aggs = {k: v for k, v in agg_dict.items() if k in df.columns}
    df_res = df.resample("1h").agg(valid_aggs)

    # Features
    df_res["ret_1h"] = df_res["close"].pct_change()
    df_res["ret_lag_1"] = df_res["ret_1h"].shift(1)
    df_res["ret_lag_2"] = df_res["ret_1h"].shift(2)
    df_res["ret_lag_6"] = df_res["ret_1h"].shift(6)
    df_res["std_24h"] = df_res["ret_1h"].rolling(24).std()
    ma_24 = df_res["close"].rolling(24).mean()
    df_res["dist_ma_24"] = (df_res["close"] - ma_24) / ma_24

    # Target
    future_ret = df_res["close"].pct_change(24).shift(-24)
    threshold = (df_res["std_24h"] * np.sqrt(24) * 0.4).fillna(0)
    conditions = [future_ret > threshold, future_ret < -threshold]
    choices = [1, 2]
    df_res["target"] = np.select(conditions, choices, default=0)

    df_res = df_res.dropna()

    feature_cols = [
        "ret_1h",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_6",
        "std_24h",
        "dist_ma_24",
        "rsi_14",
        "vol_parkinson",
    ]
    feature_cols = [c for c in feature_cols if c in df_res.columns]

    X = df_res[feature_cols].astype(np.float32)
    y = df_res["target"].astype(np.int32)

    return X, y


def create_sequences(features, targets, seq_length):
    """Create sequences for LSTM"""
    xs, ys = [], []
    if len(features) <= seq_length:
        return np.array(xs), np.array(ys)

    for i in range(len(features) - seq_length):
        x_seq = features[i : (i + seq_length)]
        y_target = targets[i + seq_length]
        xs.append(x_seq)
        ys.append(y_target)

    return np.array(xs), np.array(ys)


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
def calculate_metrics(y_true, y_pred, y_proba=None, model_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    results = {}

    # Basic accuracy
    results["accuracy"] = accuracy_score(y_true, y_pred)

    # For multiclass (3 classes: 0=Flat, 1=Up, 2=Down)
    if len(np.unique(y_true)) > 2:
        # Macro averages (treat all classes equally)
        results["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        results["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Weighted averages (account for class imbalance)
        results["precision_weighted"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        results["recall_weighted"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        results["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        results["precision_flat"] = precision_per_class[0]
        results["recall_flat"] = recall_per_class[0]
        results["f1_flat"] = f1_per_class[0]

        results["precision_up"] = precision_per_class[1]
        results["recall_up"] = recall_per_class[1]
        results["f1_up"] = f1_per_class[1]

        if len(precision_per_class) > 2:
            results["precision_down"] = precision_per_class[2]
            results["recall_down"] = recall_per_class[2]
            results["f1_down"] = f1_per_class[2]

        # AUC-ROC for multiclass (one-vs-rest)
        if y_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                if y_true_bin.shape[1] > 1:
                    results["auc_roc_macro"] = roc_auc_score(
                        y_true_bin, y_proba, average="macro", multi_class="ovr"
                    )
                    results["auc_roc_weighted"] = roc_auc_score(
                        y_true_bin, y_proba, average="weighted", multi_class="ovr"
                    )
            except Exception as e:
                print(f"Warning: Could not calculate AUC-ROC: {e}")
                results["auc_roc_macro"] = np.nan
                results["auc_roc_weighted"] = np.nan
    else:
        # Binary classification
        results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        results["recall"] = recall_score(y_true, y_pred, zero_division=0)
        results["f1"] = f1_score(y_true, y_pred, zero_division=0)

        if y_proba is not None:
            try:
                results["auc_roc"] = roc_auc_score(
                    y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                )
            except Exception as e:
                print(f"Warning: Could not calculate AUC-ROC: {e}")
                results["auc_roc"] = np.nan

    return results


def print_evaluation_report(results, model_name):
    """Print formatted evaluation report"""
    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print(f"{'=' * 60}")

    print(f"\nAccuracy: {results.get('accuracy', 0):.4f}")

    if "precision_macro" in results:
        print(f"\n--- Macro Averages (Equal Weight per Class) ---")
        print(f"Precision: {results['precision_macro']:.4f}")
        print(f"Recall:    {results['recall_macro']:.4f}")
        print(f"F1-Score:  {results['f1_macro']:.4f}")

        print(f"\n--- Weighted Averages (Account for Class Imbalance) ---")
        print(f"Precision: {results['precision_weighted']:.4f}")
        print(f"Recall:    {results['recall_weighted']:.4f}")
        print(f"F1-Score:  {results['f1_weighted']:.4f}")

        print(f"\n--- Per-Class Metrics ---")
        print(f"FLAT (Class 0):")
        print(f"  Precision: {results.get('precision_flat', 0):.4f}")
        print(f"  Recall:    {results.get('recall_flat', 0):.4f}")
        print(f"  F1-Score:  {results.get('f1_flat', 0):.4f}")

        print(f"\nUP (Class 1) - CRITICAL FOR PROFITABLE TRADING:")
        print(
            f"  Precision: {results.get('precision_up', 0):.4f} <- When model says 'Buy', % correct"
        )
        print(
            f"  Recall:    {results.get('recall_up', 0):.4f} <- % of profitable opportunities caught"
        )
        print(f"  F1-Score:  {results.get('f1_up', 0):.4f}")

        if "precision_down" in results:
            print(f"\nDOWN (Class 2):")
            print(f"  Precision: {results.get('precision_down', 0):.4f}")
            print(f"  Recall:    {results.get('recall_down', 0):.4f}")
            print(f"  F1-Score:  {results.get('f1_down', 0):.4f}")

        if not np.isnan(results.get("auc_roc_macro", np.nan)):
            print(f"\n--- AUC-ROC (Ability to Separate Classes) ---")
            print(f"Macro AUC-ROC:    {results.get('auc_roc_macro', 0):.4f}")
            print(f"Weighted AUC-ROC: {results.get('auc_roc_weighted', 0):.4f}")
    else:
        print(f"\nPrecision: {results.get('precision', 0):.4f}")
        print(f"Recall:    {results.get('recall', 0):.4f}")
        print(f"F1-Score:  {results.get('f1', 0):.4f}")
        if not np.isnan(results.get("auc_roc", np.nan)):
            print(f"AUC-ROC:   {results.get('auc_roc', 0):.4f}")

    print(f"{'=' * 60}\n")


def plot_evaluation_results(y_true, y_pred, y_proba, model_name):
    """Create comprehensive visualization of results"""
    fig = plt.figure(figsize=(20, 10))

    # Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=True)
    ax1.set_title(f"{model_name} - Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Class Distribution
    ax2 = plt.subplot(2, 3, 2)
    unique, counts = np.unique(y_true, return_counts=True)
    ax2.bar(unique, counts, alpha=0.7, label="Actual")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax2.bar(unique_pred, counts_pred, alpha=0.7, label="Predicted")
    ax2.set_title("Class Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")
    ax2.legend()

    # ROC Curves (One-vs-Rest for multiclass)
    ax3 = plt.subplot(2, 3, 3)
    if y_proba is not None and len(np.unique(y_true)) > 2:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]

        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_names = {0: "Flat", 1: "Up", 2: "Down"}
                ax3.plot(
                    fpr,
                    tpr,
                    lw=2,
                    label=f"{class_names.get(i, i)} (AUC = {roc_auc:.2f})",
                )
            except:
                pass

        ax3.plot([0, 1], [0, 1], "k--", lw=2)
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curves (One-vs-Rest)")
        ax3.legend(loc="lower right")
    elif y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
        ax3.plot([0, 1], [0, 1], "k--", lw=2)
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curve")
        ax3.legend(loc="lower right")

    # Precision-Recall per class
    ax4 = plt.subplot(2, 3, 4)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(precision_per_class))
    width = 0.25
    ax4.bar(x - width, precision_per_class, width, label="Precision", alpha=0.8)
    ax4.bar(x, recall_per_class, width, label="Recall", alpha=0.8)
    ax4.bar(x + width, f1_per_class, width, label="F1-Score", alpha=0.8)

    class_names = ["Flat", "Up", "Down"]
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Score")
    ax4.set_title("Per-Class Metrics")
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names[: len(precision_per_class)])
    ax4.legend()
    ax4.set_ylim([0, 1])

    # Prediction confidence distribution
    ax5 = plt.subplot(2, 3, 5)
    if y_proba is not None:
        max_probs = np.max(y_proba, axis=1) if y_proba.ndim > 1 else y_proba
        ax5.hist(max_probs, bins=50, alpha=0.7, edgecolor="black")
        ax5.set_xlabel("Maximum Prediction Probability")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Model Confidence Distribution")
        ax5.axvline(0.5, color="red", linestyle="--", label="Threshold")
        ax5.legend()

    # Trading signals analysis (for "Up" class)
    ax6 = plt.subplot(2, 3, 6)
    up_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
    up_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
    up_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0

    metrics = [
        "Precision\n(Buy Accuracy)",
        "Recall\n(Opportunities\nCaptured)",
        "F1-Score",
    ]
    values = [up_precision, up_recall, up_f1]
    colors = ["green" if v > 0.5 else "orange" if v > 0.3 else "red" for v in values]

    bars = ax6.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black")
    ax6.set_ylabel("Score")
    ax6.set_title('CRITICAL: "UP" Class Performance\n(Trading Signal Quality)')
    ax6.set_ylim([0, 1])
    ax6.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_evaluation.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

    return fig


# ==============================================================================
# MODEL EVALUATION FUNCTIONS
# ==============================================================================
def evaluate_lightgbm(X, y):
    """Evaluate LightGBM model"""
    print(f"\n{'=' * 60}")
    print("EVALUATING LIGHTGBM MODEL")
    print(f"{'=' * 60}")

    if not os.path.exists(LIGHTGBM_MODEL_PATH):
        print(f"Error: Model file not found: {LIGHTGBM_MODEL_PATH}")
        return None

    # Load model
    print(f"Loading model from {LIGHTGBM_MODEL_PATH}...")
    model = lgb.Booster(model_file=LIGHTGBM_MODEL_PATH)

    # Predict
    print("Making predictions...")
    y_proba = model.predict(X)
    y_pred = np.argmax(y_proba, axis=1)

    # Calculate metrics
    results = calculate_metrics(y.values, y_pred, y_proba, "LightGBM")

    # Print report
    print_evaluation_report(results, "LightGBM")

    # Classification report
    print("Detailed Classification Report:")
    print(classification_report(y.values, y_pred, target_names=["Flat", "Up", "Down"]))

    # Visualize
    plot_evaluation_results(y.values, y_pred, y_proba, "LightGBM")

    return results


def evaluate_lstm(X, y, seq_length=48):
    """Evaluate LSTM model (3_LSTM_Large uses seq_length=48)"""
    print(f"\n{'=' * 60}")
    print("EVALUATING LSTM MODEL")
    print(f"{'=' * 60}")

    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"Error: Model file not found: {LSTM_MODEL_PATH}")
        return None

    # Prepare sequences
    print("Creating sequences...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    X_seq, y_seq = create_sequences(X_scaled, y.values, seq_length)

    if len(X_seq) == 0:
        print("Error: Not enough data to create sequences")
        return None

    print(f"Created {len(X_seq)} sequences")

    # Convert to tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)

    # Load model with exact "3_LSTM_Large" configuration
    input_dim = X_seq.shape[2]

    # Use exact architecture from training: 3_LSTM_Large
    hidden_dim = 128
    num_layers = 3
    dropout = 0.3

    try:
        model = CryptoLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3,
            dropout=dropout,
        ).to(DEVICE)

        model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
        print(
            f"Successfully loaded LSTM model (3_LSTM_Large: hidden={hidden_dim}, layers={num_layers})"
        )
    except Exception as e:
        print(f"Error: Could not load LSTM model: {e}")
        return None

    # Predict
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        batch_size = 1024
        y_proba_list = []

        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_proba_list.append(probs)

        y_proba = np.vstack(y_proba_list)

    y_pred = np.argmax(y_proba, axis=1)

    # Calculate metrics
    results = calculate_metrics(y_seq, y_pred, y_proba, "LSTM")

    # Print report
    print_evaluation_report(results, "LSTM")

    # Classification report
    print("Detailed Classification Report:")
    print(classification_report(y_seq, y_pred, target_names=["Flat", "Up", "Down"]))

    # Visualize
    plot_evaluation_results(y_seq, y_pred, y_proba, "LSTM")

    return results


def evaluate_transformer(X, y, seq_length=72):
    """Evaluate Transformer model (3_Transformer_Deep uses seq_length=72)"""
    print(f"\n{'=' * 60}")
    print("EVALUATING TRANSFORMER MODEL")
    print(f"{'=' * 60}")

    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        print(f"Error: Model file not found: {TRANSFORMER_MODEL_PATH}")
        return None

    # Prepare sequences
    print("Creating sequences...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # For transformer, use sparse sampling if needed
    # Simplified: use last seq_length points
    if len(X_scaled) < seq_length:
        print(f"Warning: Data length {len(X_scaled)} < seq_length {seq_length}")
        seq_length = len(X_scaled) // 2

    X_seq, y_seq = create_sequences(X_scaled, y.values, seq_length)

    if len(X_seq) == 0:
        print("Error: Not enough data to create sequences")
        return None

    print(f"Created {len(X_seq)} sequences")

    # Convert to tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)

    # Load model with exact "3_Transformer_Deep" configuration
    input_dim = X_seq.shape[2]

    # Use exact architecture from training: 3_Transformer_Deep
    d_model = 128
    nhead = 8
    num_layers = 3
    dropout = 0.3

    try:
        model = CryptoTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=3,  # Multiclass: 0=Flat, 1=Up, 2=Down
            dropout=dropout,
            max_len=500,
        ).to(DEVICE)

        model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE))
        print(
            f"Successfully loaded Transformer model (3_Transformer_Deep: d_model={d_model}, nhead={nhead}, layers={num_layers})"
        )
    except Exception as e:
        print(f"Error: Could not load Transformer model: {e}")
        return None

    # Predict (multiclass output)
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        batch_size = 256
        y_proba_list = []

        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_proba_list.append(probs)

        y_proba = np.vstack(y_proba_list)

    y_pred = np.argmax(y_proba, axis=1)

    # Calculate metrics
    results = calculate_metrics(y_seq, y_pred, y_proba, "Transformer")

    # Print report
    print_evaluation_report(results, "Transformer")

    # Classification report
    print("Detailed Classification Report:")
    print(classification_report(y_seq, y_pred, target_names=["Flat", "Up", "Down"]))

    # Visualize
    plot_evaluation_results(y_seq, y_pred, y_proba, "Transformer")

    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 60)
    print("CRYPTOCURRENCY TRADING MODEL EVALUATION")
    print("=" * 60)
    print(f"Ticker: {TICKER}")
    print(f"Period: {TIME_START} to {TIME_END}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    try:
        df = load_evaluation_data(TICKER, TIME_START, TIME_END, INTERVAL)
        print(f"\nLoaded {len(df)} rows of data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare features
    try:
        X, y = prepare_lightgbm_features(df)
        print(f"Prepared features: {X.shape}, targets: {y.shape}")
        print(f"Target distribution: {np.bincount(y)}")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return

    # Evaluate models
    all_results = {}

    # LightGBM
    try:
        results = evaluate_lightgbm(X, y)
        if results:
            all_results["LightGBM"] = results
    except Exception as e:
        print(f"Error evaluating LightGBM: {e}")

    # LSTM (3_LSTM_Large)
    try:
        results = evaluate_lstm(X, y, seq_length=48)  # Match training seq_length
        if results:
            all_results["LSTM"] = results
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")

    # Transformer (3_Transformer_Deep)
    try:
        results = evaluate_transformer(X, y, seq_length=72)  # Match training seq_length
        if results:
            all_results["Transformer"] = results
    except Exception as e:
        print(f"Error evaluating Transformer: {e}")

    # Compare models
    if all_results:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)

        comparison_df = pd.DataFrame(all_results).T
        print(comparison_df.to_string())

        # Save to CSV
        output_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
        comparison_df.to_csv(output_csv)
        print(f"\nSaved comparison to {output_csv}")

        # Highlight best models
        print("\n" + "=" * 60)
        print("BEST MODELS BY METRIC")
        print("=" * 60)

        for metric in ["precision_up", "recall_up", "f1_up", "auc_roc_macro"]:
            if metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmax()
                best_value = comparison_df[metric].max()
                print(f"{metric:20s}: {best_model:15s} ({best_value:.4f})")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()
