import gc
import glob
import os
import random
import time
import warnings
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
#  0. CONFIG & GPU SETUP
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

SEEDS = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEEDS)


# ==============================================================================
#  1. MODEL ARCHITECTURE (LSTM)
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
#  2. DATA PROCESSING FUNCTION
# ==============================================================================
def create_sequences(features, targets, seq_length):
    """
    Converts 2D array into 3D sequences (Samples, TimeSteps, Features)
    """
    xs, ys = [], []
    # We need enough data for at least one sequence
    if len(features) <= seq_length:
        return np.array(xs), np.array(ys)

    for i in range(len(features) - seq_length):
        x_seq = features[i : (i + seq_length)]
        y_target = targets[i + seq_length]  # Predict the target at the end of sequence
        xs.append(x_seq)
        ys.append(y_target)

    return np.array(xs), np.array(ys)


def process_ticker_batch_lstm(file_paths, seq_length=24, scaler=None, is_training=True):
    all_seq_x = []
    all_seq_y = []

    # We need a temporary list to fit the scaler if it's not provided
    temp_features_for_scaling = []
    raw_data_list = []

    # 1. Load and Clean Data
    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)

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
            if not valid_aggs:
                continue

            df_res = df.resample("1h").agg(valid_aggs)
            if len(df_res) < 200:
                continue

            # --- FEATURES ---
            df_res["ret_1h"] = df_res["close"].pct_change()
            df_res["std_24h"] = df_res["ret_1h"].rolling(24).std()
            ma_24 = df_res["close"].rolling(24).mean()
            df_res["dist_ma_24"] = (df_res["close"] - ma_24) / ma_24

            # Additional lags used in original
            df_res["ret_lag_1"] = df_res["ret_1h"].shift(1)
            df_res["ret_lag_2"] = df_res["ret_1h"].shift(2)
            df_res["ret_lag_6"] = df_res["ret_1h"].shift(6)

            # --- TARGETS ---
            future_ret = df_res["close"].pct_change(24).shift(-24)
            threshold = (df_res["std_24h"] * np.sqrt(24) * 0.4).fillna(0)

            conditions = [future_ret > threshold, future_ret < -threshold]
            choices = [1, 2]  # 1=Up, 2=Down, 0=Flat
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

            # Keep only valid cols
            final_cols = [c for c in feature_cols if c in df_res.columns]

            if len(df_res) < seq_length + 10:
                continue

            # Store for scaling/sequencing
            raw_data_list.append((df_res[final_cols].values, df_res["target"].values))

            if is_training and scaler is None:
                temp_features_for_scaling.append(df_res[final_cols].values)

        except Exception:
            continue

    if not raw_data_list:
        return None, None, scaler

    # 2. Fit Scaler (If training and no scaler exists)
    if is_training and scaler is None:
        scaler = StandardScaler()
        if temp_features_for_scaling:
            # Stack roughly 10% of data to fit scaler to save memory, or all if small
            big_chunk = np.vstack(temp_features_for_scaling)
            scaler.fit(big_chunk)

    # 3. Create Sequences
    for feats, targs in raw_data_list:
        # Scale features
        if scaler:
            feats = scaler.transform(feats)

        # Create sequences per ticker to avoid boundary leakage
        X_seq, y_seq = create_sequences(feats, targs, seq_length)

        if len(X_seq) > 0:
            all_seq_x.append(X_seq)
            all_seq_y.append(y_seq)

    if not all_seq_x:
        return None, None, scaler

    # Concatenate all sequences
    X_final = np.concatenate(all_seq_x, axis=0)
    y_final = np.concatenate(all_seq_y, axis=0)

    # 4. Undersampling (Flattening the 'Flat' class)
    # We do this AFTER sequencing.
    if is_training:
        mask_flat = y_final == 0
        mask_trend = y_final != 0

        # Get indices
        idx_flat = np.where(mask_flat)[0]
        idx_trend = np.where(mask_trend)[0]

        # Sample 50% of flats
        if len(idx_flat) > 0:
            keep_flat = np.random.choice(
                idx_flat, size=int(len(idx_flat) * 0.5), replace=False
            )
            keep_indices = np.concatenate([keep_flat, idx_trend])
            np.random.shuffle(keep_indices)  # Shuffle to mix

            X_final = X_final[keep_indices]
            y_final = y_final[keep_indices]

    # Convert to Tensors
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y_final, dtype=torch.long)

    return X_tensor, y_tensor, scaler


# ==============================================================================
#  3. HYPERPARAMETER GRID (REDUCED TO 3 SETS)
# ==============================================================================

param_grid = [
    # Set 1: Small/Fast (Like "UltraSafe")
    {
        "name": "1_LSTM_Small_Reg",
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.005,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "seq_len": 12,
    },
    # Set 2: Medium/Standard (Like "Balanced")
    {
        "name": "2_LSTM_Medium",
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "seq_len": 24,
    },
    # Set 3: Large/Deep (Like "Aggressive")
    {
        "name": "3_LSTM_Large",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "batch_size": 256,
        "seq_len": 48,
    },
]

# Get Files
all_files = []
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".parquet"):
            all_files.append(os.path.join(dirname, filename))

random.shuffle(all_files)
# all_files = all_files[:200]  # Limit files for demo speed
print(f"Total Files: {len(all_files)}")

# ==============================================================================
#  4. TRAINING LOOP
# ==============================================================================

best_overall_model = None
best_overall_scaler = None
best_overall_loss = float("inf")
best_experiment_name = ""
final_history = {}

FILE_BATCH_SIZE = 10  # Process 10 parquet files per step

for experiment in param_grid:
    print(f"\n⚡ STARTING EXPERIMENT: {experiment['name']}")
    print("-" * 75)

    # Initialize Model
    # Determine input dim based on feature columns (8 in our function)
    INPUT_DIM = 8

    model = CryptoLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=experiment["hidden_dim"],
        num_layers=experiment["num_layers"],
        output_dim=3,  # Flat, Up, Down
        dropout=experiment["dropout"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=experiment["learning_rate"],
        weight_decay=experiment["weight_decay"],
    )

    # Training State
    scaler = None
    train_losses = []
    valid_losses = []
    rolling_valid_loss = deque(maxlen=3)
    experiment_failed = False
    batches_processed = 0

    # Shuffle files
    random.shuffle(all_files)

    start_time = time.time()

    # --- ITERATE THROUGH FILE BATCHES ---
    for i in range(0, len(all_files), FILE_BATCH_SIZE):
        batch_files = all_files[i : i + FILE_BATCH_SIZE]

        # Process Data
        # Note: We pass the scaler. If None (first batch), it creates one.
        X, y, scaler = process_ticker_batch_lstm(
            batch_files,
            seq_length=experiment["seq_len"],
            scaler=scaler,
            is_training=True,
        )

        if X is None or len(X) < 100:
            continue

        # Train/Valid Split
        split = int(len(X) * 0.9)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # Create DataLoaders
        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_ds, batch_size=experiment["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=experiment["batch_size"], shuffle=False
        )

        # --- INNER EPOCH LOOP (Incremental) ---
        # We do a few epochs on this specific chunk of data
        EPOCHS_PER_CHUNK = 3

        chunk_train_loss = 0
        model.train()

        for epoch in range(EPOCHS_PER_CHUNK):
            epoch_loss = 0
            count = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                count += 1
            chunk_train_loss = epoch_loss / count if count > 0 else 0

        # --- VALIDATION ---
        model.eval()
        val_loss_sum = 0
        val_count = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss_sum += loss.item()
                val_count += 1

        chunk_val_loss = val_loss_sum / val_count if val_count > 0 else 0.0

        # Record Metrics
        train_losses.append(chunk_train_loss)
        valid_losses.append(chunk_val_loss)
        rolling_valid_loss.append(chunk_val_loss)
        avg_rolling = np.mean(rolling_valid_loss)

        batches_processed += 1

        print(
            f"{batches_processed:<5} | Train: {chunk_train_loss:.4f} | Valid: {chunk_val_loss:.4f} | Avg: {avg_rolling:.4f}"
        )

        # --- EARLY STOPPING LOGIC ---
        # Rule A: Stability Check
        if batches_processed >= 4:
            if avg_rolling > 1.09:  # Slightly higher tolerance for LSTM start
                print(f"  ❌ STOP: Diverged/No Signal. Avg Loss {avg_rolling:.4f}")
                experiment_failed = True
                break

        # Rule B: Signal Detected
        if chunk_val_loss < 1.00:
            print("  >>> SIGNAL DETECTED (< 1.00) <<<")

        # GC
        del X, y, X_train, y_train, train_loader, valid_loader
        gc.collect()

    # --- END EXPERIMENT ---
    elapsed = time.time() - start_time
    if len(valid_losses) > 0:
        final_score = np.mean(valid_losses[-3:])
    else:
        final_score = float("inf")

    print(f"  🏁 Finished {experiment['name']}")
    print(f"  Time: {elapsed:.1f}s | Final Score: {final_score:.4f}")

    if not experiment_failed and final_score < best_overall_loss:
        print("  🏆 NEW BEST MODEL!")
        best_overall_loss = final_score
        best_overall_model = model
        best_overall_scaler = scaler
        best_experiment_name = experiment["name"]
        final_history = {"train": train_losses, "valid": valid_losses}

# ==============================================================================
#  5. SAVE & PLOT
# ==============================================================================
if best_overall_model:
    print(f"\nSaving Best Model: {best_experiment_name}")
    torch.save(best_overall_model.state_dict(), "best_crypto_lstm.pth")

    plt.figure(figsize=(12, 6))
    plt.plot(final_history["train"], label="Train Loss", alpha=0.6)
    plt.plot(final_history["valid"], label="Valid Loss", linewidth=2, color="red")
    plt.axhline(y=1.05, color="orange", linestyle="--")
    plt.title(f"Training History: {best_experiment_name}")
    plt.legend()
    plt.show()
else:
    print("All experiments failed.")
    exit()

# ==============================================================================
#  6. TEST ON UNSEEN DATA
# ==============================================================================
print("\n" + "=" * 60)
print("🧪 TESTING BEST MODEL ON UNSEEN DATA")
print("=" * 60)

num_test_files = int(len(all_files) * 0.2)
if num_test_files < 1:
    num_test_files = 1
test_files = all_files[-num_test_files:]

# Retrieve best seq_len from name (simple parsing or look up)
best_params = next(p for p in param_grid if p["name"] == best_experiment_name)
SEQ_LEN = best_params["seq_len"]

# Process Test Data (Reuse Scaling)
X_test, y_test, _ = process_ticker_batch_lstm(
    test_files,
    seq_length=SEQ_LEN,
    scaler=best_overall_scaler,  # IMPORTANT: Use training scaler
    is_training=False,  # No undersampling
)

if X_test is None:
    print("Not enough test data.")
else:
    best_overall_model.eval()

    # Predict in batches to avoid OOM
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            outputs = best_overall_model(xb)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.numpy())

    y_pred_prob = np.array(all_probs)
    y_pred_class = np.array(all_preds)
    y_true = np.array(all_targets)

    # Metrics
    final_log_loss = log_loss(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_class)

    print(f"\n📊 RESULTS:")
    print(f"  Final Log Loss: {final_log_loss:.4f}")
    print(f"  Overall Accuracy: {acc:.2%}")

    target_names = ["Flat (0)", "Up (1)", "Down (2)"]
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred_class, target_names=target_names))

    # Trading Reality Check
    cm = confusion_matrix(y_true, y_pred_class)

    up_precision = 0
    if (cm[1, 1] + cm[0, 1] + cm[2, 1]) > 0:
        up_precision = cm[1, 1] / (cm[0, 1] + cm[1, 1] + cm[2, 1])

    down_precision = 0
    if (cm[2, 2] + cm[0, 2] + cm[1, 2]) > 0:
        down_precision = cm[2, 2] / (cm[0, 2] + cm[1, 2] + cm[2, 2])

    print("\n💰 TRADING REALITY CHECK:")
    print(f"  'UP' Signal Precision:   {up_precision:.2%} (Target > 55%)")
    print(f"  'DOWN' Signal Precision: {down_precision:.2%} (Target > 55%)")
