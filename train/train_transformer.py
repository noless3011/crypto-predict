import math
import os

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
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_FILE = "/kaggle/input/btc-and-eth/BTCUSDT_5m_features.csv"

INPUT_SEQ_LEN = 1000
MAX_HISTORY_LOOKBACK = 20000
HORIZON = 12 * 24 * 7

# Transformer Hyperparameters
BATCH_SIZE = 128  # Reduced slightly to fit Transformer memory if needed
D_MODEL = 128  # Embedding dimension
NHEAD = 4  # Number of attention heads
NUM_LAYERS = 2  # Number of encoder layers
DIM_FEEDFORWARD = 512  # Internal dimension of FFN
DROPOUT = 0.2
EPOCHS = 5
LEARNING_RATE = 0.0001  # Transformers often prefer lower LR than LSTMs
TARGET_COL_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. GENERATE SPARSE INDICES (Logarithmic)
# ==========================================
def get_log_offsets(seq_len, max_lookback):
    raw_offsets = np.geomspace(1, max_lookback + 1, num=seq_len) - 1
    offsets = np.floor(raw_offsets).astype(int)
    offsets = np.sort(offsets)[::-1]

    if offsets[-1] != 0:
        offsets[-1] = 0

    for i in range(len(offsets) - 2, -1, -1):
        if offsets[i] <= offsets[i + 1]:
            offsets[i] = offsets[i + 1] + 1

    offsets = np.clip(offsets, 0, max_lookback)
    return offsets.copy()


offsets = get_log_offsets(INPUT_SEQ_LEN, MAX_HISTORY_LOOKBACK)

# ==========================================
# 3. DATA LOADING
# ==========================================
# (Assuming the file exists in your environment; creates dummy data if not for testing)
try:
    df = pd.read_csv(CSV_FILE).dropna().reset_index(drop=True)
    DROP_COLS = ["TARGET_class", "TARGET_next_return", "__TARGET__"]
    features = [c for c in df.columns if c not in DROP_COLS]
    data_numpy = df[features].values
except FileNotFoundError:
    print("CSV not found, generating dummy data for demonstration...")
    data_numpy = np.random.randn(30000, 20)  # 30k rows, 20 features
    features = [f"feat_{i}" for i in range(20)]

n = len(data_numpy)
train_size = int(n * 0.70)
val_size = int(n * 0.15)

train_raw = data_numpy[:train_size]
val_raw = data_numpy[train_size : train_size + val_size]
test_raw = data_numpy[train_size + val_size :]

scaler = StandardScaler()
train_data = scaler.fit_transform(train_raw)
val_data = scaler.transform(val_raw)
test_data = scaler.transform(test_raw)


# ==========================================
# 4. SPARSE HORIZON DATASET
# ==========================================
class SparseHorizonDataset(Dataset):
    def __init__(self, data, offsets, horizon, target_col_idx):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.offsets = torch.tensor(offsets, dtype=torch.long)
        self.horizon = horizon
        self.target_col_idx = target_col_idx
        self.max_lookback = offsets.max().item()

    def __len__(self):
        return len(self.data) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        current_time_idx = idx + self.max_lookback

        # Inputs based on offsets
        indices = current_time_idx - self.offsets
        x = self.data[indices]

        # Target
        start_future = current_time_idx
        end_future = current_time_idx + self.horizon
        future_subset = self.data[start_future:end_future, self.target_col_idx]
        cumulative_return = torch.sum(future_subset)
        label = 1.0 if cumulative_return > 0 else 0.0

        return x, torch.tensor(label, dtype=torch.float32)


train_dataset = SparseHorizonDataset(train_data, offsets, HORIZON, TARGET_COL_IDX)
val_dataset = SparseHorizonDataset(val_data, offsets, HORIZON, TARGET_COL_IDX)
test_dataset = SparseHorizonDataset(test_data, offsets, HORIZON, TARGET_COL_IDX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. MODEL (Transformer)
# ==========================================


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


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()

        # 1. Input Embedding: Project features to d_model size
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=INPUT_SEQ_LEN + 100, dropout=dropout
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Crucial: inputs are [Batch, Seq, Feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Output Head
        # We take the last time step output to predict
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        # Project
        x = self.input_projection(x)  # -> [batch, seq_len, d_model]

        # Add Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Pass
        # output: [batch, seq_len, d_model]
        output = self.transformer_encoder(x)

        # We assume the last element in the sequence (index -1) is the most recent "current" state
        # in your sparse sampling (offsets sorted Big -> Small).
        last_step_output = output[:, -1, :]

        return self.fc(last_step_output)


model = TimeSeriesTransformer(
    input_dim=train_data.shape[1],
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
).to(device)

criterion = nn.BCELoss()
# Transformers usually benefit from Weight Decay (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print(
    f"Model created: Transformer with {sum(p.numel() for p in model.parameters())} parameters."
)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
train_losses = []
val_losses = []

print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False
    )

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            val_bar.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

    epoch_train_loss = running_loss / len(train_loader)
    epoch_val_loss = val_loss / len(val_loader)
    acc = correct / total

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch {epoch + 1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {acc:.4f}"
    )

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.title("Transformer Loss Curve")
plt.legend()
plt.show()


# ==========================================
# 7. TEST EVALUATION (Transformer)
# ==========================================
model.eval()
y_true = []
y_probs = []

print("Running evaluation on Test Set...")

with torch.no_grad():
    for inputs, labels in test_loader:
        # Move inputs to GPU/Device
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)

        # Flatten outputs (Batch, 1) -> (Batch,) and move to CPU
        probs = outputs.squeeze().cpu().numpy()
        lbls = labels.cpu().numpy()

        # Handle case if batch_size=1 (numpy scalar vs array)
        if probs.ndim == 0:
            probs = np.array([probs])
            lbls = np.array([lbls])

        y_probs.extend(probs)
        y_true.extend(lbls)

# Convert to numpy arrays
y_probs = np.array(y_probs)
y_true = np.array(y_true)

# Apply Threshold (0.5 is standard, but you might tune this based on ROC)
threshold = 0.5
y_pred = (y_probs > threshold).astype(int)

# --- METRICS ---
print("\n" + "=" * 40)
print("TRANSFORMER TEST RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("-" * 40)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Down/Flat", "Up"]))

# --- VISUALIZATION ---
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax[0],
    cbar=False,
    xticklabels=["Pred Down", "Pred Up"],
    yticklabels=["Actual Down", "Actual Up"],
)
ax[0].set_title("Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

# 2. Probability Distribution
# Good separation means the purple bumps are near 0.0 and 1.0, not stuck at 0.5
sns.histplot(
    y_probs, bins=50, kde=True, color="purple", ax=ax[1], edgecolor="black", alpha=0.6
)
ax[1].axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
ax[1].set_title("Prediction Probability Distribution")
ax[1].set_xlabel('Probability of "Up"')
ax[1].legend()

# 3. ROC Curve (Receiver Operating Characteristic)
# Measures performance at ALL classification thresholds.
# AUC = 0.5 is random guessing, AUC = 1.0 is perfect.
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

ax[2].plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
)
ax[2].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax[2].set_xlim([0.0, 1.0])
ax[2].set_ylim([0.0, 1.05])
ax[2].set_xlabel("False Positive Rate")
ax[2].set_ylabel("True Positive Rate")
ax[2].set_title("ROC Curve")
ax[2].legend(loc="lower right")

plt.tight_layout()
plt.show()

# --- SNAPSHOT PREDICTIONS ---
print("\nSample Predictions vs Actuals (First 15):")
print(f"{'Prob (Up)':<12} | {'Pred':<6} | {'Actual':<6} | {'Result'}")
print("-" * 45)
for i in range(15):
    status = "✅" if y_pred[i] == y_true[i] else "❌"
    print(f"{y_probs[i]:.4f}       | {y_pred[i]:<6} | {int(y_true[i]):<6} | {status}")


# Define the path to save the model
output_dir = "/kaggle/working/"
model_save_path = os.path.join(output_dir, "transformer_model.pth")

# Ensure the directory exists (optional, but good practice)
os.makedirs(output_dir, exist_ok=True)

# Save the model's state_dict
torch.save(model.state_dict(), model_save_path)

print(f"Model saved successfully to {model_save_path}")
