import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_FILE = "/kaggle/input/btc-and-eth/BTCUSDT_5m_features.csv"

# --- KEY CHANGES FOR SPARSE SAMPLING ---
INPUT_SEQ_LEN = 1000  # We want exactly 1000 time steps as input to the model
MAX_HISTORY_LOOKBACK = 20000  # How far back in real time to look (20k * 5m = ~70 days)
HORIZON = 60  # Predict 60 steps ahead
# ---------------------------------------

BATCH_SIZE = 256
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 5
LEARNING_RATE = 0.001
TARGET_COL_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. GENERATE SPARSE INDICES (Logarithmic)
# ==========================================
def get_log_offsets(seq_len, max_lookback):
    # 1. Generate Logarithmic scale
    # We use geomspace to get a curve that is dense near 0 and sparse near max_lookback
    raw_offsets = np.geomspace(1, max_lookback + 1, num=seq_len) - 1
    offsets = np.floor(raw_offsets).astype(int)

    # 2. Sort Descending (Oldest -> Newest)
    # Shape: [Big Numbers ... Small Numbers]
    offsets = np.sort(offsets)[::-1]

    # 3. Fix Duplicates: Force strictly decreasing steps
    # We iterate from the NEWEST (end of array) back to the OLDEST (start of array).
    # Logic: "The current step (i) must be at least 1 unit larger than the next step (i+1)"

    # Optional: Force the very last input to be exactly 'now' (offset 0)
    if offsets[-1] != 0:
        offsets[-1] = 0

    for i in range(len(offsets) - 2, -1, -1):
        if offsets[i] <= offsets[i + 1]:
            # Push the current value up so it is distinct from the one after it
            offsets[i] = offsets[i + 1] + 1

    # 4. Safety Clip (in case incrementing pushed the oldest values beyond max_lookback)
    offsets = np.clip(offsets, 0, max_lookback)

    return offsets.copy()


# Visualize the sampling strategy
offsets = get_log_offsets(INPUT_SEQ_LEN, MAX_HISTORY_LOOKBACK)

plt.figure(figsize=(12, 4))
plt.plot(offsets, "o", markersize=2)
plt.title(
    f"Sparse Sampling Pattern: {INPUT_SEQ_LEN} inputs covering {MAX_HISTORY_LOOKBACK} steps"
)
plt.xlabel("Input Step Index (0=Oldest, 999=Newest)")
plt.ylabel("Real Time Offset (Steps back from Now)")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Example Offsets (First 10 - Oldest): {offsets[:10]}")
print(f"Example Offsets (Last 10 - Newest): {offsets[-10:]}")

# ==========================================
# 3. DATA LOADING
# ==========================================
df = pd.read_csv(CSV_FILE).dropna().reset_index(drop=True)
DROP_COLS = ["TARGET_class", "TARGET_next_return", "__TARGET__"]
features = [c for c in df.columns if c not in DROP_COLS]
data_numpy = df[features].values

# Split
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

        # We need the maximum offset to know where we can start iterating
        self.max_lookback = offsets.max().item()

    def __len__(self):
        # We start at max_lookback so we can look back enough
        # We end before length - horizon to have targets
        return len(self.data) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        # 'current_time' is the point where we make the prediction
        current_time_idx = idx + self.max_lookback

        # 1. INPUT: Select rows based on the sparse offsets
        # The offsets are [Big, ..., 0].
        # So specific indices = current_time - offset
        indices = current_time_idx - self.offsets
        x = self.data[indices]  # Shape: (1000, Features)

        # 2. TARGET: Look ahead from current_time_idx
        start_future = current_time_idx
        end_future = current_time_idx + self.horizon

        future_subset = self.data[start_future:end_future, self.target_col_idx]
        cumulative_return = torch.sum(future_subset)

        label = 1.0 if cumulative_return > 0 else 0.0

        return x, torch.tensor(label, dtype=torch.float32)


# Create Datasets using the pre-calculated log offsets
train_dataset = SparseHorizonDataset(train_data, offsets, HORIZON, TARGET_COL_IDX)
val_dataset = SparseHorizonDataset(val_data, offsets, HORIZON, TARGET_COL_IDX)
test_dataset = SparseHorizonDataset(test_data, offsets, HORIZON, TARGET_COL_IDX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(
    f"Dataset created. Input shape: ({BATCH_SIZE}, {INPUT_SEQ_LEN}, {train_data.shape[1]})"
)


# ==========================================
# 5. MODEL (Standard LSTM)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch, seq_len (1000), features]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use the last state


model = LSTMClassifier(train_data.shape[1], HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
train_losses = []
val_losses = []

print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

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
plt.title("Loss Curve")
plt.legend()
plt.show()

# ==========================================
# 7. TEST EVALUATION
# ==========================================
model.eval()
all_probs = []
all_labels = []

print("Running evaluation on Test Set...")

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        # Squeeze labels if necessary to match output shape
        labels = labels.to(device)

        outputs = model(inputs)

        # Move to CPU for metrics
        probs = outputs.cpu().numpy()
        lbls = labels.cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(lbls)

# Convert to numpy arrays
all_probs = np.array(all_probs).flatten()
all_labels = np.array(all_labels).flatten()

# Convert probabilities to binary class predictions (Threshold = 0.5)
predictions = (all_probs > 0.5).astype(int)

# --- METRICS ---
print("\n" + "=" * 40)
print("TEST SET RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy_score(all_labels, predictions):.4f}")
print("-" * 40)
print("Classification Report:")
print(classification_report(all_labels, predictions, target_names=["Down/Flat", "Up"]))

# --- VISUALIZATION ---
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 1. Confusion Matrix
cm = confusion_matrix(all_labels, predictions)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax[0],
    xticklabels=["Pred Down", "Pred Up"],
    yticklabels=["Actual Down", "Actual Up"],
)
ax[0].set_title("Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

# 2. Probability Distribution Histogram
# This tells us if the model is confident (peaks at 0 and 1) or unsure (peak at 0.5)
ax[1].hist(all_probs, bins=50, color="purple", alpha=0.7, edgecolor="black")
ax[1].set_title("Prediction Probability Distribution")
ax[1].set_xlabel("Predicted Probability ( > 0.5 = UP)")
ax[1].set_ylabel("Count")
ax[1].axvline(0.5, color="red", linestyle="--", label="Threshold")
ax[1].legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import auc, roc_curve

# --- ROC CURVE ---
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# --- SNAPSHOT PREDICTIONS ---
print("\nSample Predictions vs Actuals (First 10):")
for i in range(10):
    status = "CORRECT" if predictions[i] == all_labels[i] else "WRONG"
    print(
        f"Prob: {all_probs[i]:.4f} | Pred: {predictions[i]} | Actual: {int(all_labels[i])} | {status}"
    )


# Define the path to save the model
output_dir = "/kaggle/working/"
model_save_path = os.path.join(output_dir, "lstm_model.pth")

# Ensure the directory exists (optional, but good practice)
os.makedirs(output_dir, exist_ok=True)

# Save the model's state_dict
torch.save(model.state_dict(), model_save_path)

print(f"Model saved successfully to {model_save_path}")
