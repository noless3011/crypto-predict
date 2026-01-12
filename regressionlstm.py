# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# %%
# ==========================================
# 2. Feature Engineering (Streamlined / Strong Features Only)
# ==========================================
# Assuming the file path is correct for the Kaggle environment
df = pd.read_parquet('/kaggle/input/crypto-2/processed_data/BTCUSDT.parquet')

# 2.1 Imputation
cols_to_fix = ['open', 'high', 'low', 'close', 'volume']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan).interpolate(method='linear')
df.dropna(inplace=True)

# 2.2 Indicator Generation (Stationary Features Only)

# --- Log Returns (The most important feature) ---
# Replaces raw price. Tells the model the immediate momentum.
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# --- RSI (Relative Strength Index) ---
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# --- Volatility ---
df['Volatility'] = df['close'].pct_change().rolling(window=20).std()

# --- Log Volume ---
df['Log_Volume'] = np.log1p(df['volume'])

# --- Price Distance from Moving Average ---
# Replaces raw 'Close'. Tells model if we are overextended.
df['MA_50'] = df['close'].rolling(window=50).mean()
df['Dist_MA50'] = (df['close'] - df['MA_50']) / df['MA_50']

# 2.3 Target Engineering (Regression Target)
HORIZON = 4 # e.g., 4 Steps ahead (if 1h data -> 4h, if 5m data -> 20m)

# Calculate the return HORIZON steps into the future
# formula: log( Price(t+H) / Price(t) )
# This is what we want to predict: The magnitude and direction of the move.
df['future_ret'] = np.log(df['close'].shift(-HORIZON) / df['close'])

# Set Target directly to future_ret (Continuous value for Regression)
df['target'] = df['future_ret']

# Drop NaNs created by rolling windows and shifting
df.dropna(inplace=True)

# %%
# ==========================================
# 3. Data Preparation & Normalization
# ==========================================

feature_cols = [
    'log_ret', 
    'Dist_MA50',
    'Log_Volume', 
    'RSI', 
    'Volatility'
]

# Split Data *BEFORE* Scaling to prevent leakage
train_split = 0.8
split_idx = int(len(df) * train_split)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Scale Features
scaler = StandardScaler()

# FIT on Train, TRANSFORM both (Standard ML practice)
train_features = scaler.fit_transform(train_df[feature_cols])
test_features = scaler.transform(test_df[feature_cols])

# Extract Targets (No scaling for now, but usually valid for log-ret)
# If targets are very small (e.g. 1e-5), consider scaling y as well, but log-ret is usually ~1e-2 which is OK.
train_targets = train_df['target'].values
test_targets = test_df['target'].values

# %%
# ==========================================
# 4. Sequence Generation
# ==========================================
# Recommendation: 120 for 5m/15m data to capture enough context
seq_length = 120  

def create_sequences(features, targets, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i : i + seq_length]
        y = targets[i + seq_length - 1] # Target at the END of the sequence step
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_features, train_targets, seq_length)
X_test, y_test = create_sequences(test_features, test_targets, seq_length)

# Features: Float32
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Targets: Float32, Shape [Batch, 1] for regression
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"Train Input Shape: {X_train_tensor.shape}") # (Samples, Seq_Len, n_features)
print(f"Train Label Shape: {y_train_tensor.shape}") # (Samples, 1)

batch_size = 64 

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# ==========================================
# 5. Define LSTM Regression Model
# ==========================================
class CryptoLSTMRegression(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        
        # --- Layer 1: Catch trends (Larger) ---
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=64, 
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # --- Layer 2: Compress features (The "Funnel") ---
        self.lstm2 = nn.LSTM(
            input_size=64, 
            hidden_size=32, 
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # --- Dense Layers (Head) ---
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        
        # Final prediction (1 neuron) - NO ACTIVATION for Regression (Linear)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Pass through Layer 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Pass through Layer 2
        out, _ = self.lstm2(out)
        
        # Take only the LAST time step
        out = out[:, -1, :]
        out = self.dropout2(out)
        
        # Dense processing
        out = self.fc1(out)
        out = self.relu(out)
        
        # Final Output (Linear)
        out = self.fc2(out)
        return out

# --- Configuration ---
input_dim = X_train.shape[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CryptoLSTMRegression(
    input_dim=input_dim,
    dropout=0.2 
).to(device)

# %%
# ==========================================
# 6. Training with MSE Loss
# ==========================================

# Use MSELoss for Regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

best_val_loss = float('inf')
patience = 10  
counter = 0

print("Starting Regression Training...")

for epoch in range(epochs):
    # --- PHASE 1: TRAINING ---
    model.train() 
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # --- PHASE 2: VALIDATION ---
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad(): 
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(test_loader)

    # --- PHASE 3: CHECKPOINTING ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_crypto_reg_model.pth')
        print(f"Epoch {epoch+1} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} *SAVED*")
        counter = 0 
    else:
        print(f"Epoch {epoch+1} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        counter += 1
        
    # --- PHASE 4: EARLY STOPPING ---
    if counter >= patience:
        print(f"Early stopping triggered! No improvement for {patience} epochs.")
        break

print("Training Complete.")

# Load best weights
model.load_state_dict(torch.load('best_crypto_reg_model.pth'))

# %%
# ==========================================
# 7. Elevation (Regression Metrics)
# ==========================================

model.eval()
all_preds = []
all_labels = []

print("Running Inference on Test Set...")

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        
        # Store results
        all_preds.extend(outputs.cpu().numpy().flatten())
        all_labels.extend(batch_y.cpu().numpy().flatten())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# --- Metrics ---
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n" + "="*40)
print("       REGRESSION REPORT")
print("="*40)
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R2 Score: {r2:.4f}")  # 1.0 is perfect, 0.0 is dummy mean, Negative is worse than mean

# ==========================================
# 8. Visualizations
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot A: Actual vs Predicted Scatter ---
# Good Model: Points clustered around the diagonal line
axes[0].scatter(y_true, y_pred, alpha=0.3, color='blue', s=10)
axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
axes[0].set_title(f'Actual vs Predicted (R2={r2:.2f})')
axes[0].set_xlabel('Actual Log Return')
axes[0].set_ylabel('Predicted Log Return')

# --- Plot B: Prediction Snapshot (Zoom in) ---
# See how well it tracks a slice of time
zoom_n = 200 # First 200 points of test set
axes[1].plot(y_true[:zoom_n], label='Actual', color='black', alpha=0.7)
axes[1].plot(y_pred[:zoom_n], label='Predicted', color='green', alpha=0.7)
axes[1].set_title(f'Forecast Snapshot (First {zoom_n} test steps)')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Log Return')
axes[1].legend()

plt.tight_layout()
plt.show()
