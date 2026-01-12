# %%

import os
import sys
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D, RepeatVector, \
    TimeDistributed, LayerNormalization, Dropout, MultiHeadAttention, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



# %%

# ==========================================
# 1. Data Processing Functions
# ==========================================
def add_technical_indicators(df):
    """
    Add technical indicators to dataframe
    Requires: Open, High, Low, Close columns
    """
    print("Adding technical indicators...")
    
    # 1. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    
    # 4. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # 5. ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # 6. Price Rate of Change
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Drop NaN rows created by indicators
    df = df.dropna()
    
    print(f"  ✓ Added {len([col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Date_Str', 'Date_Only', 'Gold']])} indicators")
    
    return df
def read_data(path, dim_type, use_percentage=1, use_indicators=True):
    '''
    Reads data and resamples to HOURLY frequency
    '''
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Standardize column names
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        'datetime': 'Date', 'date': 'Date' 
    })

    # Ensure Date column exists
    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'Date' not in df.columns: 
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        else:
            raise KeyError(f"Could not find Date column. Available: {df.columns}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Original data frequency: {len(df)} rows")
    df = df.set_index('Date')
    df_hourly = df.resample('h').agg({  # Changed from 'D' to 'H'
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum' if 'Volume' in df.columns else 'mean'
    }).dropna()
    df = df_hourly.reset_index()
    print(f"After resampling to hourly: {len(df)} rows")

    if use_indicators and dim_type == 'Multi':
        df = add_technical_indicators(df)
        print(f"After adding indicators: {len(df)} rows")

    
    dates = df['Date'].values
    data_len = df.shape[0]
    data = None

    if dim_type != 'Multi':
        data = df[dim_type].values.reshape((data_len, 1))
    else:
        df["Date_Str"] = df["Date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 1. Define columns to exclude (Non-numeric or Date)
        exclude_cols = ['Date', 'Date_Str', 'Date_Only', 'Close', 'Volume'] 
        
        # 2. Get all other feature columns (Open, High, Low, RSI, MACD, etc.)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 3. Create list with 'Close' first (Target), followed by all features
        # Structure: [Close, Open, High, Low, RSI, MACD, BB_Upper, ...]
        final_cols = ['Close'] + feature_cols
        
        print(f"Features used ({len(final_cols)}): {final_cols}")
        
        # 4. Create the data array
        data = df[final_cols].values
    limit = int(np.floor(data_len * use_percentage))
    
    # Store original Close prices before log transform
    original_close = data[:, 0].copy()
    
    # Convert to log returns
    data = convert_to_log_returns(data)
    limit = int(np.floor(len(data) * use_percentage))
    
    # Dates also shift by 1 due to log returns
    return data[0:limit], limit, dates[1:limit+1], original_close[0:limit+1]

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def data_transform(data, anti=False, scaler=None, method='robust'):
    '''
    Normalization (RobustScaler recommended for financial data with outliers)
    method: 'minmax', 'robust', or 'standard'
    '''
    if not anti:
        scalers = {}
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            if method == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            
            col = data[:, i].reshape(-1, 1)
            normalized_data[:, i] = scaler.fit_transform(col).ravel()
            scalers[i] = scaler
        return normalized_data, scalers
    else:
        if data.ndim == 3: 
            data = data.squeeze(axis=2)
        restored = np.zeros_like(data)
        for i in range(data.shape[1]):
            # Retrieve the correct scaler for this feature
            # Note: This assumes column order is preserved exactly
            column_scaler = scaler[i]
            col = data[:, i].reshape(-1, 1)
            restored[:, i] = column_scaler.inverse_transform(col).ravel()
        return restored

# Add after the data_transform function (around line 100)

def convert_to_log_returns(data):
    """
    Convert price data to log returns
    First column (Close) is converted, rest remain as is
    """
    log_data = data.copy()
    # Convert Close price (column 0) to log returns
    log_data[1:, 0] = np.log(data[1:, 0] / data[:-1, 0])
    # Remove first row (no previous price for log return)
    return log_data[1:]

def inverse_log_returns(log_returns, initial_price):
    """
    Convert log returns back to prices
    """
    prices = np.zeros(len(log_returns) + 1)
    prices[0] = initial_price
    for i in range(len(log_returns)):
        prices[i + 1] = prices[i] * np.exp(log_returns[i])
    return prices[1:]  # Return without initial price

# ==========================================
# 2. Split Sequence (Paper's Method)
# ==========================================

def split_sequence(sequence, dim_type, n_steps_in, n_steps_out):
    '''
    Create sequences exactly like the paper
    CRITICAL: For Multi mode, excludes Close from input (column 0)
    '''
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        if out_end_ix > len(sequence):
            break
        
        if dim_type == 'Multi':
            # PAPER'S METHOD: Use columns 1: (Open, High, Low) as input
            # Predict column 0 (Close)
            seq_x = sequence[i:end_ix, 1:]  
            seq_y = sequence[end_ix:out_end_ix, 0]
        else:
            seq_x = sequence[i:end_ix]
            seq_y = sequence[end_ix:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

# ==========================================
# 3. Model Creation
# ==========================================

def create_transformer_model(input_seq, output_seq, n_features, d_model=64, num_heads=12, ff_dim=32, blocks=3):
    '''
    Transformer model matching paper's architecture
    '''
    inputs = Input(shape=(input_seq, n_features))
    x = Dense(d_model)(inputs)
    
    for _ in range(blocks):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn = Dropout(0.1)(attn)
        x = LayerNormalization(epsilon=1e-6)(x + attn)
        
        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dropout(0.5)(ff)
        ff = Dense(d_model)(ff)
        x = LayerNormalization(epsilon=1e-6)(x + ff)
    
    # Take last timestep for forecasting
    outputs = Dense(output_seq)(x[:, -1, :])
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_model(model_type, n_features, n_steps_in, n_steps_out):
    '''
    Create model exactly matching paper's architecture
    '''
    model = Sequential()
    opt = Adam(learning_rate=0.001)
    
    if model_type == 'LSTM':
        model.add(Input(shape=(n_steps_in, n_features)))
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(Dense(n_steps_out))
        
    elif model_type == 'BD LSTM':
        model.add(Input(shape=(n_steps_in, n_features)))
        model.add(Bidirectional(LSTM(50, activation='sigmoid')))
        model.add(Dense(n_steps_out))
        
    elif model_type == 'ED LSTM':
        model.add(Input(shape=(n_steps_in, n_features)))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.add(Flatten())  # Flatten to get correct shape
        
    elif model_type == 'CNN':
        model.add(Input(shape=(n_steps_in, n_features)))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))
        
    elif model_type == 'Convolutional LSTM':
        model.add(Input(shape=(n_steps_in, n_features)))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(LSTM(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))
        
    elif model_type == 'Transformer':
        return create_transformer_model(n_steps_in, n_steps_out, n_features)
        
    elif model_type == 'MLP':
        model.add(Input(shape=(n_steps_in, n_features)))
        model.add(Dense(20, activation='relu'))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))
    else:
        return None
        
    model.compile(optimizer=opt, loss='mse')
    return model

def train_and_forecast(model, train_X, train_y, test_X, test_y, epochs, verbose):
    '''
    Train model with periodic progress updates and return loss history
    '''
    from tensorflow.keras.callbacks import LambdaCallback, History
    
    # History callback to capture loss
    history = History()
    
    # Callback to show progress every 20 epochs
    progress_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: 
            print(f"      Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f}") 
            if (epoch + 1) % 5 == 0 or epoch == 0 else None
    )
    
    # Train
    model.fit(train_X, train_y, epochs=epochs, batch_size=32, 
              verbose=verbose, callbacks=[progress_callback, history])
    
    # Predict
    train_pred = model.predict(train_X, verbose=verbose)
    test_pred = model.predict(test_X, verbose=verbose)
    
    # Return predictions AND loss history
    return train_pred, test_pred, history.history['loss']


def eval_result(result, target, n_steps_out, is_log_return=False):
    '''
    Evaluate using multiple metrics for both Price and Log Return scales.
    
    If inputs are Log Returns (is_log_return=True), also calculates Direction Accuracy.
    If inputs are Prices (is_log_return=False), only calculates RMSE/MAPE.
    
    Returns a dictionary of metrics.
    '''
    metrics = {}
    
    # --- 1. Basic Metrics (RMSE, MAPE) ---
    # RMSE
    rmse_per_step = []
    for i in range(n_steps_out):
        rmse = np.sqrt(np.mean((result[:, i] - target[:, i]) ** 2))
        rmse_per_step.append(rmse)
    metrics['rmse_avg'] = np.mean(rmse_per_step)
    
    # MAPE (Avoid division by zero)
    mape_per_step = []
    for i in range(n_steps_out):
        # Add epsilon to handle 0 values in target
        mape = np.mean(np.abs((target[:, i] - result[:, i]) / (target[:, i] + 1e-9))) * 100
        mape_per_step.append(mape)
    metrics['mape_avg'] = np.mean(mape_per_step)
    
    # --- 2. Log Return Specific Metrics (Direction Accuracy) ---
    if is_log_return:
        # Direction Accuracy: Compare sign of predicted vs actual log return
        # Shape: (Samples, Steps)
        
        # We need the previous value to determine direction if we were doing Price, 
        # but for Log Returns, the VALUE ITSELF is the change (relative to prev step).
        # So Positive Log Return = Price Up, Negative = Price Down.
        
        # 1. Sign Comparison
        actual_sign = np.sign(target)
        pred_sign = np.sign(result)
        
        # Handle 0 cases if needed (e.g. treat 0 as no change)
        correct_direction = (actual_sign == pred_sign)
        
        # Calculate per step
        da_per_step = np.mean(correct_direction, axis=0) * 100
        metrics['direction_accuracy_avg'] = np.mean(da_per_step)
        metrics['direction_accuracy_first_step'] = da_per_step[0]
        
    return metrics

# ==========================================
# 4. Visualization Functions
# ==========================================
def visualize_best_model(model_name, dates, y_true, y_pred, n_steps_out):
    """
    Detailed visualization for best model (Price Data)
    Updated with increased spacing to prevent overlaps.
    """
    print(f"\nGenerating visualization for: {model_name}...")
    
    min_len = min(len(dates), len(y_true), len(y_pred))
    plot_dates = dates[:min_len]
    y_true_1step = y_true[:min_len, 0] if y_true.ndim > 1 else y_true[:min_len]
    y_pred_1step = y_pred[:min_len, 0] if y_pred.ndim > 1 else y_pred[:min_len]
    
    # EDIT 1: Increased figure height from 10 to 12 for more breathing room
    fig = plt.figure(figsize=(16, 12))
    
    # EDIT 2: Increased hspace from 0.3 to 0.6 to prevent label overlap
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.3)
    
    # 1. Full Time Series - Price
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(plot_dates, y_true_1step, label='Actual Close', 
             color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(plot_dates, y_pred_1step, label='Predicted Close', 
             color='#007acc', linewidth=1.5, alpha=0.7)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    ax1.set_title(f'{model_name}: Full Price Prediction (Daily Data)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed View - First 100 samples
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_samples = min(100, len(plot_dates))
    
    ax2.plot(plot_dates[:zoom_samples], y_true_1step[:zoom_samples], 
             label='Actual', color='black', linewidth=2, marker='o', markersize=3)
    ax2.plot(plot_dates[:zoom_samples], y_pred_1step[:zoom_samples], 
             label='Predicted', color='#007acc', linewidth=2, marker='s', markersize=3)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.set_title(f'Zoom View - First {zoom_samples} Days', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Price', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Zoomed View - Last 100 samples
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.plot(plot_dates[-zoom_samples:], y_true_1step[-zoom_samples:], 
             label='Actual', color='black', linewidth=2, marker='o', markersize=3)
    ax3.plot(plot_dates[-zoom_samples:], y_pred_1step[-zoom_samples:], 
             label='Predicted', color='#007acc', linewidth=2, marker='s', markersize=3)
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax3.set_title(f'Zoom View - Last {zoom_samples} Days', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_ylabel('Price', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Scatter Plot
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(y_true_1step, y_pred_1step, alpha=0.5, s=20, color='purple')
    
    min_val = min(np.min(y_true_1step), np.min(y_pred_1step))
    max_val = max(np.max(y_true_1step), np.max(y_pred_1step))
    ax4.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction')
    
    ax4.set_title(f'{model_name}: Actual vs Predicted Price', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Actual Price', fontweight='bold')
    ax4.set_ylabel('Predicted Price', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    residuals = y_true_1step - y_pred_1step
    
    sns.histplot(residuals, kde=True, color='green', bins=50, ax=ax5, edgecolor='black', alpha=0.6)
    
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.axvline(x=np.mean(residuals), color='black', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    
    ax5.set_title('Price Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Error (Actual - Predicted)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    clean_name = model_name.replace(" ", "_").replace("/", "-")
    plt.savefig(f'price_prediction_{clean_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved price visualization to 'price_prediction_{clean_name}.png'")
    
    plt.show()


def visualize_log_returns(model_name, test_dates, y_true_log, y_pred_log, n_steps_out):
    """
    Visualize log return predictions with full time series and zoom view
    
    Args:
        model_name: Name of the model
        test_dates: Array of test dates
        y_true_log: True log returns (samples, n_steps_out)
        y_pred_log: Predicted log returns (samples, n_steps_out)
        n_steps_out: Number of forecast steps
    """
    # Take only the first step predictions for time series plot
    y_true_1step = y_true_log[:, 0]
    y_pred_1step = y_pred_log[:, 0]
    
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.3)
    # 1. Full Time Series - Log Returns
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(test_dates, y_true_1step, label='Actual Log Returns', 
             color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(test_dates, y_pred_1step, label='Predicted Log Returns', 
             color='red', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title(f'{model_name} - Full Log Return Predictions (1-step ahead)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Log Return', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed View - First 100 samples
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_samples = min(100, len(test_dates))
    ax2.plot(test_dates[:zoom_samples], y_true_1step[:zoom_samples], 
             label='Actual', color='black', linewidth=2, marker='o', markersize=3)
    ax2.plot(test_dates[:zoom_samples], y_pred_1step[:zoom_samples], 
             label='Predicted', color='red', linewidth=2, marker='s', markersize=3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title('Zoom View - First 100 Samples', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Log Return', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Zoomed View - Last 100 samples
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(test_dates[-zoom_samples:], y_true_1step[-zoom_samples:], 
             label='Actual', color='black', linewidth=2, marker='o', markersize=3)
    ax3.plot(test_dates[-zoom_samples:], y_pred_1step[-zoom_samples:], 
             label='Predicted', color='red', linewidth=2, marker='s', markersize=3)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_title('Zoom View - Last 100 Samples', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_ylabel('Log Return', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Scatter Plot - Actual vs Predicted
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(y_true_1step, y_pred_1step, alpha=0.5, s=20, color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_true_1step.min(), y_pred_1step.min())
    max_val = max(y_true_1step.max(), y_pred_1step.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction')
    
    ax4.set_title('Actual vs Predicted Log Returns', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Actual Log Return', fontweight='bold')
    ax4.set_ylabel('Predicted Log Return', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    residuals = y_true_1step - y_pred_1step
    ax5.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.axvline(x=residuals.mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {residuals.mean():.6f}')
    ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Prediction Error (Actual - Predicted)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f'log_returns_{model_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved log return visualization to 'log_returns_{model_name.replace(' ', '_')}.png'")
    plt.show()

def plot_training_losses(all_losses, epochs):
    """
    Plot training loss curves for all models and all rounds
    """
    n_models = len(all_losses)
    
    # Create subplots: one per model
    fig, axes = plt.subplots(
        (n_models + 1) // 2, 2, 
        figsize=(16, 5 * ((n_models + 1) // 2))
    )
    
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model_name, losses) in enumerate(all_losses.items()):
        ax = axes[idx]
        
        # Plot each round's loss curve
        for round_idx, loss_curve in enumerate(losses):
            ax.plot(
                range(1, len(loss_curve) + 1), 
                loss_curve, 
                alpha=0.3, 
                linewidth=1,
                color='steelblue'
            )
        
        # Calculate and plot mean loss across all rounds
        min_len = min(len(l) for l in losses)
        losses_array = np.array([l[:min_len] for l in losses])
        mean_loss = np.mean(losses_array, axis=0)
        std_loss = np.std(losses_array, axis=0)
        
        epochs_range = range(1, min_len + 1)
        ax.plot(epochs_range, mean_loss, 
                color='darkblue', linewidth=2.5, label='Mean Loss')
        ax.fill_between(epochs_range, 
                        mean_loss - std_loss, 
                        mean_loss + std_loss,
                        alpha=0.2, color='steelblue', label='±1 Std Dev')
        
        # Styling
        ax.set_title(f'{model_name} - Training Loss ({len(losses)} rounds)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Log scale if loss varies greatly
        if mean_loss[0] / mean_loss[-1] > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (MSE) - Log Scale')
    
    # Hide unused subplots
    for idx in range(len(all_losses), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('training_losses_all_models.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training loss plot to 'training_losses_all_models.png'")
    plt.show()

def plot_final_loss_comparison(all_losses):
    """
    Compare final loss values across all models
    """
    model_names = []
    final_losses_mean = []
    final_losses_std = []
    
    for model_name, losses in all_losses.items():
        model_names.append(model_name)
        # Get final epoch loss from each round
        final_losses = [loss_curve[-1] for loss_curve in losses]
        final_losses_mean.append(np.mean(final_losses))
        final_losses_std.append(np.std(final_losses))
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(model_names))
    
    plt.bar(x_pos, final_losses_mean, 
            yerr=final_losses_std, 
            capsize=5,
            color='steelblue', 
            alpha=0.7,
            edgecolor='black')
    
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Final Training Loss (MSE)', fontweight='bold')
    plt.title('Final Training Loss Comparison (Mean ± Std Dev)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig('final_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved final loss comparison to 'final_loss_comparison.png'")
    plt.show()
    
def visualize_comparison(df_results):
    '''
    Compare all models
    '''
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Avg Test RMSE', data=df_results, hue='Model', palette='viridis', legend=False)
    plt.title('Model Comparison: RMSE (Lower is Better)')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='Avg Test MAPE', data=df_results, hue='Model', palette='magma', legend=False)
    plt.title('Model Comparison: MAPE (Lower is Better)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


# %%

# ==========================================
# 5. Main Execution
# ==========================================

print("="*60)
print("Time Series Forecasting - Fixed Version (Daily Data)")
print("="*60)

# --- CONFIG ---
file_path = '/kaggle/input/crypto-2/processed_data/BTCUSDT.parquet'
save_dir = './saved_models'
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)

dim_type = 'Multi'  # Use multivariate
n_steps_in = 36      # Paper uses 6
n_steps_out = 5     # Paper uses 5
split_pct = 0.7     # Paper uses 70% train
use_percentage = 1.0
use_indicator = True

# --- LOAD & RESAMPLE DATA ---
print("\n1. Loading and resampling data to daily frequency...")
try:
    data, data_len, dates, original_close_prices = read_data(file_path, dim_type, use_percentage, use_indicator)
    print(f"   ✓ Data converted to log returns. Shape: {data.shape}")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    # return # Note: return only works inside a function, removed for script safety

# --- NORMALIZE ---
print("\n2. Normalizing data...")
data, scalers = data_transform(data)

# --- SPLIT TRAIN/TEST ---
train_len = int(np.floor(data_len * split_pct))
train_set = data[0:train_len]
test_set = data[train_len:]
print(f"   ✓ Train size: {train_len} hours")
print(f"   ✓ Test size: {data_len - train_len} hours")

# --- CREATE SEQUENCES ---
print("\n3. Creating sequences...")
n_features = train_set.shape[1] - 1  # Exclude Close from input
print(f"   ✓ Input features: {n_features} (excluding Close)")
print(f"   ✓ Input timesteps: {n_steps_in}")
print(f"   ✓ Output timesteps: {n_steps_out}")



# Calculate test dates for visualization
test_dates_start = train_len + n_steps_in

results = []
best_rmse = float('inf')
best_prediction = None
best_model_name = ""
best_y_true = None
best_test_dates = None
all_model_losses = {}



# %%
epochs = 20        # Default epochs for standard models
rounds = 5        # Paper uses 30 rounds
model_list = ['LSTM', 'BD LSTM', 'ED LSTM', 'CNN', 'Convolutional LSTM', 'MLP', 'Transformer'] # Add other models back here: 'LSTM', 'CNN', etc.
verbose = 1
# epochs = 1        # Default epochs for standard models
# rounds = 1        # Paper uses 30 rounds
# model_list = ['MLP'] # Add other models back here: 'LSTM', 'CNN', etc.
print("\n4. Training models...")
print("="*60)

# Dictionary to store the best prediction data for EACH model to visualize later
all_model_predictions = {} 

for model_name in model_list:
    current_epochs = epochs
    if 'LSTM' in model_name:
        current_epochs = 100
        print(f"\n⚡ DETECTED LSTM VARIANT: Increasing epochs to {current_epochs}")
    print(f"   Rounds: {rounds} | Epochs per round: {current_epochs}")
    
    round_rmses = []
    round_mapes = []
    round_losses = [] 
    
    # Track the best round for THIS specific model
    this_model_best_rmse = float('inf')
    this_model_saved_data = None
    
    for r in range(rounds):
        print(f"   Round {r+1}/{rounds}...", end=" ")
        
        K.clear_session()
        
        # Create sequences
        train_X, train_y = split_sequence(train_set, dim_type, n_steps_in, n_steps_out)
        test_X, test_y = split_sequence(test_set, dim_type, n_steps_in, n_steps_out)

        # Reshape
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
        
        # Create model
        model = create_model(model_name, n_features, n_steps_in, n_steps_out)
        if model is None: 
            continue
        
        # Train and predict
        train_pred, test_pred, loss_history = train_and_forecast(model, train_X, train_y, test_X, test_y, current_epochs, verbose)
        
        round_losses.append(loss_history)
        
        # Handle Output shape
        if train_pred.ndim == 3: train_pred = train_pred.squeeze(axis=-1)
        if test_pred.ndim == 3: test_pred = test_pred.squeeze(axis=-1)
        
        # Inverse transform
        train_pred_inv = np.zeros_like(train_pred)
        test_pred_inv = np.zeros_like(test_pred)
        train_y_inv = np.zeros_like(train_y)
        test_y_inv = np.zeros_like(test_y)
        
        scaler_close = scalers[0]
        
        # Inverse normalize to get LOG RETURNS
        for i in range(n_steps_out):
            train_pred_inv[:, i] = scaler_close.inverse_transform(train_pred[:, i].reshape(-1,1)).ravel()
            test_pred_inv[:, i] = scaler_close.inverse_transform(test_pred[:, i].reshape(-1,1)).ravel()
            train_y_inv[:, i] = scaler_close.inverse_transform(train_y[:, i].reshape(-1,1)).ravel()
            test_y_inv[:, i] = scaler_close.inverse_transform(test_y[:, i].reshape(-1,1)).ravel()
        
        # 1. Evaluate Log Returns
        log_metrics = eval_result(test_pred_inv, test_y_inv, n_steps_out, is_log_return=True)
        
        # 2. Convert to Prices
        train_start_idx = n_steps_in
        test_start_idx = train_len + n_steps_in
        
        train_pred_price = np.zeros_like(train_pred_inv)
        train_y_price = np.zeros_like(train_y_inv)
        test_pred_price = np.zeros_like(test_pred_inv)
        test_y_price = np.zeros_like(test_y_inv)

        for i in range(len(train_pred_inv)):
            initial_price = original_close_prices[train_start_idx + i - 1]
            train_pred_price[i, :] = inverse_log_returns(train_pred_inv[i, :], initial_price)
            train_y_price[i, :] = inverse_log_returns(train_y_inv[i, :], initial_price)
        
        for i in range(len(test_pred_inv)):
            initial_price = original_close_prices[test_start_idx + i - 1]
            test_pred_price[i, :] = inverse_log_returns(test_pred_inv[i, :], initial_price)
            test_y_price[i, :] = inverse_log_returns(test_y_inv[i, :], initial_price)
        
        # 3. Evaluate Prices
        price_metrics = eval_result(test_pred_price, test_y_price, n_steps_out, is_log_return=False)
        test_rmse = price_metrics['rmse_avg'] 
        
        round_rmses.append(test_rmse)
        round_mapes.append(price_metrics['mape_avg'])
        
        print(f"   [Log Returns] RMSE: {log_metrics['rmse_avg']:.5f}  | MAPE: {log_metrics['mape_avg']:.2f}%")
        print(f"   [Price]       RMSE: {price_metrics['rmse_avg']:.2f}  | MAPE: {price_metrics['mape_avg']:.2f}%")
        
        # --- STORAGE LOGIC ---
        
        # A. Save "Best Ever" (Global)
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model_name = model_name
            # Note: We rely on the "Visualize All" loop below for plotting now, 
            # but we keep these variables if you want to highlight the single best.
            best_prediction = test_pred_price
            best_y_true = test_y_price
            best_log_pred = test_pred_inv.copy()
            best_log_y_true = test_y_inv.copy()
            best_test_dates = dates[test_dates_start : test_dates_start + len(test_y_inv)]
            
        # B. Save "Best for Current Model" (Local)
        if test_rmse < this_model_best_rmse:
            this_model_best_rmse = test_rmse
            # Save deep copies of data for visualization
            this_model_saved_data = {
                'dates': dates[test_dates_start : test_dates_start + len(test_y_inv)],
                'y_true_price': test_y_price.copy(),
                'y_pred_price': test_pred_price.copy(),
                'y_true_log': test_y_inv.copy(),
                'y_pred_log': test_pred_inv.copy()
            }
        
        # Save model file
        m_path = f"{save_dir}/{model_name.replace(' ','_')}_r{r+1}.keras"
        s_path = f"{save_dir}/{model_name.replace(' ','_')}_r{r+1}_scaler.pkl"
        model.save(m_path)
        joblib.dump(scalers, s_path)
        
    # Add the best version of this model to the dictionary
    if this_model_saved_data:
        all_model_predictions[model_name] = this_model_saved_data

    all_model_losses[model_name] = round_losses 
    avg_rmse = np.mean(round_rmses)
    avg_mape = np.mean(round_mapes)
    std_rmse = np.std(round_rmses)
    std_mape = np.std(round_mapes)
    
    print(f"\n   ✓ {model_name} Results:")
    print(f"     RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}")
    print(f"     MAPE: {avg_mape:.2f}% ± {std_mape:.2f}%")
    
    results.append({
        'Model': model_name,
        'Avg Test RMSE': avg_rmse,
        'Avg Test MAPE': avg_mape
    })

# %%
# --- REPORTING ---
print("\n" + "="*60)
print("5. FINAL RESULTS")
print("="*60)

df_res = pd.DataFrame(results).sort_values('Avg Test RMSE')
print("\n" + df_res.to_string(index=False))
df_res.to_csv('final_model_comparison.csv', index=False)
print("\n✓ Results saved to 'final_model_comparison.csv'")

print("\n" + "="*60)
print(f"🏆 BEST MODEL OVERALL: {best_model_name}")
print(f"   RMSE: {best_rmse:.2f}")
print("="*60)

# 1. Bar Chart Comparison
print("\n6. Generating comparison bar charts...")
visualize_comparison(df_res)

# 2. Loss Curves
print("\n7. Generating Loss Curves...")
plot_training_losses(all_model_losses, epochs)
plot_final_loss_comparison(all_model_losses)

# 3. VISUALIZE ALL MODELS
print("\n" + "="*60)
print("8. Generating Detailed Visualizations for ALL Models")
print("="*60)

for m_name, data in all_model_predictions.items():
    print(f"\n>> Visualizing: {m_name}")
    
    # 1. Visualize Price
    visualize_best_model(
        model_name=m_name, 
        dates=data['dates'], 
        y_true=data['y_true_price'], 
        y_pred=data['y_pred_price'], 
        n_steps_out=n_steps_out
    )
    
    # 2. Visualize Log Returns
    visualize_log_returns(
        model_name=m_name, 
        test_dates=data['dates'], 
        y_true_log=data['y_true_log'], 
        y_pred_log=data['y_pred_log'], 
        n_steps_out=n_steps_out
    )

print("\n✓ Process Complete!")
print(f"✓ Models saved in {save_dir}")


