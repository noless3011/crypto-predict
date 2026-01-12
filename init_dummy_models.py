
import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import RobustScaler

# Configuration
SAVE_DIR = os.path.join('models')
N_STEPS_IN = 36
N_STEPS_OUT = 5
# Features calculated:
# Base: Open, High, Low (3)
# Indicators: RSI(1), MACD(2), MA(3), BB(4), ATR(1), ROC(1)
# Total Features = 3 + 12 = 15
N_FEATURES_INPUT = 15 
# Total Columns for Scaler = 1 (Close) + 15 (Features) = 16
N_TOTAL_COLS = 16 

MODEL_NAMES = [
    'LSTM', 
    'BD LSTM', 
    'ED LSTM', 
    'CNN', 
    'Convolutional LSTM', 
    'MLP', 
    'Transformer'
]


class IdentityScaler:
    def fit(self, x): return self
    def transform(self, x): return x
    def inverse_transform(self, x): return x

def create_dummy_model(name):
    """
    Dummy model that outputs current price ± random percentage
    in range [0.0001, 0.001]
    """

    def price_with_noise(x):
        # current price = last timestep, feature 0
        current_price = x[:, -1, 0:1]  # shape (batch, 1)

        # random percentage in [0.0001, 0.001]
        rand_pct = tf.random.uniform(
            shape=tf.shape(current_price),
            minval=0.0001,
            maxval=0.001
        )

        # randomly choose + or -
        sign = tf.random.uniform(tf.shape(current_price), -1.0, 1.0)
        sign = tf.sign(sign)

        noisy_price = current_price * (1.0 + sign * rand_pct)

        # repeat for N_STEPS_OUT
        return tf.tile(noisy_price, [1, N_STEPS_OUT])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(N_STEPS_IN, N_FEATURES_INPUT)),
        tf.keras.layers.Lambda(price_with_noise)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def create_dummy_scalers():
    scalers = {}

    for i in range(N_TOTAL_COLS):
        if i == 0:  # Close price
            scalers[i] = IdentityScaler()
        else:
            scaler = RobustScaler()
            scaler.fit(np.random.rand(100, 1))
            scalers[i] = scaler

    return scalers


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    print(f"Initializing dummy models in {SAVE_DIR}...")
    
    for name in MODEL_NAMES:
        clean_name = name.replace(" ", "_")
        
        # Paths
        model_path = os.path.join(SAVE_DIR, f"{clean_name}_r1.keras")
        scaler_path = os.path.join(SAVE_DIR, f"{clean_name}_r1_scaler.pkl")
        
        # 1. Create and Save Model
        if not os.path.exists(model_path):
            model = create_dummy_model(name)
            model.save(model_path)
            print(f"  [+] Created model: {model_path}")
        else:
            print(f"  [.] Model exists: {model_path}")
            
        # 2. Create and Save Scaler
        if not os.path.exists(scaler_path):
            scalers = create_dummy_scalers()
            joblib.dump(scalers, scaler_path)
            print(f"  [+] Created scalers: {scaler_path}")
        else:
            print(f"  [.] Scalers exist: {scaler_path}")

    print("\nDone! Dummy models are ready for testing.")

if __name__ == "__main__":
    main()
