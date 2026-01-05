import gc
import glob
import os
import random
import time
import warnings
from collections import deque

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ==============================================================================
#  1. DATA PROCESSING FUNCTION
# ==============================================================================
def process_ticker_batch(file_paths):
    dfs = []
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
            df_res["ret_lag_1"] = df_res["ret_1h"].shift(1)
            df_res["ret_lag_2"] = df_res["ret_1h"].shift(2)
            df_res["ret_lag_6"] = df_res["ret_1h"].shift(6)
            df_res["std_24h"] = df_res["ret_1h"].rolling(24).std()
            ma_24 = df_res["close"].rolling(24).mean()
            df_res["dist_ma_24"] = (df_res["close"] - ma_24) / ma_24

            # --- TARGETS (AGGRESSIVE FIX) ---
            future_ret = df_res["close"].pct_change(24).shift(-24)

            # LOWER THRESHOLD from 0.8 to 0.4
            # This forces more data into Class 1 and 2
            threshold = (df_res["std_24h"] * np.sqrt(24) * 0.4).fillna(0)

            conditions = [future_ret > threshold, future_ret < -threshold]
            choices = [1, 2]
            df_res["target"] = np.select(conditions, choices, default=0)

            df_res = df_res.dropna()
            dfs.append(df_res)

        except Exception:
            continue

    if not dfs:
        return None, None
    full_df = pd.concat(dfs)

    # --- UNDERSAMPLING FLAT CLASS (Noise Reduction) ---
    # We drop 50% of the "Flat" (0) rows randomly to force the model to look at trends
    mask_flat = full_df["target"] == 0
    mask_trend = full_df["target"] != 0

    # Keep all trends, but only 50% of flats
    df_flat = full_df[mask_flat].sample(frac=0.5, random_state=42)
    df_trend = full_df[mask_trend]

    full_df_balanced = pd.concat([df_flat, df_trend]).sort_index()

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
    feature_cols = [c for c in feature_cols if c in full_df_balanced.columns]

    X = full_df_balanced[feature_cols].astype(np.float32)
    y = full_df_balanced["target"].astype(np.int32)

    return X, y
    # ==============================================================================
    #  2. CONFIGURATION & GPU SETUP
    # ==============================================================================

    # GPU Config
    gpu_params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}

    # Use a robust "Random Forest" style parameter set to reduce variance
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,  # Standard speed
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,  # Randomly select features (adds robustness)
        "bagging_fraction": 0.7,  # Randomly select rows
        "bagging_freq": 1,
        "lambda_l1": 1.0,  # Conservative Regularization
        "lambda_l2": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
        "seed": 42,
    }
    params.update(gpu_params)

    # Get Files
    all_files = []
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            if filename.endswith(".parquet"):
                all_files.append(os.path.join(dirname, filename))

    # Shuffle to mix sectors
    random.shuffle(all_files)
    all_files = all_files[:200]  # Use more files to stabilize
    print(f"Total Files: {len(all_files)}")


# ==============================================================================
#  3. EXPANDED EXPERIMENT GRID (30 CONFIGURATIONS)
# ==============================================================================

param_grid = []

# --- GROUP 1: ULTRA CONSERVATIVE (The "Safety First" Approach) ---
# Goal: Prevent overfitting at all costs. High Regularization, Shallow Trees.
# Best for: High noise environments where most patterns are fake.
param_grid.extend(
    [
        {
            "name": "1_UltraSafe_Slow",
            "learning_rate": 0.005,
            "num_leaves": 15,
            "max_depth": 3,
            "lambda_l1": 5.0,
            "lambda_l2": 10.0,
            "bagging_fraction": 0.5,
            "feature_fraction": 0.5,
        },
        {
            "name": "2_UltraSafe_Med",
            "learning_rate": 0.01,
            "num_leaves": 20,
            "max_depth": 3,
            "lambda_l1": 3.0,
            "lambda_l2": 5.0,
            "bagging_fraction": 0.6,
            "feature_fraction": 0.6,
        },
        {
            "name": "3_Safe_DeepReg",
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": 4,
            "lambda_l1": 10.0,
            "lambda_l2": 20.0,
            "bagging_fraction": 0.7,
            "feature_fraction": 0.5,
        },
        {
            "name": "4_LowVar_Tiny",
            "learning_rate": 0.02,
            "num_leaves": 8,
            "max_depth": 2,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.5,
            "feature_fraction": 0.5,
        },
        {
            "name": "5_HighL1_Pruner",
            "learning_rate": 0.015,
            "num_leaves": 25,
            "max_depth": 4,
            "lambda_l1": 8.0,
            "lambda_l2": 2.0,
            "bagging_fraction": 0.6,
            "feature_fraction": 0.7,
        },
        {
            "name": "6_HighL2_Smoother",
            "learning_rate": 0.015,
            "num_leaves": 25,
            "max_depth": 4,
            "lambda_l1": 0.1,
            "lambda_l2": 25.0,
            "bagging_fraction": 0.6,
            "feature_fraction": 0.7,
        },
    ]
)

# --- GROUP 2: BALANCED / STANDARD (The "Sweet Spot") ---
# Goal: Typical LightGBM settings that work well on tabular data.
param_grid.extend(
    [
        {
            "name": "7_Balanced_Ref",
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
        },
        {
            "name": "8_Balanced_v2",
            "learning_rate": 0.03,
            "num_leaves": 40,
            "max_depth": 6,
            "lambda_l1": 0.5,
            "lambda_l2": 2.0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.7,
        },
        {
            "name": "9_Balanced_v3",
            "learning_rate": 0.025,
            "num_leaves": 35,
            "max_depth": 5,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.75,
            "feature_fraction": 0.75,
        },
        {
            "name": "10_Standard_Fast",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "lambda_l1": 0.2,
            "lambda_l2": 0.2,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
        },
        {
            "name": "11_Standard_Slow",
            "learning_rate": 0.01,
            "num_leaves": 40,
            "max_depth": 6,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
        },
        {
            "name": "12_Wide_Shallow",
            "learning_rate": 0.03,
            "num_leaves": 60,
            "max_depth": 4,
            "lambda_l1": 0.5,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.6,
        },
    ]
)

# --- GROUP 3: AGGRESSIVE / HIGH CAPACITY (Deep Learners) ---
# Goal: Capture complex non-linear interactions. High risk of overfitting.
# Best for: If you believe there are complex hidden fractals in the charts.
param_grid.extend(
    [
        {
            "name": "13_Deep_Learner",
            "learning_rate": 0.02,
            "num_leaves": 64,
            "max_depth": 8,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.9,
        },
        {
            "name": "14_Deep_Reg",
            "learning_rate": 0.01,
            "num_leaves": 100,
            "max_depth": 10,
            "lambda_l1": 5.0,
            "lambda_l2": 5.0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
        },
        {
            "name": "15_Aggro_Fast",
            "learning_rate": 0.08,
            "num_leaves": 50,
            "max_depth": 7,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.9,
        },
        {
            "name": "16_Aggro_Loose",
            "learning_rate": 0.05,
            "num_leaves": 128,
            "max_depth": -1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.1,
            "bagging_fraction": 1.0,
            "feature_fraction": 1.0,
        },
        {
            "name": "17_MidDeep_Slow",
            "learning_rate": 0.005,
            "num_leaves": 80,
            "max_depth": 9,
            "lambda_l1": 2.0,
            "lambda_l2": 2.0,
            "bagging_fraction": 0.7,
            "feature_fraction": 0.7,
        },
        {
            "name": "18_Complex_Fit",
            "learning_rate": 0.04,
            "num_leaves": 255,
            "max_depth": 12,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.9,
        },
    ]
)

# --- GROUP 4: FEATURE SUBSAMPLING FOCUS (Random Forest Style) ---
# Goal: Force the model to look at different indicators by hiding features.
# Best for: If specific indicators (like RSI) are dominating/overpowering others.
param_grid.extend(
    [
        {
            "name": "19_Feat_Starve",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.3,
        },
        {
            "name": "20_Feat_Half",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.5,
        },
        {
            "name": "21_Bagging_Heavy",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.4,
            "feature_fraction": 0.8,
        },
        {
            "name": "22_Double_Drop",
            "learning_rate": 0.02,
            "num_leaves": 40,
            "max_depth": 6,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.5,
            "feature_fraction": 0.5,
        },
        {
            "name": "23_Tiny_Feat",
            "learning_rate": 0.05,
            "num_leaves": 60,
            "max_depth": 8,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "bagging_fraction": 0.9,
            "feature_fraction": 0.2,
        },
        {
            "name": "24_Row_Subsample",
            "learning_rate": 0.04,
            "num_leaves": 40,
            "max_depth": 6,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "bagging_fraction": 0.3,
            "feature_fraction": 0.9,
        },
    ]
)
# ==============================================================================
#  4. TRAINING LOOP (FIXED: BATCHING + STABILITY CHECKS)
# ==============================================================================

best_overall_model = None
best_overall_loss = float("inf")
best_experiment_name = ""
final_history = {}

BATCH_SIZE = 10  # Process 10 files at a time to stabilize loss

for experiment in param_grid:
    print(f"\n⚡ STARTING EXPERIMENT: {experiment['name']}")
    print("-" * 75)
    print(
        f"{'Batch':<5} | {'Valid Loss':<10} | {'Status':<15} | {'Class Dist [Flat, Up, Down]'}"
    )

    # Merge base params with experiment params
    current_params = {**base_params, **experiment}

    model = None
    batches_processed = 0

    # Metrics tracking
    train_losses = []
    valid_losses = []
    rolling_valid_loss = deque(maxlen=3)  # Track last 3 batches

    start_time = time.time()
    experiment_failed = False

    # Shuffle files
    random.shuffle(all_files)

    # ITERATE BY BATCHES (Not single files)
    for i in range(0, len(all_files), BATCH_SIZE):
        # 1. Load a Cluster of Files
        batch_files = all_files[i : i + BATCH_SIZE]
        X, y = process_ticker_batch(batch_files)

        if X is None or len(X) < 100:
            continue

        # Check Class Balance (Debugging)
        counts = np.bincount(y, minlength=3)
        total_samples = len(y)
        if total_samples == 0:
            continue
        class_dist = f"[{counts[0] / total_samples:.2f}, {counts[1] / total_samples:.2f}, {counts[2] / total_samples:.2f}]"

        # Split (Standard 90/10)
        split_point = int(len(X) * 0.9)
        X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
        X_valid, y_valid = X.iloc[split_point:], y.iloc[split_point:]

        train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_ds = lgb.Dataset(
            X_valid, label=y_valid, reference=train_ds, free_raw_data=False
        )

        # 2. Incremental Learning
        # First batch gets 50 trees, updates get 10
        rounds = 50 if model is None else 10

        evals_result = {}

        try:
            model = lgb.train(
                current_params,
                train_set=train_ds,
                num_boost_round=rounds,
                valid_sets=[train_ds, valid_ds],
                valid_names=["train", "valid"],
                init_model=model,
                keep_training_booster=True,
                callbacks=[lgb.record_evaluation(evals_result)],
            )

            # Record metrics (Using last iteration)
            t_loss = evals_result["train"]["multi_logloss"][-1]
            v_loss = evals_result["valid"]["multi_logloss"][-1]

            train_losses.append(t_loss)
            valid_losses.append(v_loss)
            rolling_valid_loss.append(v_loss)
            avg_rolling = np.mean(rolling_valid_loss)

            batches_processed += 1

            # Print Status
            print(
                f"{batches_processed:<5} | {v_loss:.4f}     | {'OK' if v_loss < 1.05 else 'High'}            | {class_dist}"
            )

            # --- 3. EARLY STOPPING LOGIC ---

            # Rule A: Stability Check (After 4 batches = ~40 files)
            # If loss is still > 1.07 (Random guessing), kill it.
            if batches_processed >= 4:
                if avg_rolling > 1.07:
                    print(f"  ❌ STOP: No signal. Avg Loss {avg_rolling:.4f} > 1.07")
                    experiment_failed = True
                    break

            # Rule B: Signal Found?
            if v_loss < 1.00:
                print("  >>> SIGNAL DETECTED (< 1.00) <<<")

        except Exception as e:
            print(f"Error: {e}")
            continue

        # GC
        del X, y, X_train, y_train, train_ds, valid_ds
        gc.collect()

    # --- END EXPERIMENT ---
    elapsed = time.time() - start_time
    # Use average of last 3 batches for final score
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
        best_experiment_name = experiment["name"]
        final_history = {"train": train_losses, "valid": valid_losses}

    del model
    gc.collect()

# ==============================================================================
#  5. SAVE BEST
# ==============================================================================
if best_overall_model:
    print(
        f"\nSaving Best Model: {best_experiment_name} (Loss: {best_overall_loss:.4f})"
    )
    best_overall_model.save_model("best_crypto_model_batch.txt")

# ==============================================================================
#  5. RESULTS
# ==============================================================================

if best_overall_model is None:
    print("\n❌ All experiments failed.")
else:
    print("\n" + "=" * 60)
    print(f"WINNER: {best_experiment_name}")
    print(f"Final Validation Loss: {best_overall_loss:.4f}")
    print("=" * 60)

    best_overall_model.save_model("best_crypto_model_gpu.txt")

    plt.figure(figsize=(12, 6))
    plt.plot(final_history["train"], label="Train Loss", alpha=0.6)
    plt.plot(final_history["valid"], label="Valid Loss", linewidth=2, color="red")
    plt.axhline(y=1.05, color="orange", linestyle="--", label="Stability (1.05)")
    plt.axhline(y=1.00, color="green", linestyle="--", label="Target (1.00)")
    plt.title(f"Training History: {best_experiment_name}")
    plt.xlabel("Files")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("best_model_performance.png")
    plt.show()


import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

# ==============================================================================
#  6. TEST ON UNSEEN DATA (HOLDOUT SET)
# ==============================================================================

print("\n" + "=" * 60)
print("🧪 TESTING BEST MODEL ON UNSEEN DATA")
print("=" * 60)

# 1. Define Holdout Set
# We take the last 20% of files that were likely NOT used in the training loop
# (Assuming you shuffled and used a subset, or we just split the list now)
num_test_files = int(len(all_files) * 0.2)
if num_test_files < 1:
    num_test_files = 1

test_files = all_files[-num_test_files:]
print(f"Testing on {len(test_files)} unseen files...")

# 2. Prepare Data (Reuse the batch processor to ensure identical features)
# We process them all into one big dataframe for evaluation
X_test, y_test = process_ticker_batch(test_files)

if X_test is None or len(X_test) == 0:
    print("❌ Error: Not enough data in test files to evaluate.")
else:
    # 3. Predict
    # LightGBM returns probabilities: [[prob_0, prob_1, prob_2], ...]
    y_pred_prob = best_overall_model.predict(X_test)
    y_pred_class = np.argmax(y_pred_prob, axis=1)  # Convert probs to 0, 1, 2

    # 4. Calculate Metrics
    final_log_loss = log_loss(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred_class)

    print(f"\n📊 RESULTS:")
    print(f"  Final Log Loss: {final_log_loss:.4f} (Lower is better)")
    print(f"  Overall Accuracy: {acc:.2%}")

    # 5. Detailed Classification Report
    # Precision = When model predicts UP, how often is it actually UP? (Crucial for trading)
    target_names = ["Flat (0)", "Up (1)", "Down (2)"]
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred_class, target_names=target_names))

    # ==============================================================================
    #  7. VISUALIZATION
    # ==============================================================================

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # A. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    # Normalize by row (True Class) to see recall rates
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=axes[0],
        xticklabels=target_names,
        yticklabels=target_names,
    )
    axes[0].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # B. Feature Importance
    # See what the model actually learned
    importance = best_overall_model.feature_importance(importance_type="gain")
    feature_names = best_overall_model.feature_name()

    # Create DataFrame for plotting
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(10)

    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=axes[1])
    axes[1].set_title("Top 10 Features (Gain)", fontweight="bold")
    axes[1].set_xlabel("Gain (Contribution to Loss Reduction)")

    plt.tight_layout()
    plt.savefig("final_evaluation.png")
    plt.show()

    print("\n✅ Evaluation Complete. Chart saved to 'final_evaluation.png'")

    # --- TRADING REALITY CHECK ---
    # Quick heuristic to see if the model is tradeable
    up_precision = 0
    down_precision = 0

    # Extract precision from report manually or calculation
    # (Precision for class 1 and 2)
    # Class 1 (Up)
    tp_up = cm[1, 1]
    fp_up = cm[0, 1] + cm[2, 1]
    if (tp_up + fp_up) > 0:
        up_precision = tp_up / (tp_up + fp_up)

    # Class 2 (Down)
    tp_down = cm[2, 2]
    fp_down = cm[0, 2] + cm[1, 2]
    if (tp_down + fp_down) > 0:
        down_precision = tp_down / (tp_down + fp_down)

    print("\n💰 TRADING REALITY CHECK:")
    print(f"  'UP' Signal Precision:   {up_precision:.2%} (Target > 55%)")
    print(f"  'DOWN' Signal Precision: {down_precision:.2%} (Target > 55%)")

    if up_precision > 0.55 or down_precision > 0.55:
        print("  🚀 POTENTIAL: Model has an edge in at least one direction.")
    else:
        print(
            "  ⚠️ CAUTION: Precision is near random (33-50%). Needs more feature engineering."
        )
