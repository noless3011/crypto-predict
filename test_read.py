import os

import pandas as pd

file_path = "0GUSDT.parquet"

if os.path.exists(file_path):
    df = pd.read_parquet(file_path)

    print(df.head())
    print("\nColumns:", df.columns.tolist())
    # print total number of rows
    print("Total rows:", len(df))
else:
    print("File for not found.")

debug_features_path = "debug_features/0GUSDT_features.csv"
if os.path.exists(debug_features_path):
    debug_df = pd.read_csv(debug_features_path)
    print("Debug features row count:", len(debug_df))
else:
    print("Debug features file not found.")
