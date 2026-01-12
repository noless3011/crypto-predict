import pandas as pd
import os

DATA_FILE = "BTCUSDT.csv"

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    
    # Check for 'openTime' or 'Date'
    if 'openTime' in df.columns:
        date_col = 'openTime'
    elif 'Date' in df.columns:
        date_col = 'Date'
    else:
        print("Could not find date column (openTime or Date)")
        exit()
        
    # Convert if numeric
    if pd.api.types.is_numeric_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], unit='ms')
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        
    latest_date = df[date_col].max()
    print(f"Latest Date in {DATA_FILE}: {latest_date}")
else:
    print(f"{DATA_FILE} not found.")
