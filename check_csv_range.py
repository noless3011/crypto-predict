
import pandas as pd

def check_csv():
    try:
        df = pd.read_csv("BTCUSDT.csv")
        # Assuming openTime is in ms or date string needed parsing
        if 'openTime' in df.columns:
             # Check type
             if pd.api.types.is_numeric_dtype(df['openTime']):
                 start = pd.to_datetime(df['openTime'].min(), unit='ms')
                 end = pd.to_datetime(df['openTime'].max(), unit='ms')
             else:
                 start = pd.to_datetime(df['openTime'].min())
                 end = pd.to_datetime(df['openTime'].max())
             print(f"CSV Range: {start} to {end}")
             print(f"Rows: {len(df)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_csv()
