import pandas as pd
from data.data import ClickhouseHelper
import os

def download_data():
    print("Downloading BTCUSDT data from Clickhouse...")
    try:
        # Fetch all data for BTCUSDT
        # Using None for time_start and time_end to get everything
        df = ClickhouseHelper.get_data_between(
            ticker="BTCUSDT",
            time_start=None,
            time_end=None,
            verbose=True
        )
        
        if df.empty:
            print("No data found for BTCUSDT")
            return

        # Ensure directory exists (though we are saving to current dir)
        output_file = "BTCUSDT.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} rows to {output_file}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
