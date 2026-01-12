
from data.data import ClickhouseHelper
import pandas as pd
from datetime import datetime

def check_news():
    print("Checking news table range...")
    query = "SELECT min(publishedOn), max(publishedOn), count(*) FROM news"
    # verbose=False to avoid seeing the query again
    df = ClickhouseHelper.run_to_df(query, verbose=False)
    print("News table summary:")
    print(df.to_string())
    
    query2 = "SELECT publishedOn FROM news ORDER BY publishedOn DESC LIMIT 5"
    df2 = ClickhouseHelper.run_to_df(query2, verbose=False)
    print("\nRecent 5 news timestamps:")
    print(df2.to_string())

if __name__ == "__main__":
    check_news()
