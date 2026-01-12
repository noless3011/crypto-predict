import os

class Settings:
    CLICKHOUSE_HOST: str = "128.199.197.160"
    CLICKHOUSE_PORT: int = 32014
    CLICKHOUSE_USER: str = "readonly_user"
    CLICKHOUSE_PASSWORD: str = "grA5SfKxOQ"
    CLICKHOUSE_DB: str = "default"
    
    # Path to ticker.csv, assuming it's in the project root or data folder
    # Adjust this path based on where the API is run from
    TICKER_CSV_PATH: str = r"d:\ProgrammingProjects\data_science\BTL\data\ticker.csv"

settings = Settings()
