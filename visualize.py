import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
CSV_FILE = "feature_output.csv"  # Matches the output from the fixed generator
ROWS_TO_PLOT = 1000  # Plot last N rows


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)

        # 1. Handle Missing Timestamps (The LSTM data drops dates)
        if "openTime" in df.columns:
            df["datetime"] = pd.to_datetime(df["openTime"], unit="ms")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        else:
            # Create a dummy index for plotting if date is gone
            print("Notice: No timestamp found (expected for LSTM data). Using index.")
            df["datetime"] = df.index

        # 2. Filter Ticker if it exists (it likely doesn't in LSTM data)
        if "ticker" in df.columns:
            # Auto-select first ticker if not specified
            unique_tickers = df["ticker"].unique()
            df = df[df["ticker"] == unique_tickers[0]].copy()

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def plot_technical_dashboard(df):
    """
    Adaptive Dashboard:
    - If Raw Prices exist: Plots Candles.
    - If only Features exist: Plots Cumulative Return (Price Proxy).
    """
    df_plot = df.tail(ROWS_TO_PLOT).reset_index(drop=True)

    # Determine what to plot for "Price"
    has_ohlc = all(col in df_plot.columns for col in ["open", "high", "low", "close"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            "Price Action (or Proxy)",
            "Volume / Volatility",
            "Momentum / Oscillators",
        ),
    )

    # --- ROW 1: Price or Proxy ---
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df_plot["datetime"],
                open=df_plot["open"],
                high=df_plot["high"],
                low=df_plot["low"],
                close=df_plot["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    else:
        # Reconstruct a price proxy from log returns
        if "log_return" in df_plot.columns:
            # Start at 100 and apply returns
            df_plot["price_proxy"] = 100 * np.exp(df_plot["log_return"].cumsum())
            fig.add_trace(
                go.Scatter(
                    x=df_plot["datetime"],
                    y=df_plot["price_proxy"],
                    mode="lines",
                    name="Cum. Return (Price Proxy)",
                    line=dict(color="#00ffcc"),
                ),
                row=1,
                col=1,
            )

    # --- ROW 2: Volume or Volatility ---
    # Check for Volume (Raw or Log)
    if "volume" in df_plot.columns:
        fig.add_trace(
            go.Bar(x=df_plot["datetime"], y=df_plot["volume"], name="Volume"),
            row=2,
            col=1,
        )
    elif "volume_log" in df_plot.columns:
        fig.add_trace(
            go.Bar(
                x=df_plot["datetime"],
                y=df_plot["volume_log"],
                name="Log Volume",
                marker_color="teal",
            ),
            row=2,
            col=1,
        )
    # Fallback: Plot Volatility if no volume
    elif "rolling_volatility_36" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"],
                y=df_plot["rolling_volatility_36"],
                name="Volatility",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

    # --- ROW 3: Indicators ---
    # RSI (Handle 0-100 or 0-1 scale)
    if "rsi_14" in df_plot.columns:
        rsi_max = df_plot["rsi_14"].max()
        is_normalized = rsi_max <= 1.05

        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"],
                y=df_plot["rsi_14"],
                name="RSI",
                line=dict(color="purple"),
            ),
            row=3,
            col=1,
        )

        # Add Lines
        upper = 0.7 if is_normalized else 70
        lower = 0.3 if is_normalized else 30
        fig.add_hline(y=upper, line_dash="dot", row=3, col=1, line_color="gray")
        fig.add_hline(y=lower, line_dash="dot", row=3, col=1, line_color="gray")

    fig.update_layout(
        title="Feature & Data Dashboard",
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
    )
    fig.show()


def plot_correlation_heatmap(df):
    """Plots correlation, excluding datetime/index columns."""
    exclude = ["openTime", "datetime", "ticker", "ignore"]
    # Only pick numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    cols = [c for c in numeric_df.columns if c not in exclude]

    if len(cols) > 50:
        print("Note: Too many columns for legible heatmap. Selecting top 40 + Targets.")
        target_cols = [c for c in cols if "TARGET" in c]
        feature_cols = [c for c in cols if "TARGET" not in c][:40]
        cols = feature_cols + target_cols

    corr_matrix = df[cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=False,  # Turn off text for large matrices
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(template="plotly_dark", height=800)
    fig.show()


def plot_target_analysis(df):
    """Checks the distribution of the Target variable."""
    if "TARGET_class" in df.columns:
        counts = df["TARGET_class"].value_counts()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            title="Class Balance (Target Class)",
            labels={"x": "Class (0=Down, 1=Up)", "y": "Count"},
            color=counts.index.astype(str),
        )
        fig.update_layout(template="plotly_dark", height=400, showlegend=False)
        fig.show()

    if "TARGET_next_return" in df.columns:
        fig = px.histogram(
            df,
            x="TARGET_next_return",
            nbins=100,
            title="Regression Target Distribution",
            marginal="box",
        )
        fig.update_layout(template="plotly_dark", height=400)
        fig.show()


def plot_all_distributions(df):
    """Plots distributions with Log Y-Axis to see outliers."""
    exclude = ["openTime", "datetime", "ticker", "TARGET_class"]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]

    # Limit to first 32 features to avoid browser crash if huge
    if len(numeric_cols) > 32:
        print(f"Plotting first 32 features out of {len(numeric_cols)}...")
        numeric_cols = numeric_cols[:32]

    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)

    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1

        fig.add_trace(
            go.Histogram(x=df[col], name=col, marker_color="steelblue"),
            row=row,
            col=col_pos,
        )

    fig.update_layout(
        title="Feature Distributions (Log Scale Y)",
        template="plotly_dark",
        height=250 * n_rows,
        showlegend=False,
    )
    fig.update_yaxes(type="log")  # Log scale helps see outliers
    fig.show()


if __name__ == "__main__":
    df = load_data(CSV_FILE)

    if not df.empty:
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

        # 1. Check Data Integrity
        print("\n--- Checking Missing Values ---")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        # 2. Visualizations
        plot_technical_dashboard(df)  # Visual check of time-series
        plot_target_analysis(df)  # Check if classes are balanced
        plot_correlation_heatmap(df)  # Check for multicollinearity
        plot_all_distributions(df)  # Check for stationarity/outliers
    else:
        print("DataFrame is empty. Please check the CSV file path.")
