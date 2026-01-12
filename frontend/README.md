# Crypto Prediction Frontend

A modern, premium React application for analyzing and predicting cryptocurrency prices.

## Technologies
- **Vite + React**: Fast development and optimized build.
- **Lightweight Charts**: High-performance financial charting (Binance-like).
- **Lucide React**: Beautiful icons.
- **Axios**: API integration.

## Setup

1.  **Install Dependencies** (if not already done):
    ```bash
    cd frontend
    npm install
    ```

2.  **Run Development Server**:
    ```bash
    npm run dev
    ```

3.  **Open in Browser**:
    Usually http://localhost:5173

## Features

- **Ticker Selection**: Search and select from available tickers.
- **Dashboard**:
    - **Price Graph**: Candlestick chart with Volume and RSI indicators.
    - **News**: Real-time news feed with sentiment analysis.
    - **Tech Indicators**: MACD and Bollinger Bands charts.
    - **AI Forecast**: 24-hour price direction prediction using LSTM.

## Configuration

API URL is properly set in `src/services/api.js`. Default: `http://localhost:8000/api/v1`.
Ensure the backend API is running on port 8000.
