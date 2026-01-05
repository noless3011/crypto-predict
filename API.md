---

## 1. General Configuration
*   **Base URL:** `/api/v1`
*   **Date Format:** ISO 8601 (`YYYY-MM-DDTHH:mm:ssZ`) hoặc Unix Timestamp (tùy team thống nhất, ở dưới ví dụ dùng Unix Timestamp cho gọn biểu đồ).

---

## 2. API Chi tiết

### A. Ticker Selection (Màn hình Home & Search)
Dùng để tìm kiếm mã chứng khoán/coin và lấy thông tin cơ bản.

**1. Search/List Tickers**
*   **Endpoint:** `GET /market/tickers`
*   **Query Params:**
    *   `keyword`: string (Optional - dùng để search, ví dụ "BTC" hoặc "Vinamilk")
*   **Response:**
    ```json
    [
      {
        "symbol": "BTCUSDT",
        "name": "Bitcoin",
        "type": "CRYPTO", // hoặc STOCK
        "exchange": "BINANCE"
      },
      {
        "symbol": "VNM",
        "name": "Vinamilk",
        "type": "STOCK",
        "exchange": "HOSE"
      }
    ]
    ```

**2. Ticker Summary (Header thông tin)**
*   **Endpoint:** `GET /market/summary/{symbol}`
*   **Response:**
    ```json
    {
      "symbol": "BTCUSDT",
      "current_price": 45000.50,
      "change_24h": 2.5, // % thay đổi
      "volume_24h": 1000000,
      "updated_at": 1704466800000
    }
    ```

---

### B. Price Graph Tab (Biểu đồ giá + Volume + RSI)
Tab này cần dữ liệu nến (OHLC) để vẽ biểu đồ chính và dữ liệu RSI.

**3. Historical Data (Candlestick & Volume)**
*   **Endpoint:** `GET /market/history/{symbol}`
*   **Query Params:**
    *   `interval`: string (e.g., `1h`, `4h`, `1d`, `1w`)
    *   `start_time`: long (Unix timestamp)
    *   `end_time`: long (Unix timestamp)
*   **Response:**
    ```json
    {
      "symbol": "BTCUSDT",
      "interval": "1d",
      "data": [
        // [Time, Open, High, Low, Close, Volume]
        [1704380400000, 42000, 42500, 41900, 42300, 500],
        [1704466800000, 42300, 43000, 42100, 42800, 600]
      ]
    }
    ```

**4. RSI Indicator (Cho biểu đồ phụ bên dưới)**
*   **Endpoint:** `GET /indicators/rsi/{symbol}`
*   **Query Params:**
    *   `interval`: string (`1h`, `1d`...) - *Phải khớp với biểu đồ giá*
    *   `period`: int (Default `14`)
    *   `start_time`: long
    *   `end_time`: long
*   **Response:**
    ```json
    {
      "indicator": "RSI",
      "data": [
        { "time": 1704380400000, "value": 45.5 },
        { "time": 1704466800000, "value": 50.2 }
      ]
    }
    ```

---

### C. News Tab (Tin tức)
Lấy danh sách tin tức liên quan đến mã đó trong khoảng thời gian chọn.

**5. Get News**
*   **Endpoint:** `GET /news/{symbol}`
*   **Query Params:**
    *   `start_time`: long
    *   `end_time`: long
    *   `limit`: int (Default 20)
*   **Response:**
    ```json
    [
      {
        "id": "news_123",
        "title": "Bitcoin vượt mốc 45k",
        "summary": "Do sự kiện ETF được thông qua...",
        "source": "CoinDesk",
        "url": "https://coindesk.com/...",
        "published_at": 1704466800000,
        "sentiment": "POSITIVE" // (Optional: Nếu Backend có AI phân tích cảm xúc tin)
      },
      {
        "id": "news_124",
        "title": "Thị trường biến động mạnh",
        "summary": "...",
        "source": "VnExpress",
        "url": "...",
        "published_at": 1704380400000,
        "sentiment": "NEUTRAL"
      }
    ]
    ```

---

### D. Tech Indicator Tab (Các chỉ báo kỹ thuật khác)
Phần này Cường backend implement thư viện TA-Lib (Python) hoặc tương đương để tính toán. Dưới đây là các chỉ báo phổ biến nên có:

**6. Technical Indicators Collection**
Có thể gộp chung vào 1 API hoặc tách lẻ. Đề xuất tách lẻ hoặc dùng query param `type` để linh hoạt.

*   **Endpoint:** `GET /indicators/{type}/{symbol}`
*   **Path Params:**
    *   `type`: `macd`, `bollinger`, `ema`, `sma`
*   **Query Params:**
    *   `interval`: string (`1d`, `4h`...)
    *   `start_time`: long
    *   `end_time`: long
*   **Response Examples:**

    *   **MACD (Moving Average Convergence Divergence):**
        ```json
        {
          "indicator": "MACD",
          "data": [
            {
              "time": 1704466800000,
              "macd": 120.5,
              "signal": 115.0,
              "histogram": 5.5
            }
          ]
        }
        ```

    *   **Bollinger Bands (Dải Bollinger):**
        ```json
        {
          "indicator": "BOLLINGER",
          "data": [
            {
              "time": 1704466800000,
              "upper": 44000,
              "middle": 42000,
              "lower": 40000
            }
          ]
        }
        ```

    *   **EMA (Exponential Moving Average - Đường trung bình lũy thừa):**
        *   Param thêm: `period` (ví dụ 34, 89)
        ```json
        {
           "indicator": "EMA",
           "period": 34,
           "data": [ { "time": 1704466800000, "value": 42500 } ]
        }
        ```

---

### E. Prediction (Dự báo - Tính năng AI)
Đây là tính năng "ăn tiền" của app.

**7. Price Forecast**
*   **Endpoint:** `GET /forecast/{symbol}`
*   **Query Params:**
    *   `days`: int (Số ngày muốn dự báo, ví dụ 7 ngày tới)
*   **Response:**
```json
{
  "model_name": "LSTM", // Có thể là LSTM, TRANSFORMER hoặc LIGHTGBM tùy chọn
  "forecast_data": [
    {
      "time": 1704553200000, // Tương lai
      "direction": "UP", // UP, DOWN, or FLAT
      "confidence": 0.75 // Độ tin cậy (0-1)
    },
    {
      "time": 1704639600000,
      "direction": "UP",
      "confidence": 0.68
    }
  ]
}

```

---
