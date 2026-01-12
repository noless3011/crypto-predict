from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import market, indicators, news, forecast

app = FastAPI(
    title="Crypto Prediction API",
    description="API for accessing market data, technical indicators, news, and AI-powered price forecasts.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
origins = [
    "*", # Allow all for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers with prefix /api/v1
APP_PREFIX = "/api/v1"

app.include_router(market.router, prefix=APP_PREFIX)
app.include_router(indicators.router, prefix=APP_PREFIX)
app.include_router(news.router, prefix=APP_PREFIX)
app.include_router(forecast.router, prefix=APP_PREFIX)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Crypto Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
