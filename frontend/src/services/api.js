import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const marketApi = {
  getTickers: (keyword) => api.get('/market/tickers', { params: { keyword } }),
  getSummary: (symbol) => api.get(`/market/summary/${symbol}`),
  getHistory: (symbol, interval, start_time, end_time) => 
    api.get(`/market/history/${symbol}`, { params: { interval, start_time, end_time } }),
};

export const indicatorsApi = {
  getRsi: (symbol, interval, period, start_time, end_time) => 
    api.get(`/indicators/rsi/${symbol}`, { params: { interval, period, start_time, end_time } }),
  getGeneric: (type, symbol, interval, start_time, end_time) =>
    api.get(`/indicators/${type}/${symbol}`, { params: { interval, start_time, end_time } }),
};

export const newsApi = {
  getNews: (symbol, start_time, end_time) =>
    api.get(`/news/${symbol}`, { params: { start_time, end_time } }),
};

export const forecastApi = {
  getForecast: (symbol, days) => api.get(`/forecast/${symbol}`, { params: { days } }),
};

export const evaluateApi = {
  evaluateModel: (ticker, model_type, start_date, end_date) => 
    api.post('/evaluate', { ticker, model_type, start_date, end_date }),
};
