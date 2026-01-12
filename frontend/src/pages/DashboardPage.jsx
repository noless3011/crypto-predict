import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown, Activity, Newspaper, BarChart2, Zap } from 'lucide-react';
import { marketApi, indicatorsApi, newsApi, forecastApi } from '../services/api';
import PriceChart from '../components/PriceChart';
import TechChart from '../components/TechChart';

const COLORS = {
    up: '#10b981',
    down: '#ef4444',
    accent: '#6366f1',
    textSecondary: '#94a3b8'
};

const DashboardPage = () => {
    const { symbol } = useParams();
    const navigate = useNavigate();

    const [summary, setSummary] = useState(null);
    const [interval, setInterval] = useState('1h');
    const [activeTab, setActiveTab] = useState('price'); // price, news, tech

    const [historyData, setHistoryData] = useState([]);
    const [rsiData, setRsiData] = useState([]);
    const [newsData, setNewsData] = useState([]);
    const [forecasts, setForecasts] = useState([]);

    const [macdData, setMacdData] = useState([]);
    const [bbData, setBbData] = useState([]);

    const [loading, setLoading] = useState(true);

    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

    // helper to convert date input string (YYYY-MM-DD) to unix timestamp (seconds)
    const toTimestamp = (dateStr, isEnd = false) => {
        if (!dateStr) return null;
        const d = new Date(dateStr);
        if (isEnd) d.setHours(23, 59, 59);
        else d.setHours(0, 0, 0);
        return Math.floor(d.getTime() / 1000);
    };

    // Initial Data Load
    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            const startTs = toTimestamp(startDate);
            const endTs = toTimestamp(endDate, true);

            try {
                // Core data
                // Need to reload ALL data when date range changes
                const p1 = marketApi.getSummary(symbol); // Summary is always latest, ignoring range usually? Or should it valid for range? Usually summary is "now".
                // But history depends on range
                const p2 = marketApi.getHistory(symbol, interval, startTs, endTs);
                const p3 = indicatorsApi.getRsi(symbol, interval, 14, startTs, endTs);
                const p4 = newsApi.getNews(symbol, startTs, endTs);
                const p5 = forecastApi.getForecast(symbol, 7); // Forecast is future, keeps as is

                const [sumRes, histRes, rsiRes, newsRes, fcRes] = await Promise.all([p1, p2, p3, p4, p5]);

                setSummary(sumRes.data);
                setHistoryData(histRes.data.data);
                setRsiData(rsiRes.data.data);
                setNewsData(newsRes.data);
                setForecasts(fcRes.data);

                // Secondary data for tech tab
                const [macdRes, bbRes] = await Promise.all([
                    indicatorsApi.getGeneric('macd', symbol, interval, startTs, endTs),
                    indicatorsApi.getGeneric('bollinger', symbol, interval, startTs, endTs)
                ]);
                setMacdData(macdRes.data.data);
                setBbData(bbRes.data.data);

            } catch (e) {
                console.error("Error loading dashboard data", e);
            } finally {
                setLoading(false);
            }
        };

        if (symbol) loadData();
    }, [symbol, interval, startDate, endDate]); // Added startDate, endDate dependency

    const handleBack = () => navigate('/');

    const formatPrice = (price) => {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(price);
    };

    const renderPrediction = () => {
        if (!forecasts || forecasts.length === 0) return null;

        return (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                {forecasts.map((f, idx) => {
                    const dataPoint = f.forecast_data[0];
                    if (!dataPoint) return null;

                    const isUp = dataPoint.direction === 'UP';
                    const color = isUp ? COLORS.up : (dataPoint.direction === 'DOWN' ? COLORS.down : COLORS.textSecondary);

                    return (
                        <div key={idx} className="card animate-fade-in" style={{
                            background: `linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9))`,
                            border: `1px solid ${color}40`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                <div style={{ padding: '0.75rem', borderRadius: '50%', background: `${color}20` }}>
                                    <Zap size={24} color={color} />
                                </div>
                                <div>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{f.model_name}</div>
                                    <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: color }}>
                                        {dataPoint.direction} <span style={{ fontSize: '0.9rem', opacity: 0.8 }}>({(dataPoint.confidence * 100).toFixed(0)}%)</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        );
    };

    return (
        <div className="container animate-fade-in">
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
                <button onClick={handleBack} className="btn btn-ghost">
                    <ArrowLeft size={20} />
                </button>
                {summary && (
                    <div>
                        <h1 style={{ margin: 0, fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            {summary.symbol}
                            <span className={`badge ${summary.change_24h >= 0 ? 'text-up' : 'text-down'}`}
                                style={{
                                    fontSize: '1rem',
                                    color: summary.change_24h >= 0 ? COLORS.up : COLORS.down,
                                    background: summary.change_24h >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                                    padding: '0.2rem 0.5rem',
                                    borderRadius: '4px'
                                }}>
                                {summary.change_24h > 0 ? '+' : ''}{summary.change_24h}%
                            </span>
                        </h1>
                        <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{formatPrice(summary.current_price)}</div>
                    </div>
                )}
            </div>

            {/* Prediction Card */}
            {renderPrediction()}

            {/* Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', gap: '0.5rem', background: 'var(--bg-card)', padding: '0.25rem', borderRadius: 'var(--radius-md)' }}>
                    {['15m', '1h', '4h', '1d'].map(i => (
                        <button
                            key={i}
                            className={`btn ${interval === i ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setInterval(i)}
                            style={{ fontSize: '0.8rem', padding: '0.25rem 0.75rem' }}
                        >
                            {i.toUpperCase()}
                        </button>
                    ))}
                </div>

                {/* Date Range Picker */}
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', background: 'var(--bg-card)', padding: '0.5rem', borderRadius: 'var(--radius-md)' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Range:</span>
                    <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        style={{
                            background: 'var(--bg-default)',
                            border: '1px solid var(--bg-hover)',
                            color: 'var(--text-primary)',
                            padding: '0.25rem',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                        }}
                    />
                    <span style={{ color: 'var(--text-secondary)' }}>-</span>
                    <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        style={{
                            background: 'var(--bg-default)',
                            border: '1px solid var(--bg-hover)',
                            color: 'var(--text-primary)',
                            padding: '0.25rem',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                        }}
                    />
                    {(startDate || endDate) && (
                        <button
                            onClick={() => { setStartDate(''); setEndDate(''); }}
                            style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.8rem' }}
                        >
                            Clear
                        </button>
                    )}
                </div>
            </div>

            {/* Tabs */}
            <div style={{ borderBottom: '1px solid var(--bg-hover)', marginBottom: '1.5rem', display: 'flex', gap: '1rem' }}>
                {[
                    { id: 'news', label: 'News', icon: Newspaper },
                    { id: 'price', label: 'Price Graph', icon: Activity },
                    { id: 'tech', label: 'Tech Indicator', icon: BarChart2 }
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            borderBottom: activeTab === tab.id ? `2px solid var(--color-accent)` : '2px solid transparent',
                            color: activeTab === tab.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                            padding: '0.75rem 0',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            fontSize: '1rem',
                            transition: 'all 0.2s'
                        }}
                    >
                        <tab.icon size={18} /> {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            <div className="card" style={{ minHeight: '400px' }}>
                {loading ? (
                    <div className="flex-center" style={{ height: '300px' }}>Loading data...</div>
                ) : (
                    <>
                        {activeTab === 'price' && (
                            <PriceChart data={historyData} rsiData={rsiData} colors={COLORS} />
                        )}

                        {activeTab === 'news' && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                {newsData.length === 0 ? <p>No news found within this range.</p> : newsData.map(n => (
                                    <div key={n.id} style={{ borderBottom: '1px solid var(--bg-hover)', paddingBottom: '1rem' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem' }}>{n.title}</h3>
                                            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{new Date(n.published_at * 1000).toLocaleDateString()}</span>
                                        </div>
                                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', margin: 0 }}>{n.summary || "No summary available."}</p>
                                        <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', display: 'flex', gap: '1rem' }}>
                                            <a href={n.url} target="_blank" rel="noreferrer" style={{ color: 'var(--color-accent)' }}>Read Source</a>
                                            <span style={{
                                                color: n.sentiment === 'POSITIVE' ? COLORS.up : (n.sentiment === 'NEGATIVE' ? COLORS.down : COLORS.textSecondary)
                                            }}>
                                                Sentiment: {n.sentiment}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'tech' && (
                            <div>
                                <TechChart data={macdData} type="MACD" colors={COLORS} />
                                <TechChart data={bbData} type="BOLLINGER" colors={COLORS} />
                            </div>
                        )}
                    </>
                )}
            </div>

        </div>
    );
};

export default DashboardPage;
