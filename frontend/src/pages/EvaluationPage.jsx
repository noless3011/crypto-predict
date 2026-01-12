import React, { useState } from 'react';
import { evaluateApi } from '../services/api';
import { ArrowLeft, Play, BarChart2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const EvaluationPage = () => {
    const navigate = useNavigate();
    const [ticker, setTicker] = useState('BTCUSDT');
    const [modelType, setModelType] = useState('lightgbm');
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState('2023-06-30');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleEvaluate = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResults(null);
        try {
            const response = await evaluateApi.evaluateModel(ticker, modelType, startDate, endDate);
            setResults(response.data);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "Evaluation failed");
        } finally {
            setLoading(false);
        }
    };

    const MetricCard = ({ label, value, color }) => (
        <div className="card" style={{ padding: '1rem', flex: 1, minWidth: '150px', borderLeft: `4px solid ${color || 'var(--color-primary)'}` }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{label}</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{value !== null && value !== undefined ? (typeof value === 'number' ? value.toFixed(4) : value) : '-'}</div>
        </div>
    );

    return (
        <div className="flex-center" style={{ minHeight: '100vh', flexDirection: 'column', padding: '2rem', boxSizing: 'border-box' }}>
            <div style={{ width: '100%', maxWidth: '1000px' }}>
                <button onClick={() => navigate('/')} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', marginBottom: '1rem' }}>
                    <ArrowLeft size={20} /> Back to Home
                </button>

                <h1 style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <BarChart2 size={32} color="var(--color-accent)" />
                    Model Evaluation
                </h1>

                <div className="card" style={{ marginBottom: '2rem' }}>
                    <form onSubmit={handleEvaluate} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', alignItems: 'end' }}>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Ticker</label>
                            <input
                                type="text"
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value)}
                                style={{ width: '100%', padding: '0.8rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-hover)', background: 'var(--bg-app)', color: 'white' }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Model</label>
                            <select
                                value={modelType}
                                onChange={(e) => setModelType(e.target.value)}
                                style={{ width: '100%', padding: '0.8rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-hover)', background: 'var(--bg-app)', color: 'white' }}
                            >
                                <option value="lightgbm">LightGBM</option>
                                <option value="lstm">LSTM</option>
                                <option value="transformer">Transformer</option>
                            </select>
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Start Date</label>
                            <input
                                type="date"
                                value={startDate}
                                onChange={(e) => setStartDate(e.target.value)}
                                style={{ width: '100%', padding: '0.8rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-hover)', background: 'var(--bg-app)', color: 'white' }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>End Date</label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                style={{ width: '100%', padding: '0.8rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-hover)', background: 'var(--bg-app)', color: 'white' }}
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={loading}
                            style={{
                                padding: '0.8rem',
                                borderRadius: 'var(--radius-md)',
                                background: 'var(--color-accent)',
                                color: 'black',
                                fontWeight: 'bold',
                                border: 'none',
                                cursor: loading ? 'wait' : 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.5rem'
                            }}
                        >
                            {loading ? 'Running...' : <><Play size={18} /> Run Evaluation</>}
                        </button>
                    </form>
                </div>

                {error && (
                    <div className="card" style={{ background: '#ef444420', borderColor: '#ef4444', color: '#ef4444', marginBottom: '2rem' }}>
                        {error}
                    </div>
                )}

                {results && (
                    <div className="animate-fade-in">
                        <h2 style={{ marginBottom: '1rem' }}>Results: {results.model_name}</h2>

                        <h3 style={{ borderBottom: '1px solid var(--bg-hover)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Overall Metrics</h3>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
                            <MetricCard label="Accuracy" value={results.metrics.accuracy} />
                            <MetricCard label="Macro F1" value={results.metrics.f1_macro} />
                            <MetricCard label="Weighted F1" value={results.metrics.f1_weighted} />
                        </div>

                        <h3 style={{ borderBottom: '1px solid var(--bg-hover)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Class Performance (Up/Buy)</h3>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
                            <MetricCard label="Precision (Buy Reliability)" value={results.metrics.precision_up} color="#22c55e" />
                            <MetricCard label="Recall (Opportunity Capture)" value={results.metrics.recall_up} color="#eab308" />
                            <MetricCard label="F1 Score" value={results.metrics.f1_up} color="#3b82f6" />
                        </div>

                        {results.metrics.auc_roc_macro && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
                                <MetricCard label="AUC-ROC (Macro)" value={results.metrics.auc_roc_macro} color="#a855f7" />
                            </div>
                        )}

                        <div className="card">
                            <h3>Raw Data Summary</h3>
                            <p>Total Samples: {results.actuals.length}</p>
                            <p>Predictions returned. Charts coming soon!</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default EvaluationPage;
