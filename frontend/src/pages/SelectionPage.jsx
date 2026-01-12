import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, TrendingUp, Cpu } from 'lucide-react';
import { marketApi } from '../services/api';

const SelectionPage = () => {
    const [keyword, setKeyword] = useState('');
    const [tickers, setTickers] = useState([]);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        fetchTickers();
    }, []);

    const fetchTickers = async (query = '') => {
        setLoading(true);
        try {
            const response = await marketApi.getTickers(query);
            setTickers(response.data);
        } catch (error) {
            console.error("Failed to fetch tickers", error);
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = (e) => {
        const val = e.target.value;
        setKeyword(val);
        fetchTickers(val);
    };

    const handleSelect = (symbol) => {
        navigate(`/dashboard/${symbol}`);
    };

    return (
        <div className="flex-center" style={{ minHeight: '100vh', flexDirection: 'column', gap: '2rem' }}>

            <div className="text-center animate-fade-in">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem', marginBottom: '1rem' }}>
                    <Cpu size={48} color="var(--color-accent)" />
                    <h1 style={{ fontSize: '3rem', margin: 0, background: 'linear-gradient(to right, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        Antigravity
                    </h1>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem' }}>
                    AI-Powered Crypto Prediction Engine
                </p>
            </div>

            <div className="card animate-fade-in" style={{ width: '100%', maxWidth: '600px', animationDelay: '0.1s' }}>
                <div style={{ position: 'relative', marginBottom: '1.5rem' }}>
                    <Search style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                    <input
                        type="text"
                        placeholder="Search coin (e.g. BTC, ETH)"
                        value={keyword}
                        onChange={handleSearch}
                        style={{
                            width: '100%',
                            padding: '1rem 1rem 1rem 3rem',
                            backgroundColor: 'var(--bg-app)',
                            border: '1px solid var(--bg-hover)',
                            borderRadius: 'var(--radius-md)',
                            color: 'white',
                            fontSize: '1.1rem',
                            outline: 'none',
                            boxSizing: 'border-box'
                        }}
                    />
                </div>

                <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                    {loading ? (
                        <div className="text-center p-4" style={{ color: 'var(--text-secondary)' }}>Scanning market...</div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {tickers.length === 0 ? (
                                <div className="text-center p-4" style={{ color: 'var(--text-secondary)' }}>No tickers found</div>
                            ) : (
                                tickers.map((t) => (
                                    <div
                                        key={t.symbol}
                                        onClick={() => handleSelect(t.symbol)}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            padding: '1rem',
                                            borderRadius: 'var(--radius-md)',
                                            cursor: 'pointer',
                                            transition: 'background 0.2s',
                                        }}
                                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-hover)'}
                                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                            <div style={{
                                                width: '40px', height: '40px', borderRadius: '50%',
                                                background: 'linear-gradient(135deg, var(--bg-hover), var(--bg-card))',
                                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                fontWeight: 'bold', color: 'var(--color-accent)'
                                            }}>
                                                {t.symbol.substring(0, 1)}
                                            </div>
                                            <div>
                                                <div style={{ fontWeight: 'bold' }}>{t.symbol}</div>
                                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{t.exchange} • {t.type}</div>
                                            </div>
                                        </div>
                                        <TrendingUp size={16} color="var(--color-accent)" />
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </div>
            </div>

            <button
                onClick={() => navigate('/evaluation')}
                style={{
                    background: 'transparent',
                    border: '1px solid var(--bg-hover)',
                    color: 'var(--text-secondary)',
                    padding: '0.5rem 1rem',
                    borderRadius: 'var(--radius-md)',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                }}
            >
                Or evaluate existing models directly
            </button>
        </div>
    );
};

export default SelectionPage;
