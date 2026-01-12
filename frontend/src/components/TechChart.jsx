import React, { useEffect, useRef } from 'react';
import { createChart, LineSeries, HistogramSeries } from 'lightweight-charts';

const TechChart = ({ data, type, colors }) => {
    const containerRef = useRef();
    const chartRef = useRef();

    useEffect(() => {
        if (!data || data.length === 0) return;

        const chart = createChart(containerRef.current, {
            layout: { background: { color: 'transparent' }, textColor: colors.textSecondary },
            grid: { vertLines: { color: 'rgba(255, 255, 255, 0.05)' }, horzLines: { color: 'rgba(255, 255, 255, 0.05)' } },
            timeScale: { visible: true, timeVisible: true },
            height: 250,
        });

        if (type === 'MACD') {
            const macdSeries = chart.addSeries(LineSeries, { color: colors.accent, lineWidth: 2, title: 'MACD' });
            const signalSeries = chart.addSeries(LineSeries, { color: '#f59e0b', lineWidth: 2, title: 'Signal' });
            const histSeries = chart.addSeries(HistogramSeries, { color: colors.textSecondary, title: 'Hist' }); // Default color

            macdSeries.setData(data.map(d => ({ time: d.time / 1000, value: d.macd })));
            signalSeries.setData(data.map(d => ({ time: d.time / 1000, value: d.signal })));

            histSeries.setData(data.map(d => ({
                time: d.time / 1000,
                value: d.histogram,
                color: d.histogram >= 0 ? colors.up : colors.down
            })));
        }
        else if (type === 'BOLLINGER') {
            const upper = chart.addSeries(LineSeries, { color: colors.accent, lineWidth: 1, title: 'Upper' });
            const middle = chart.addSeries(LineSeries, { color: '#ffffff', lineWidth: 1, title: 'Middle' });
            const lower = chart.addSeries(LineSeries, { color: colors.accent, lineWidth: 1, title: 'Lower' });

            upper.setData(data.map(d => ({ time: d.time / 1000, value: d.upper })));
            middle.setData(data.map(d => ({ time: d.time / 1000, value: d.middle })));
            lower.setData(data.map(d => ({ time: d.time / 1000, value: d.lower })));
        }
        else {
            // Generic Line (EMA/SMA)
            const series = chart.addSeries(LineSeries, { color: colors.accent, lineWidth: 2 });
            series.setData(data.map(d => ({ time: d.time / 1000, value: d.value })));
        }

        chartRef.current = chart;

        return () => chart.remove();
    }, [data, type]);

    return (
        <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem', color: colors.textSecondary }}>{type}</h3>
            <div ref={containerRef} style={{ width: '100%' }} />
        </div>
    );
};

export default TechChart;
