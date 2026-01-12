import React, { useEffect, useRef } from 'react';
import { createChart, CrosshairMode, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts';

const PriceChart = ({ data, rsiData, colors }) => {
    const chartContainerRef = useRef();
    const volumeContainerRef = useRef();
    const rsiContainerRef = useRef();

    const chartRef = useRef();
    const volumeChartRef = useRef();
    const rsiChartRef = useRef();

    useEffect(() => {
        if (!data || data.length === 0) return;

        // --- 1. Main Price Chart ---
        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { color: 'transparent' },
                textColor: colors.textSecondary
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            crosshair: { mode: CrosshairMode.Normal },
            timeScale: { visible: true, timeVisible: true },
            height: 300,
        });

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: colors.up,
            downColor: colors.down,
            borderVisible: false,
            wickUpColor: colors.up,
            wickDownColor: colors.down,
        });

        // Format data: [Time, Open, High, Low, Close, Volume]
        const candleData = data.map(d => ({
            time: d[0] / 1000, // Unix timestamp in seconds
            open: d[1],
            high: d[2],
            low: d[3],
            close: d[4],
        }));
        candleSeries.setData(candleData);
        chartRef.current = chart;

        // --- 2. Volume Chart ---
        const volChart = createChart(volumeContainerRef.current, {
            layout: { background: { color: 'transparent' }, textColor: colors.textSecondary },
            grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(255, 255, 255, 0.05)' } },
            timeScale: { visible: false },
            height: 100,
        });

        const volumeSeries = volChart.addSeries(HistogramSeries, {
            color: colors.accent,
            priceFormat: { type: 'volume' },
        });

        const volumeData = data.map(d => ({
            time: d[0] / 1000,
            value: d[5],
            color: d[4] >= d[1] ? colors.up : colors.down,
        }));
        volumeSeries.setData(volumeData);
        volumeChartRef.current = volChart;

        // --- 3. RSI Chart ---
        let rsiChart = null;
        if (rsiData && rsiData.length > 0) {
            rsiChart = createChart(rsiContainerRef.current, {
                layout: { background: { color: 'transparent' }, textColor: colors.textSecondary },
                grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(255, 255, 255, 0.05)' } },
                timeScale: { visible: true, timeVisible: true },
                height: 100,
            });

            const rsiSeries = rsiChart.addSeries(LineSeries, {
                color: colors.accent,
                lineWidth: 2,
            });

            const rsiMapped = rsiData.map(d => ({
                time: d.time / 1000,
                value: d.value,
            }));
            rsiSeries.setData(rsiMapped);

            // Add standard lines (30, 70)
            const lineUp = rsiChart.addSeries(LineSeries, { color: 'rgba(255,255,255,0.3)', lineWidth: 1, lineStyle: 2 });
            lineUp.setData(rsiMapped.map(d => ({ time: d.time, value: 70 })));
            const lineDown = rsiChart.addSeries(LineSeries, { color: 'rgba(255,255,255,0.3)', lineWidth: 1, lineStyle: 2 });
            lineDown.setData(rsiMapped.map(d => ({ time: d.time, value: 30 })));

            rsiChartRef.current = rsiChart;
        }

        // --- Syncing (Basic) ---
        // Lightweight charts doesn't have built-in sync API easily accessible without wrapper.
        // For now, let's keep them separate but aligned.
        // Ideally we sync VisibleLogicalRange.

        const syncCharts = (source, targets) => {
            source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                targets.forEach(t => t && t.timeScale().setVisibleLogicalRange(range));
            });
        };

        const charts = [chart, volChart, rsiChart].filter(Boolean);
        charts.forEach(c => {
            const others = charts.filter(x => x !== c);
            syncCharts(c, others);
        });

        return () => {
            chart.remove();
            volChart.remove();
            if (rsiChart) rsiChart.remove();
        };
    }, [data, rsiData]);

    return (
        <div className="flex-col" style={{ gap: '0.5rem' }}>
            <div ref={chartContainerRef} style={{ width: '100%' }} />
            <div ref={volumeContainerRef} style={{ width: '100%' }} />
            <div ref={rsiContainerRef} style={{ width: '100%' }} />
        </div>
    );
};

export default PriceChart;
