import React, { useEffect, useRef, useState } from "react";
import { createChart, ColorType } from "lightweight-charts";
import { getIndicators } from "../services/api";

const TechIndicators = ({ startTime, endTime, shouldLoad }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  const chartContainerRef1 = useRef();
  const chartContainerRef2 = useRef();
  const chartContainerRef3 = useRef();

  useEffect(() => {
    if (!shouldLoad || !startTime || !endTime) return;

    const fetchData = async () => {
      try {
        setLoading(true);
        const startDate = new Date(startTime * 1000).toISOString();
        const endDate = new Date(endTime * 1000).toISOString();

        const params = {};
        if (startDate) params.start = startDate;
        if (endDate) params.end = endDate;

        const result = await getIndicators(params);

        result.sort((a, b) => new Date(a.Date) - new Date(b.Date));

        // Lightweight charts needs { time, value }
        const validData = result.filter(
          (d) => d.RSI !== null && d.MACD !== null,
        );
        setData(validData);
      } catch (error) {
        console.error("Failed to fetch indicators", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [shouldLoad, startTime, endTime]);

  // Effect for RSI Chart
  useEffect(() => {
    if (!data.length || !chartContainerRef1.current) return;

    const chart = createChart(chartContainerRef1.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1e293b" },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: chartContainerRef1.current.clientWidth,
      height: 250,
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const rsiSeries = chart.addLineSeries({
      color: "#c084fc",
      lineWidth: 2,
      title: "RSI",
    });
    rsiSeries.setData(
      data.map((d) => ({
        time: new Date(d.Date).getTime() / 1000,
        value: d.RSI,
      })),
    );

    // Add 70/30 lines
    const line70 = chart.addLineSeries({
      color: "#ef4444",
      lineWidth: 1,
      lineStyle: 2,
      title: "Overbought",
    });
    line70.setData(
      data.map((d) => ({ time: new Date(d.Date).getTime() / 1000, value: 70 })),
    );

    const line30 = chart.addLineSeries({
      color: "#22c55e",
      lineWidth: 1,
      lineStyle: 2,
      title: "Oversold",
    });
    line30.setData(
      data.map((d) => ({ time: new Date(d.Date).getTime() / 1000, value: 30 })),
    );

    window.addEventListener("resize", () =>
      chart.resize(chartContainerRef1.current.clientWidth, 250),
    );
    return () => {
      chart.remove();
      window.removeEventListener("resize", () =>
        chart.resize(chartContainerRef1.current.clientWidth, 250),
      );
    };
  }, [data]);

  // Effect for MACD Chart
  useEffect(() => {
    if (!data.length || !chartContainerRef2.current) return;

    const chart = createChart(chartContainerRef2.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1e293b" },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: chartContainerRef2.current.clientWidth,
      height: 250,
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const macdSeries = chart.addLineSeries({
      color: "#3b82f6",
      lineWidth: 2,
      title: "MACD",
    });
    macdSeries.setData(
      data.map((d) => ({
        time: new Date(d.Date).getTime() / 1000,
        value: d.MACD,
      })),
    );

    const signalSeries = chart.addLineSeries({
      color: "#f97316",
      lineWidth: 2,
      title: "Signal",
    });
    signalSeries.setData(
      data.map((d) => ({
        time: new Date(d.Date).getTime() / 1000,
        value: d.MACD_Signal,
      })),
    );

    window.addEventListener("resize", () =>
      chart.resize(chartContainerRef2.current.clientWidth, 250),
    );
    return () => {
      chart.remove();
      window.removeEventListener("resize", () =>
        chart.resize(chartContainerRef2.current.clientWidth, 250),
      );
    };
  }, [data]);

  // Effect for ATR Chart
  useEffect(() => {
    if (!data.length || !chartContainerRef3.current) return;

    const chart = createChart(chartContainerRef3.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1e293b" },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: chartContainerRef3.current.clientWidth,
      height: 250,
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const atrSeries = chart.addLineSeries({
      color: "#22d3ee",
      lineWidth: 2,
      title: "ATR",
    });
    atrSeries.setData(
      data.map((d) => ({
        time: new Date(d.Date).getTime() / 1000,
        value: d.ATR,
      })),
    );

    window.addEventListener("resize", () =>
      chart.resize(chartContainerRef3.current.clientWidth, 250),
    );
    return () => {
      chart.remove();
      window.removeEventListener("resize", () =>
        chart.resize(chartContainerRef3.current.clientWidth, 250),
      );
    };
  }, [data]);

  if (loading)
    return <div className="p-4 text-center">Loading Indicators...</div>;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-6">
      <div>
        <h3 className="text-lg font-bold mb-2">
          RSI (Relative Strength Index)
        </h3>
        <div
          ref={chartContainerRef1}
          className="w-full rounded-lg overflow-hidden border border-slate-700"
        />
      </div>
      <div>
        <h3 className="text-lg font-bold mb-2">
          MACD (Moving Average Convergence Divergence)
        </h3>
        <div
          ref={chartContainerRef2}
          className="w-full rounded-lg overflow-hidden border border-slate-700"
        />
      </div>
      <div>
        <h3 className="text-lg font-bold mb-2">ATR (Average True Range)</h3>
        <div
          ref={chartContainerRef3}
          className="w-full rounded-lg overflow-hidden border border-slate-700"
        />
      </div>
    </div>
  );
};

export default TechIndicators;
