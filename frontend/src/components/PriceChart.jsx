import React, { useEffect, useRef, useState } from "react";
import { createChart, ColorType } from "lightweight-charts";
import { getHistory, getPrediction, getModels } from "../services/api";
import { Loader } from "lucide-react";

const PriceChart = ({ startTime, endTime, shouldLoad }) => {
  const chartContainerRef = useRef();
  const [data, setData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [predictionHours, setPredictionHours] = useState(5);
  const [loading, setLoading] = useState(true);
  const [fetchingData, setFetchingData] = useState(false);
  const [predicting, setPredicting] = useState(false);

  // 1. Init: Fetch Models
  useEffect(() => {
    const initModels = async () => {
      try {
        const modelsList = await getModels();
        setModels(modelsList);
        if (modelsList.length > 0) setSelectedModel(modelsList[0]);
      } catch (error) {
        console.error("Model fetch failed", error);
      } finally {
        setLoading(false);
      }
    };
    initModels();
  }, []);

  // Fetch data when shouldLoad trigger fires
  useEffect(() => {
    if (!shouldLoad || !startTime || !endTime) return;

    const fetchData = async () => {
      setFetchingData(true);
      try {
        const startDate = new Date(startTime * 1000).toISOString();
        const endDate = new Date(endTime * 1000).toISOString();

        const historyData = await getHistory(startDate, endDate);

        // Format for lightweight charts
        const formattedData = historyData
          .map((d) => ({
            time: new Date(d.Date).getTime() / 1000,
            open: d.Open,
            high: d.High,
            low: d.Low,
            close: d.Close,
          }))
          .sort((a, b) => a.time - b.time);

        // Remove duplicates on time
        const uniqueData = [];
        const timeSet = new Set();
        for (const item of formattedData) {
          if (!timeSet.has(item.time)) {
            timeSet.add(item.time);
            uniqueData.push(item);
          }
        }

        setData(uniqueData);
      } catch (err) {
        console.error("Failed to fetch data", err);
      } finally {
        setFetchingData(false);
      }
    };

    fetchData();
  }, [shouldLoad, startTime, endTime]);

  // Manual Prediction Handler
  const handlePredict = async () => {
    if (!selectedModel || data.length === 0) return;

    setPredicting(true);
    try {
      // Pass the time range and prediction hours to the API
      const startDate = startTime
        ? new Date(startTime * 1000).toISOString()
        : null;
      const endDate = endTime ? new Date(endTime * 1000).toISOString() : null;

      const result = await getPrediction(
        selectedModel,
        startDate,
        endDate,
        predictionHours,
      );
      // Result is array of { time: isoString, value: number }
      const formattedPred = result
        .map((p) => ({
          time: new Date(p.time).getTime() / 1000,
          value: p.value,
        }))
        .filter((p) => p.value !== null && !isNaN(p.value)); // Filter invalid points

      setPrediction(formattedPred);
    } catch (error) {
      console.error("Prediction failed", error);
      setPrediction(null);
    } finally {
      setPredicting(false);
    }
  };

  // Chart Rendering
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Only render if we have data or at least container
    // But lightweight charts can be empty.

    // Clean up previous chart if creating new one?
    // Usually we create chart once and update series.
    // But simpler is to recreate or use series Ref.
    // Given the structure, recreating is safer for state reset unless we track series.
    // But recreating flickers.
    // Let's try to keeping chart instance?
    // The previous code recreated chart on data change. Let's stick to that for now to minimize complex bugs,
    // though optimizing to update series is better.

    if (chartContainerRef.current.firstChild) {
      chartContainerRef.current.innerHTML = "";
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1e293b" },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 600,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: "#475569",
      },
      rightPriceScale: {
        borderColor: "#475569",
      },
    });

    // Candlestick Series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    if (data.length > 0) {
      candleSeries.setData(data);
    }

    // Prediction Series
    if (prediction && prediction.length > 0) {
      const predSeries = chart.addLineSeries({
        color: "#facc15", // Yellow
        lineWidth: 2,
        lineStyle: 0, // Solid
        title: "Prediction",
      });
      predSeries.setData(prediction);
    }

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      chart.remove();
      window.removeEventListener("resize", handleResize);
    };
  }, [data, prediction]);

  // Helper to format timestamps for display
  const formatDate = (unixTime) => {
    if (!unixTime) return "";
    return new Date(unixTime * 1000).toLocaleString();
  };

  // Get current view dates from data slice
  const dispStartTime = data.length > 0 ? data[0].time : null;
  const dispEndTime = data.length > 0 ? data[data.length - 1].time : null;

  if (loading) return <div className="p-4 text-center">Loading...</div>;

  return (
    <div className="h-full flex flex-col p-4 bg-slate-900">
      {/* Controls Header */}
      <div className="flex flex-wrap items-center gap-4 mb-4 bg-slate-800 p-4 rounded-lg border border-slate-700">
        <div className="flex flex-col">
          <label className="text-xs text-slate-400 mb-1">Model Selection</label>
          <select
            className="bg-slate-700 text-white rounded px-3 py-1 border border-slate-600 focus:outline-none focus:border-blue-500"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col">
          <label className="text-xs text-slate-400 mb-1">
            Prediction Hours (divisible by 5)
          </label>
          <input
            type="number"
            min="5"
            step="5"
            value={predictionHours}
            onChange={(e) => {
              const value = parseInt(e.target.value);
              if (value > 0 && value % 5 === 0) {
                setPredictionHours(value);
              }
            }}
            className="bg-slate-700 text-white rounded px-3 py-1 border border-slate-600 focus:outline-none focus:border-blue-500 w-24"
          />
        </div>

        <button
          onClick={handlePredict}
          disabled={!selectedModel || data.length === 0 || predicting}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded font-medium transition-colors flex items-center gap-2"
        >
          {predicting ? (
            <>
              <Loader className="animate-spin" size={16} />
              Predicting...
            </>
          ) : (
            "Predict"
          )}
        </button>

        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="text-sm font-mono text-blue-400">
            {dispStartTime ? formatDate(dispStartTime) : "--"}
            <span className="mx-2 text-slate-500">to</span>
            {dispEndTime ? formatDate(dispEndTime) : "--"}
          </div>
        </div>

        {fetchingData && (
          <div className="text-blue-400 text-sm flex items-center gap-2">
            <Loader className="animate-spin" size={16} />
            Loading Data...
          </div>
        )}
      </div>

      {/* Chart Canvas */}
      <div className="flex-1 w-full bg-slate-800 rounded-lg border border-slate-700 p-2 overflow-hidden relative">
        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </div>
  );
};

export default PriceChart;
