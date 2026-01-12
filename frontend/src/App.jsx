import React, { useState, useEffect, useRef, useMemo } from "react";
import PriceChart from "./components/PriceChart";
import TechIndicators from "./components/TechIndicators";
import News from "./components/News";
import { LineChart, Newspaper, Activity, Loader } from "lucide-react";
import { getHistoryMeta } from "./services/api";

function App() {
  const [activeTab, setActiveTab] = useState("chart");

  // Slider state - all times in unix seconds
  const [sliderBeginTime, setSliderBeginTime] = useState(null);
  const [sliderEndTime, setSliderEndTime] = useState(null);
  const [knob1Time, setKnob1Time] = useState(null);
  const [knob2Time, setKnob2Time] = useState(null);
  const [hoverKnob, setHoverKnob] = useState(null); // 'knob1' or 'knob2'
  const [shouldLoadData, setShouldLoadData] = useState(false);

  // Initialize slider times on mount (last 3 months)
  useEffect(() => {
    const now = Math.floor(Date.now() / 1000);
    const threeMonthsAgo = now - 90 * 24 * 60 * 60;
    const oneYearAgo = now - 365 * 24 * 60 * 60;

    setSliderBeginTime(oneYearAgo);
    setSliderEndTime(now);
    setKnob1Time(threeMonthsAgo);
    setKnob2Time(now);
  }, []);

  // Helper to format timestamps for display
  const formatDate = (unixTime) => {
    if (!unixTime) return "";
    return new Date(unixTime * 1000).toLocaleString();
  };

  // Helper to format date for input field (YYYY-MM-DDTHH:mm)
  const formatDateForInput = (unixTime) => {
    if (!unixTime) return "";
    const date = new Date(unixTime * 1000);
    return date.toISOString().slice(0, 16);
  };

  // Helper to parse input date to unix time
  const parseDateInput = (dateString) => {
    if (!dateString) return null;
    return Math.floor(new Date(dateString).getTime() / 1000);
  };

  // Handle preset range buttons
  const handlePresetRange = (range) => {
    if (!sliderEndTime) return;

    const now = sliderEndTime;
    let newBeginTime;

    switch (range) {
      case "1w":
        newBeginTime = now - 7 * 24 * 60 * 60;
        break;
      case "1m":
        newBeginTime = now - 30 * 24 * 60 * 60;
        break;
      case "3m":
        newBeginTime = now - 90 * 24 * 60 * 60;
        break;
      case "1y":
        newBeginTime = now - 365 * 24 * 60 * 60;
        break;
      default:
        return;
    }

    const absoluteMin = sliderEndTime - 2 * 365 * 24 * 60 * 60; // 2 years max
    setSliderBeginTime(Math.max(absoluteMin, newBeginTime));
    setKnob1Time(Math.max(absoluteMin, newBeginTime));
    setKnob2Time(now);
  };

  // Calculate slider knob positions as percentages
  const getKnobPositions = () => {
    if (!sliderBeginTime || !sliderEndTime || !knob1Time || !knob2Time) {
      return { knob1Pos: 0, knob2Pos: 100 };
    }

    const totalRange = sliderEndTime - sliderBeginTime;
    const knob1Pos = ((knob1Time - sliderBeginTime) / totalRange) * 100;
    const knob2Pos = ((knob2Time - sliderBeginTime) / totalRange) * 100;

    return { knob1Pos, knob2Pos };
  };

  // Handle Load Data button
  const handleLoadData = () => {
    setShouldLoadData(true);
    // Reset after a brief delay to allow components to react
    setTimeout(() => setShouldLoadData(false), 100);
  };

  return (
    <div className="flex h-screen w-screen bg-slate-900 text-slate-50 overflow-hidden">
      {/* Sidebar / Navbar */}
      <nav className="w-20 bg-slate-950 flex flex-col items-center py-6 border-r border-slate-800 z-10">
        <div className="mb-8 p-2 bg-blue-600 rounded-lg">
          {/* Logo placeholder */}
          <Activity size={24} className="text-white" />
        </div>

        <div className="flex flex-col gap-6 w-full">
          <button
            onClick={() => setActiveTab("chart")}
            className={`p-3 mx-auto rounded-xl transition-all ${activeTab === "chart" ? "bg-slate-800 text-blue-400" : "text-slate-500 hover:text-slate-300"}`}
            title="Price Chart"
          >
            <LineChart size={24} />
          </button>

          <button
            onClick={() => setActiveTab("indicators")}
            className={`p-3 mx-auto rounded-xl transition-all ${activeTab === "indicators" ? "bg-slate-800 text-purple-400" : "text-slate-500 hover:text-slate-300"}`}
            title="Technical Indicators"
          >
            <Activity size={24} />
          </button>

          <button
            onClick={() => setActiveTab("news")}
            className={`p-3 mx-auto rounded-xl transition-all ${activeTab === "news" ? "bg-slate-800 text-green-400" : "text-slate-500 hover:text-slate-300"}`}
            title="News"
          >
            <Newspaper size={24} />
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 h-full relative flex flex-col">
        <div className="flex-1 overflow-hidden relative">
          {activeTab === "chart" && (
            <PriceChart
              startTime={knob1Time}
              endTime={knob2Time}
              shouldLoad={shouldLoadData}
            />
          )}
          {activeTab === "indicators" && (
            <TechIndicators
              startTime={knob1Time}
              endTime={knob2Time}
              shouldLoad={shouldLoadData}
            />
          )}
          {activeTab === "news" && (
            <News
              startTime={knob1Time}
              endTime={knob2Time}
              shouldLoad={shouldLoadData}
            />
          )}
        </div>

        {/* Date Time Range Picker Footer */}
        <div className="bg-slate-950 border-t border-slate-800 p-4 z-20">
          {/* Preset Range Buttons */}
          <div className="flex gap-2 mb-4 items-center">
            <span className="text-sm text-slate-400 mr-2">Quick Range:</span>
            {["1w", "1m", "3m", "1y"].map((range) => (
              <button
                key={range}
                onClick={() => handlePresetRange(range)}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-sm transition-colors border border-slate-600"
              >
                {range.toUpperCase()}
              </button>
            ))}

            {/* Load Data Button */}
            <button
              onClick={handleLoadData}
              disabled={!knob1Time || !knob2Time}
              className="ml-auto px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded font-semibold text-sm transition-colors border border-blue-500 disabled:border-slate-500 flex items-center gap-2"
            >
              Load Data
            </button>
          </div>

          {/* Editable Date Inputs */}
          <div className="flex gap-4 mb-4">
            <div className="flex-1 flex flex-col">
              <label className="text-xs text-slate-400 mb-1">
                Slider Begin
              </label>
              <input
                type="datetime-local"
                value={formatDateForInput(sliderBeginTime)}
                onChange={(e) => {
                  const newTime = parseDateInput(e.target.value);
                  if (newTime && newTime < sliderEndTime) {
                    setSliderBeginTime(newTime);
                    if (knob1Time < newTime) setKnob1Time(newTime);
                  }
                }}
                className="bg-slate-700 text-white rounded px-3 py-1 border border-slate-600 focus:outline-none focus:border-blue-500 text-sm"
              />
            </div>

            <div className="flex-1 flex flex-col">
              <label className="text-xs text-slate-400 mb-1">Slider End</label>
              <input
                type="datetime-local"
                value={formatDateForInput(sliderEndTime)}
                onChange={(e) => {
                  const newTime = parseDateInput(e.target.value);
                  if (newTime && newTime > sliderBeginTime) {
                    setSliderEndTime(newTime);
                    if (knob2Time > newTime) setKnob2Time(newTime);
                  }
                }}
                className="bg-slate-700 text-white rounded px-3 py-1 border border-slate-600 focus:outline-none focus:border-blue-500 text-sm"
              />
            </div>
          </div>

          {/* Dual Range Slider */}
          <div className="relative pt-8 pb-4">
            {/* Knob 1 Tooltip */}
            {hoverKnob === "knob1" && knob1Time && (
              <div
                className="absolute bg-slate-900 text-white px-2 py-1 rounded text-xs whitespace-nowrap border border-slate-600 z-10"
                style={{
                  left: `${getKnobPositions().knob1Pos}%`,
                  top: "-30px",
                  transform: "translateX(-50%)",
                }}
              >
                {formatDate(knob1Time)}
              </div>
            )}

            {/* Knob 2 Tooltip */}
            {hoverKnob === "knob2" && knob2Time && (
              <div
                className="absolute bg-slate-900 text-white px-2 py-1 rounded text-xs whitespace-nowrap border border-slate-600 z-10"
                style={{
                  left: `${getKnobPositions().knob2Pos}%`,
                  top: "-30px",
                  transform: "translateX(-50%)",
                }}
              >
                {formatDate(knob2Time)}
              </div>
            )}

            {/* Custom Dual Range Slider */}
            <div className="relative h-2 bg-slate-700 rounded-full">
              {/* Active range highlight */}
              <div
                className="absolute h-full bg-blue-500 rounded-full"
                style={{
                  left: `${getKnobPositions().knob1Pos}%`,
                  width: `${getKnobPositions().knob2Pos - getKnobPositions().knob1Pos}%`,
                }}
              />

              {/* Knob 1 */}
              <div
                className="absolute w-4 h-4 bg-blue-400 rounded-full cursor-pointer hover:bg-blue-300 transition-colors border-2 border-white"
                style={{
                  left: `${getKnobPositions().knob1Pos}%`,
                  top: "50%",
                  transform: "translate(-50%, -50%)",
                }}
                onMouseEnter={() => setHoverKnob("knob1")}
                onMouseLeave={() => setHoverKnob(null)}
                onMouseDown={(e) => {
                  const slider = e.currentTarget.parentElement;
                  const rect = slider.getBoundingClientRect();

                  const handleMove = (moveE) => {
                    const x = Math.max(
                      0,
                      Math.min(moveE.clientX - rect.left, rect.width),
                    );
                    const percent = (x / rect.width) * 100;
                    const newTime =
                      sliderBeginTime +
                      ((sliderEndTime - sliderBeginTime) * percent) / 100;

                    if (newTime < knob2Time) {
                      setKnob1Time(newTime);
                    }
                  };

                  const handleUp = () => {
                    document.removeEventListener("mousemove", handleMove);
                    document.removeEventListener("mouseup", handleUp);
                  };

                  document.addEventListener("mousemove", handleMove);
                  document.addEventListener("mouseup", handleUp);
                }}
              />

              {/* Knob 2 */}
              <div
                className="absolute w-4 h-4 bg-blue-400 rounded-full cursor-pointer hover:bg-blue-300 transition-colors border-2 border-white"
                style={{
                  left: `${getKnobPositions().knob2Pos}%`,
                  top: "50%",
                  transform: "translate(-50%, -50%)",
                }}
                onMouseEnter={() => setHoverKnob("knob2")}
                onMouseLeave={() => setHoverKnob(null)}
                onMouseDown={(e) => {
                  const slider = e.currentTarget.parentElement;
                  const rect = slider.getBoundingClientRect();

                  const handleMove = (moveE) => {
                    const x = Math.max(
                      0,
                      Math.min(moveE.clientX - rect.left, rect.width),
                    );
                    const percent = (x / rect.width) * 100;
                    const newTime =
                      sliderBeginTime +
                      ((sliderEndTime - sliderBeginTime) * percent) / 100;

                    if (newTime > knob1Time) {
                      setKnob2Time(newTime);
                    }
                  };

                  const handleUp = () => {
                    document.removeEventListener("mousemove", handleMove);
                    document.removeEventListener("mouseup", handleUp);
                  };

                  document.addEventListener("mousemove", handleMove);
                  document.addEventListener("mouseup", handleUp);
                }}
              />
            </div>

            {/* Time labels below slider */}
            <div className="flex justify-between mt-2 text-xs text-slate-400">
              <span>
                {sliderBeginTime ? formatDate(sliderBeginTime) : "--"}
              </span>
              <span>{sliderEndTime ? formatDate(sliderEndTime) : "--"}</span>
            </div>
          </div>

          {/* Selected Range Display */}
          <div className="mt-3 text-center text-sm text-slate-300">
            <span className="font-mono text-green-400">
              {knob1Time ? formatDate(knob1Time) : "--"}
            </span>
            <span className="mx-2 text-slate-500">→</span>
            <span className="font-mono text-green-400">
              {knob2Time ? formatDate(knob2Time) : "--"}
            </span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
