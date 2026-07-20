import { useCallback, useState } from "react";
import ModelStatusBar from "./components/ModelStatusBar";
import Controls from "./components/Controls";
import PredictionCard from "./components/PredictionCard";
import PriceHistoryChart from "./components/PriceHistoryChart";
import DriftPanel from "./components/DriftPanel";
import PredictionHistoryTable from "./components/PredictionHistoryTable";
import type { ModelType } from "./types";

export default function App() {
  const [ticker, setTicker] = useState("NVDA");
  const [modelType, setModelType] = useState<ModelType>("lstm");
  const [historyRefreshSignal, setHistoryRefreshSignal] = useState(0);

  const handlePredicted = useCallback(() => {
    setHistoryRefreshSignal((n) => n + 1);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Quantum Stock MLOps Dashboard</h1>
        <ModelStatusBar />
      </header>

      <Controls
        ticker={ticker}
        onTickerChange={setTicker}
        modelType={modelType}
        onModelTypeChange={setModelType}
      />

      <div className="layout">
        <div className="layout-main">
          <PriceHistoryChart ticker={ticker} modelType={modelType} />
          <PredictionHistoryTable ticker={ticker} refreshSignal={historyRefreshSignal} />
        </div>
        <div className="layout-side">
          <PredictionCard ticker={ticker} modelType={modelType} onPredicted={handlePredicted} />
          <DriftPanel ticker={ticker} />
        </div>
      </div>
    </div>
  );
}
