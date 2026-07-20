import { useApi } from "../useApi";
import { fetchTickers } from "../api";
import type { ModelType } from "../types";

interface Props {
  ticker: string;
  onTickerChange: (ticker: string) => void;
  modelType: ModelType;
  onModelTypeChange: (modelType: ModelType) => void;
}

export default function Controls({ ticker, onTickerChange, modelType, onModelTypeChange }: Props) {
  const { data: tickers, loading, error } = useApi(() => fetchTickers(), []);

  return (
    <div className="controls">
      <label>
        Ticker
        <select value={ticker} onChange={(e) => onTickerChange(e.target.value)} disabled={loading}>
          {loading && <option>Loading…</option>}
          {error && <option>Failed to load tickers</option>}
          {tickers?.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>

      <label>
        Model
        <select value={modelType} onChange={(e) => onModelTypeChange(e.target.value as ModelType)}>
          <option value="lstm">LSTM</option>
          <option value="qlstm">QLSTM</option>
        </select>
      </label>
    </div>
  );
}
