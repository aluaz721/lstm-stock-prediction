import { useEffect } from "react";
import { useApi } from "../useApi";
import { fetchPrediction } from "../api";
import type { ModelType } from "../types";

interface Props {
  ticker: string;
  modelType: ModelType;
  onPredicted?: () => void;
}

export default function PredictionCard({ ticker, modelType, onPredicted }: Props) {
  const { data, loading, error, refetch } = useApi(
    () => fetchPrediction(ticker, modelType),
    [ticker, modelType]
  );

  // Every successful fetch (including the initial one) corresponds to a
  // new row written to prediction_log on the backend -- /predict has a
  // side effect despite being a GET, matching how the API was built.
  // Notify the parent so sibling components (the history table) can stay
  // in sync without polling.
  useEffect(() => {
    if (data) onPredicted?.();
  }, [data, onPredicted]);

  const change = data ? data.predicted_close - data.last_known_close : null;
  const changePct = data ? (change! / data.last_known_close) * 100 : null;
  const isUp = change !== null && change >= 0;

  return (
    <div className="card prediction-card">
      <div className="card-header">
        <h2>Next-Day Prediction</h2>
        <button onClick={refetch} disabled={loading}>
          {loading ? "Predicting…" : "Refresh"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {data && (
        <>
          <div className={`prediction-value ${isUp ? "up" : "down"}`}>
            ${data.predicted_close.toFixed(2)}
            <span className="prediction-change">
              {isUp ? "▲" : "▼"} {Math.abs(changePct!).toFixed(2)}%
            </span>
          </div>
          <dl className="prediction-meta">
            <dt>Last known close</dt>
            <dd>${data.last_known_close.toFixed(2)}</dd>
            <dt>Prediction for</dt>
            <dd>{new Date(data.prediction_for_date).toLocaleDateString()}</dd>
            <dt>Model version</dt>
            <dd>v{data.model_version}</dd>
            <dt>Generated at</dt>
            <dd>{new Date(data.generated_at).toLocaleTimeString()}</dd>
          </dl>
        </>
      )}
    </div>
  );
}
