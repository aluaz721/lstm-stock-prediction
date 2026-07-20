import { useApi } from "../useApi";
import { fetchPredictionHistory } from "../api";

interface Props {
  ticker: string;
  refreshSignal: number;
}

export default function PredictionHistoryTable({ ticker, refreshSignal }: Props) {
  const { data, loading, error } = useApi(
    () => fetchPredictionHistory(ticker, 20),
    [ticker, refreshSignal]
  );

  return (
    <div className="card history-card">
      <h2>Recent Predictions — {ticker}</h2>
      {loading && <p>Loading…</p>}
      {error && <p className="error">{error}</p>}

      {data && data.length === 0 && (
        <p className="muted">No predictions logged yet. Click "Refresh" on the prediction card above.</p>
      )}

      {data && data.length > 0 && (
        <table className="history-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Predicted</th>
              <th>From</th>
              <th>Change</th>
              <th>For date</th>
              <th>Generated</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => {
              const change = row.predicted_close - row.last_known_close;
              const changePct = (change / row.last_known_close) * 100;
              return (
                <tr key={i}>
                  <td>{row.model_type.toUpperCase()} v{row.model_version}</td>
                  <td>${row.predicted_close.toFixed(2)}</td>
                  <td>${row.last_known_close.toFixed(2)}</td>
                  <td className={change >= 0 ? "up" : "down"}>
                    {change >= 0 ? "▲" : "▼"} {Math.abs(changePct).toFixed(2)}%
                  </td>
                  <td>{new Date(row.prediction_for_date).toLocaleDateString()}</td>
                  <td>{new Date(row.created_at).toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
