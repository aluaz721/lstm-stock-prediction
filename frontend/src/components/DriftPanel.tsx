import { useApi } from "../useApi";
import { fetchDriftStatus } from "../api";

interface Props {
  ticker: string;
}

export default function DriftPanel({ ticker }: Props) {
  const { data, loading, error, refetch } = useApi(() => fetchDriftStatus(ticker), [ticker]);

  const anyDrifted = data?.some((row) => row.is_drifted) ?? false;

  return (
    <div className="card drift-card">
      <div className="card-header">
        <h2>Drift Status</h2>
        <button onClick={refetch} disabled={loading}>
          {loading ? "Checking…" : "Refresh"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {data && data.length === 0 && (
        <p className="muted">
          No drift checks recorded yet for {ticker}. Run{" "}
          <code>python scripts/refresh_and_check_drift.py</code> to populate this.
        </p>
      )}

      {data && data.length > 0 && (
        <>
          {anyDrifted && <p className="drift-warning">⚠ Drift detected in one or more features</p>}
          <table className="drift-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>PSI Score</th>
                <th>Threshold</th>
                <th>Status</th>
                <th>Checked</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row) => (
                <tr key={row.feature_name} className={row.is_drifted ? "drifted" : ""}>
                  <td>{row.feature_name}</td>
                  <td>{row.psi_score.toFixed(4)}</td>
                  <td>{row.threshold.toFixed(2)}</td>
                  <td>{row.is_drifted ? "⚠ Drifted" : "OK"}</td>
                  <td>{new Date(row.checked_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
