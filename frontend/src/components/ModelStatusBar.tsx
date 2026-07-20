import { useApi } from "../useApi";
import { fetchModelsStatus } from "../api";

export default function ModelStatusBar() {
  const { data, loading, error } = useApi(() => fetchModelsStatus(), []);

  if (loading) return <div className="status-bar">Loading model status…</div>;
  if (error) return <div className="status-bar error">Could not load model status: {error}</div>;
  if (!data) return null;

  return (
    <div className="status-bar">
      {Object.entries(data).map(([modelType, status]) => (
        <div key={modelType} className={`model-badge ${status.production_version ? "live" : "idle"}`}>
          <span className="model-badge-name">{modelType.toUpperCase()}</span>
          {status.production_version ? (
            <span className="model-badge-version">v{status.production_version} in production</span>
          ) : (
            <span className="model-badge-version">not yet trained/promoted</span>
          )}
        </div>
      ))}
    </div>
  );
}
