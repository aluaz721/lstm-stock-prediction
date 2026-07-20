import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useApi } from "../useApi";
import { fetchPriceHistory, fetchForecast } from "../api";
import type { ModelType } from "../types";

interface Props {
  ticker: string;
  modelType: ModelType;
}

interface ChartPoint {
  date: string;
  actual?: number;
  forecast?: number;
}

export default function PriceHistoryChart({ ticker, modelType }: Props) {
  const history = useApi(() => fetchPriceHistory(ticker, 90), [ticker]);
  const forecast = useApi(() => fetchForecast(ticker, modelType, 10), [ticker, modelType]);

  const loading = history.loading || forecast.loading;
  const error = history.error ?? forecast.error;

  let chartData: ChartPoint[] = [];
  if (history.data && forecast.data) {
    const actualPoints: ChartPoint[] = history.data.map((p) => ({ date: p.date, actual: p.close }));

    // Anchor the forecast line to the last actual point so the two
    // series connect visually instead of leaving a gap.
    if (actualPoints.length > 0) {
      const last = actualPoints[actualPoints.length - 1];
      last.forecast = last.actual;
    }

    const forecastPoints: ChartPoint[] = forecast.data.forecast.map((p) => ({
      date: p.date,
      forecast: p.predicted_close,
    }));

    chartData = [...actualPoints, ...forecastPoints];
  }

  return (
    <div className="card chart-card">
      <h2>
        {ticker} — price history &amp; {modelType.toUpperCase()} forecast
      </h2>
      {loading && <p>Loading chart…</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
            <XAxis
              dataKey="date"
              tickFormatter={(d) => new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
              minTickGap={30}
            />
            <YAxis domain={["auto", "auto"]} tickFormatter={(v) => `$${v}`} width={70} />
            <Tooltip
              formatter={(value) =>
                typeof value === "number" ? `$${value.toFixed(2)}` : String(value)
              }
              labelFormatter={(d) => new Date(d).toLocaleDateString()}
            />
            <Legend />
            <Line type="monotone" dataKey="actual" name="Actual close" stroke="#2563eb" dot={false} strokeWidth={2} />
            <Line
              type="monotone"
              dataKey="forecast"
              name="Forecast"
              stroke="#dc2626"
              strokeDasharray="5 4"
              dot={{ r: 3 }}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
