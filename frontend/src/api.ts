import type {
  DriftCheckRow,
  ForecastResponse,
  ModelsStatus,
  PredictResponse,
  PredictionHistoryRow,
  PriceHistoryPoint,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

class ApiRequestError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

async function request<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url.searchParams.set(key, String(value));
    }
  }

  const res = await fetch(url.toString());
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON -- fall back to statusText
    }
    throw new ApiRequestError(detail, res.status);
  }
  return res.json();
}

export async function fetchTickers(): Promise<string[]> {
  return request<string[]>("/tickers");
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  return request<ModelsStatus>("/models");
}

export async function fetchPriceHistory(ticker: string, days = 90): Promise<PriceHistoryPoint[]> {
  return request<PriceHistoryPoint[]>(`/history/${ticker}`, { days });
}

export async function fetchPrediction(ticker: string, modelType: string): Promise<PredictResponse> {
  return request<PredictResponse>(`/predict/${ticker}`, { model_type: modelType });
}

export async function fetchForecast(
  ticker: string,
  modelType: string,
  horizonDays = 10
): Promise<ForecastResponse> {
  return request<ForecastResponse>(`/forecast/${ticker}`, {
    model_type: modelType,
    horizon_days: horizonDays,
  });
}

export async function fetchPredictionHistory(
  ticker: string,
  limit = 20
): Promise<PredictionHistoryRow[]> {
  return request<PredictionHistoryRow[]>(`/predictions/${ticker}`, { limit });
}

export async function fetchDriftStatus(ticker: string): Promise<DriftCheckRow[]> {
  return request<DriftCheckRow[]>(`/drift/${ticker}`);
}

export { ApiRequestError };
