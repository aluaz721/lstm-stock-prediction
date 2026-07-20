export type ModelType = "lstm" | "qlstm";

export interface ModelStatusEntry {
  registered_name: string;
  production_version: number | null;
}

export type ModelsStatus = Record<ModelType, ModelStatusEntry>;

export interface PriceHistoryPoint {
  date: string;
  close: number;
}

export interface PredictResponse {
  ticker: string;
  model_type: ModelType;
  model_version: number;
  predicted_close: number;
  last_known_close: number;
  prediction_for_date: string;
  generated_at: string;
}

export interface ForecastPoint {
  date: string;
  predicted_close: number;
}

export interface ForecastResponse {
  ticker: string;
  model_type: ModelType;
  model_version: number;
  forecast: ForecastPoint[];
}

export interface PredictionHistoryRow {
  model_type: ModelType;
  model_version: number;
  predicted_close: number;
  last_known_close: number;
  prediction_for_date: string;
  created_at: string;
}

export interface DriftCheckRow {
  feature_name: string;
  psi_score: number;
  threshold: number;
  is_drifted: boolean;
  checked_at: string;
}

export interface ApiError {
  message: string;
}
