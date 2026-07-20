-- Runs automatically on first container startup (docker-entrypoint-initdb.d
-- convention). POSTGRES_DB env var already creates quantum_stock_mlops;
-- this creates the second database MLflow's tracking server uses for its
-- own backend store (runs, params, metrics, registered models) -- kept
-- separate from the monitoring schema (prediction_log, drift_check, etc.)
-- since they're different concerns owned by different code (MLflow itself
-- owns this one; src/storage/models.py owns the other).
CREATE DATABASE mlflow;
