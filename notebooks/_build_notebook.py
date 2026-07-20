"""
Builds notebooks/demo.ipynb programmatically via nbformat, rather than
hand-writing the JSON, so the notebook's structure can't get subtly
malformed. Run this script to regenerate the notebook if cells need
changing; edit notebook cells directly afterward, don't hand-edit this
generator for small text tweaks.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# Quantum Stock MLOps -- Demo Notebook

Trains the classical **LSTM** and the hybrid **QLSTM** on the 10-ticker
universe, tracks both runs in MLflow, then visualizes:

1. Test-set predictions vs. actual prices
2. A multi-day-ahead forecast
3. A single next-day prediction

**Note on the recursive multi-day forecast:** predicting day *t+2* requires
technical-indicator features for day *t+1*, which is itself a prediction --
there's no real OHLCV for it yet. We handle this by synthesizing a full
OHLCV row from the predicted Close (Open=High=Low=Close=prediction,
Volume=trailing mean) and recomputing indicators over the extended series.
Uncertainty compounds with each step forecasted; treat the far end of any
multi-day forecast as much lower-confidence than the near end.
""")

code("""\
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))  # so `src` is importable from notebooks/

import mlflow
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data.fetch import fetch_universe, TICKERS
from src.data.features import build_features_for_universe, FEATURE_COLUMNS
from src.data.dataset import (
    chronological_split,
    fit_scalers_per_ticker,
    transform_with_scalers,
    build_sequences,
)
from src.models.lstm import StockLSTM
from src.models.qlstm import StockQLSTM
from src.train import run_training_loop, log_scaler_artifacts
from src.registry import log_and_register_model, promote_to_production
from src.inference import make_torch_predict_fn, predict_next_close, forecast_horizon

SEQ_LEN = 10
TRAIN_FRAC = 0.65

mlflow.set_experiment("quantum-stock-mlops")
""")

md("## 1. Load and prepare data\n\nRequires normal internet access to reach Yahoo Finance (this won't work from a network-restricted sandbox).")

code("""\
universe = fetch_universe(TICKERS, start="2015-01-01")
featured = build_features_for_universe(universe)
split = chronological_split(featured, train_frac=TRAIN_FRAC)
scalers = fit_scalers_per_ticker(split.train)

train_scaled = transform_with_scalers(split.train, scalers)
test_scaled = transform_with_scalers(split.test, scalers)

X_train, y_train, _, _ = build_sequences(train_scaled, seq_len=SEQ_LEN)
X_test, y_test, tickers_test, dates_test = build_sequences(test_scaled, seq_len=SEQ_LEN)

print(f"train sequences: {X_train.shape}, test sequences: {X_test.shape}")
print(f"split date: {split.split_date}")
""")

code("""\
from torch.utils.data import DataLoader, TensorDataset

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
""")

md("## 2. Train the classical LSTM")

code("""\
lstm_model = StockLSTM(num_features=len(FEATURE_COLUMNS), hidden_size=32)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
lstm_arch_params = {"model_type": "lstm", "num_features": len(FEATURE_COLUMNS), "hidden_size": 32}

with mlflow.start_run(run_name="lstm-32h-notebook"):
    mlflow.log_params({
        "seq_len": SEQ_LEN, "lr": 1e-3, "train_frac": TRAIN_FRAC,
        "split_date": str(split.split_date), **lstm_arch_params,
    })
    lstm_history = run_training_loop(
        lstm_model, train_loader, test_loader, lstm_optimizer, epochs=30
    )
    log_scaler_artifacts(scalers)
    lstm_model_info = log_and_register_model(lstm_model, lstm_arch_params, "stock-lstm")
    print(f"Registered stock-lstm version {lstm_model_info.registered_model_version}")
""")

code("""\
lstm_epochs = [h["epoch"] for h in lstm_history]
lstm_train_mse = [h["train_mse"] for h in lstm_history]
lstm_test_mse = [h["test_mse"] for h in lstm_history]

plt.figure(figsize=(8, 4))
plt.plot(lstm_epochs, lstm_train_mse, label="train MSE")
plt.plot(lstm_epochs, lstm_test_mse, label="test MSE")
plt.xlabel("epoch"); plt.ylabel("MSE (scaled Close)"); plt.title("LSTM training curve")
plt.legend(); plt.tight_layout(); plt.show()
""")

md("## 3. Train the QLSTM\n\nThe quantum circuit simulation makes each epoch noticeably slower than the classical LSTM -- fewer epochs by default for that reason.")

code("""\
qlstm_model = StockQLSTM(num_features=len(FEATURE_COLUMNS), hidden_size=8, n_qubits=4, n_qlayers=1)
qlstm_optimizer = torch.optim.Adam(qlstm_model.parameters(), lr=1e-2)
qlstm_arch_params = {
    "model_type": "qlstm", "num_features": len(FEATURE_COLUMNS),
    "hidden_size": 8, "n_qubits": 4, "n_qlayers": 1,
}

with mlflow.start_run(run_name="qlstm-4q-notebook"):
    mlflow.log_params({
        "seq_len": SEQ_LEN, "lr": 1e-2, "train_frac": TRAIN_FRAC,
        "split_date": str(split.split_date), **qlstm_arch_params,
    })
    qlstm_history = run_training_loop(
        qlstm_model, train_loader, test_loader, qlstm_optimizer, epochs=15
    )
    log_scaler_artifacts(scalers)
    qlstm_model_info = log_and_register_model(qlstm_model, qlstm_arch_params, "stock-qlstm")
    print(f"Registered stock-qlstm version {qlstm_model_info.registered_model_version}")
""")

code("""\
qlstm_epochs = [h["epoch"] for h in qlstm_history]
qlstm_train_mse = [h["train_mse"] for h in qlstm_history]
qlstm_test_mse = [h["test_mse"] for h in qlstm_history]

plt.figure(figsize=(8, 4))
plt.plot(qlstm_epochs, qlstm_train_mse, label="train MSE")
plt.plot(qlstm_epochs, qlstm_test_mse, label="test MSE")
plt.xlabel("epoch"); plt.ylabel("MSE (scaled Close)"); plt.title("QLSTM training curve")
plt.legend(); plt.tight_layout(); plt.show()
""")

md("## 4. Promote both models to production\n\nRegistering a model does NOT make it start serving -- that's a deliberate separate step, so a freshly trained run never silently replaces what's currently live. Promoting here is reasonable since this is a first-time setup with no production model to compare against yet; in practice you'd check test-set metrics (or drift/reproducibility checks) before promoting a real candidate.")

code("""\
promote_to_production("stock-lstm", int(lstm_model_info.registered_model_version))
promote_to_production("stock-qlstm", int(qlstm_model_info.registered_model_version))
print("Both models promoted to production -- the FastAPI serving layer can now load them.")
""")

md("## 5. Test-set predictions vs. actual prices\n\nBatch inference on the held-out test set, inverse-transformed back to real price per ticker.")

code("""\
def batch_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

lstm_preds_scaled = batch_predict(lstm_model, X_test)
qlstm_preds_scaled = batch_predict(qlstm_model, X_test)

close_idx = FEATURE_COLUMNS.index("Close")

def inverse_close_batch(scaled_values, tickers, scalers):
    out = np.zeros_like(scaled_values)
    for i, (val, ticker) in enumerate(zip(scaled_values, tickers)):
        out[i] = val * scalers[ticker].scale_[close_idx] + scalers[ticker].mean_[close_idx]
    return out

lstm_preds = inverse_close_batch(lstm_preds_scaled, tickers_test, scalers)
qlstm_preds = inverse_close_batch(qlstm_preds_scaled, tickers_test, scalers)
actual = inverse_close_batch(y_test, tickers_test, scalers)

results_df = pd.DataFrame({
    "date": dates_test, "ticker": tickers_test,
    "actual": actual, "lstm_pred": lstm_preds, "qlstm_pred": qlstm_preds,
})
results_df.head()
""")

code("""\
def plot_ticker_predictions(ticker):
    sub = results_df[results_df["ticker"] == ticker].sort_values("date")
    plt.figure(figsize=(10, 4))
    plt.plot(sub["date"], sub["actual"], label="actual", linewidth=2)
    plt.plot(sub["date"], sub["lstm_pred"], label="LSTM", alpha=0.8)
    plt.plot(sub["date"], sub["qlstm_pred"], label="QLSTM", alpha=0.8)
    plt.title(f"{ticker}: test-set predictions vs. actual")
    plt.xlabel("date"); plt.ylabel("price ($)")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

plot_ticker_predictions("NVDA")
""")

code("""\
plot_ticker_predictions("AAPL")
""")

code("""\
from sklearn.metrics import mean_squared_error, mean_absolute_error

def directional_accuracy(actual, pred, prev_actual):
    actual_dir = np.sign(actual - prev_actual)
    pred_dir = np.sign(pred - prev_actual)
    return float(np.mean(actual_dir == pred_dir))

rows = []
for ticker in results_df["ticker"].unique():
    sub = results_df[results_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    prev_actual = sub["actual"].shift(1).bfill()
    rows.append({
        "ticker": ticker,
        "lstm_rmse": mean_squared_error(sub["actual"], sub["lstm_pred"]) ** 0.5,
        "qlstm_rmse": mean_squared_error(sub["actual"], sub["qlstm_pred"]) ** 0.5,
        "lstm_mae": mean_absolute_error(sub["actual"], sub["lstm_pred"]),
        "qlstm_mae": mean_absolute_error(sub["actual"], sub["qlstm_pred"]),
        "lstm_directional_acc": directional_accuracy(sub["actual"], sub["lstm_pred"], prev_actual),
        "qlstm_directional_acc": directional_accuracy(sub["actual"], sub["qlstm_pred"], prev_actual),
    })

metrics_df = pd.DataFrame(rows).set_index("ticker")
metrics_df
""")

md("## 6. Multi-day-ahead forecast\n\nRecursive forecast for one ticker -- see the compounding-uncertainty note in the intro cell.")

code("""\
FORECAST_TICKER = "NVDA"
HORIZON_DAYS = 10

ticker_history = universe[universe["ticker"] == FORECAST_TICKER].sort_index()
ticker_scaler = scalers[FORECAST_TICKER]

lstm_predict_fn = make_torch_predict_fn(lstm_model)
qlstm_predict_fn = make_torch_predict_fn(qlstm_model)

lstm_forecast = forecast_horizon(lstm_predict_fn, ticker_history, ticker_scaler, SEQ_LEN, HORIZON_DAYS)
qlstm_forecast = forecast_horizon(qlstm_predict_fn, ticker_history, ticker_scaler, SEQ_LEN, HORIZON_DAYS)

lstm_forecast.head()
""")

code("""\
recent_actual = ticker_history["Close"].tail(60)

plt.figure(figsize=(10, 5))
plt.plot(recent_actual.index, recent_actual.values, label="recent actual", linewidth=2, color="black")
plt.plot(lstm_forecast.index, lstm_forecast["predicted_close"], label="LSTM forecast", linestyle="--", marker="o")
plt.plot(qlstm_forecast.index, qlstm_forecast["predicted_close"], label="QLSTM forecast", linestyle="--", marker="o")
plt.axvline(recent_actual.index[-1], color="gray", linestyle=":", label="forecast start")
plt.title(f"{FORECAST_TICKER}: {HORIZON_DAYS}-day forecast")
plt.xlabel("date"); plt.ylabel("price ($)")
plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
""")

md("## 7. Single next-day prediction")

code("""\
lstm_next = predict_next_close(lstm_predict_fn, ticker_history, ticker_scaler, SEQ_LEN)
qlstm_next = predict_next_close(qlstm_predict_fn, ticker_history, ticker_scaler, SEQ_LEN)
last_close = ticker_history["Close"].iloc[-1]

print(f"{FORECAST_TICKER} -- last close: ${last_close:.2f}")
print(f"  LSTM next-day prediction:  ${lstm_next:.2f}  ({'+' if lstm_next >= last_close else ''}{lstm_next - last_close:.2f})")
print(f"  QLSTM next-day prediction: ${qlstm_next:.2f}  ({'+' if qlstm_next >= last_close else ''}{qlstm_next - last_close:.2f})")
""")

md("""\
## Next steps

Built so far: the data pipeline, LSTM + QLSTM, MLflow experiment tracking
and Model Registry (this notebook just registered and promoted both models
to production), the FastAPI serving layer (`src/api/main.py`), and
Postgres-backed prediction logging. Try it once this notebook has run:

```bash
uvicorn src.api.main:app --reload
curl "http://localhost:8000/predict/NVDA?model_type=lstm"
```

Still to build: PSI-based drift detection, the scheduled GitHub Actions job
that fetches new data and checks for it, the deploy step in CI, and the
React dashboard for live predictions + monitoring -- all designed to read
from the same MLflow runs and Postgres tables this notebook already
produces, so nothing here gets thrown away once those exist.
""")

nb["cells"] = cells
nbf.write(nb, "/home/claude/quantum-stock-mlops/notebooks/demo.ipynb")
print("notebook written")
