"""
Serving layer: exposes prediction/forecast endpoints backed by whichever
model version is currently aliased 'production' in the MLflow registry.

Model/scaler loading and raw data fetching are injected as FastAPI
dependencies (get_model_provider, get_data_provider) rather than called
directly in route handlers. This is the same pattern used throughout this
project (forecast_horizon takes a predict_fn rather than a torch model
directly) -- it lets route logic (validation, DB writes, response shape,
error handling) be tested with stub providers, without needing a working
torch install or network access to Yahoo Finance. See
tests/test_api.py, which does exactly this and runs for real in this
sandbox.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.data.fetch import TICKERS, fetch_ticker_history
from src.data.features import FEATURE_COLUMNS, add_technical_indicators
from src.inference import predict_next_close, forecast_horizon, make_pyfunc_predict_fn, next_business_day
from src.monitoring.drift import compute_psi, is_drifted, DEFAULT_THRESHOLD
from src.storage.database import get_db, init_db
from src.storage import crud

SEQ_LEN = 10
MODEL_TYPES = ["lstm", "qlstm"]
DRIFT_CHECK_WINDOW_DAYS = 30  # "current" sample size for PSI comparison

# Creates the monitoring schema's tables if they don't exist yet -- see
# init_db's docstring for why this runs here AND in reference_capture.py
# rather than assuming one has already run before the other.
init_db()

app = FastAPI(title="Quantum Stock MLOps API", version="0.1.0")

# Local-only CORS config: the dashboard runs on a different port (5173)
# than the API (8000), which browsers treat as a different origin even
# though both are on your own machine. This does NOT need to change for
# a live deployment to keep working -- it just needs the deployed
# frontend's real origin added to this list at that point.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dependency providers -------------------------------------------------
# ModelProvider returns (predict_fn, scalers, model_version) for a given
# model_type. DataProvider returns raw OHLCV history for a given ticker.
ModelProvider = Callable[[str], tuple]
DataProvider = Callable[[str], object]


def _real_model_provider(model_type: str):
    from src.registry import load_production_model, load_production_scalers, get_production_version

    registered_name = f"stock-{model_type}"
    version = get_production_version(registered_name)
    if version is None:
        raise HTTPException(
            status_code=503,
            detail=f"No production model set for '{registered_name}' yet. "
            f"Train and promote one first (see src/promote_model.py).",
        )
    model = load_production_model(registered_name)
    scalers = load_production_scalers(registered_name)
    predict_fn = make_pyfunc_predict_fn(model)
    return predict_fn, scalers, version


def _real_data_provider(ticker: str):
    return fetch_ticker_history(ticker, start="2023-01-01")


def get_model_provider() -> ModelProvider:
    return _real_model_provider


def get_data_provider() -> DataProvider:
    return _real_data_provider


def verify_admin_token(authorization: str | None = Header(default=None)) -> None:
    """
    Simple bearer-token check for admin endpoints -- deliberately simple
    (a single shared token via env var) rather than full auth/RBAC, since
    this is called by one thing: your own scheduled job, not end users.
    Swap for something more robust before this is a multi-operator system.
    """
    expected = os.environ.get("ADMIN_API_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_API_TOKEN not configured on the server")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


# --- Routes -----------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    from src.registry import get_production_version

    return {
        model_type: {"registered_name": f"stock-{model_type}", "production_version": get_production_version(f"stock-{model_type}")}
        for model_type in MODEL_TYPES
    }


def _validate_ticker_and_model(ticker: str, model_type: str):
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'. Known: {TICKERS}")
    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown model_type '{model_type}'. Known: {MODEL_TYPES}")


@app.get("/tickers")
def list_tickers():
    return TICKERS


@app.get("/history/{ticker}")
def price_history(
    ticker: str,
    days: int = 90,
    data_provider: DataProvider = Depends(get_data_provider),
):
    """
    Actual historical Close prices -- separate from /predict and
    /forecast, which are forward-looking. The dashboard needs this to
    chart predictions in context against real price history, not just as
    isolated numbers.
    """
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'. Known: {TICKERS}")
    if not (1 <= days <= 365):
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")

    history = data_provider(ticker)
    recent = history.tail(days)
    return [
        {"date": date.isoformat(), "close": float(row["Close"])}
        for date, row in recent.iterrows()
    ]


@app.get("/predict/{ticker}")
def predict(
    ticker: str,
    model_type: str,
    db: Session = Depends(get_db),
    model_provider: ModelProvider = Depends(get_model_provider),
    data_provider: DataProvider = Depends(get_data_provider),
):
    _validate_ticker_and_model(ticker, model_type)

    predict_fn, scalers, version = model_provider(model_type)
    if ticker not in scalers:
        raise HTTPException(status_code=500, detail=f"No scaler found for '{ticker}' in the production model's run")

    history = data_provider(ticker)
    predicted_close = predict_next_close(predict_fn, history, scalers[ticker], seq_len=SEQ_LEN)
    last_known_close = float(history["Close"].iloc[-1])
    last_date = history.index[-1]

    prediction_for_date = next_business_day(last_date)

    crud.log_prediction(
        db,
        ticker=ticker,
        model_type=model_type,
        model_version=version,
        predicted_close=predicted_close,
        prediction_for_date=prediction_for_date.to_pydatetime(),
        last_known_close=last_known_close,
    )

    return {
        "ticker": ticker,
        "model_type": model_type,
        "model_version": version,
        "predicted_close": predicted_close,
        "last_known_close": last_known_close,
        "prediction_for_date": prediction_for_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/forecast/{ticker}")
def forecast(
    ticker: str,
    model_type: str,
    horizon_days: int = 10,
    model_provider: ModelProvider = Depends(get_model_provider),
    data_provider: DataProvider = Depends(get_data_provider),
):
    _validate_ticker_and_model(ticker, model_type)
    if not (1 <= horizon_days <= 30):
        raise HTTPException(status_code=400, detail="horizon_days must be between 1 and 30")

    predict_fn, scalers, version = model_provider(model_type)
    if ticker not in scalers:
        raise HTTPException(status_code=500, detail=f"No scaler found for '{ticker}' in the production model's run")

    history = data_provider(ticker)
    forecast_df = forecast_horizon(predict_fn, history, scalers[ticker], seq_len=SEQ_LEN, horizon_days=horizon_days)

    return {
        "ticker": ticker,
        "model_type": model_type,
        "model_version": version,
        "forecast": [
            {"date": date.isoformat(), "predicted_close": float(row["predicted_close"])}
            for date, row in forecast_df.iterrows()
        ],
    }


@app.get("/predictions/{ticker}")
def prediction_history(ticker: str, limit: int = 50, db: Session = Depends(get_db)):
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'. Known: {TICKERS}")

    rows = crud.get_recent_predictions(db, ticker=ticker, limit=limit)
    return [
        {
            "model_type": r.model_type,
            "model_version": r.model_version,
            "predicted_close": r.predicted_close,
            "last_known_close": r.last_known_close,
            "prediction_for_date": r.prediction_for_date.isoformat(),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.post("/admin/refresh-and-check-drift")
def refresh_and_check_drift(
    _: None = Depends(verify_admin_token),
    db: Session = Depends(get_db),
    data_provider: DataProvider = Depends(get_data_provider),
):
    """
    For every ticker, pulls the most recent DRIFT_CHECK_WINDOW_DAYS of
    feature values and compares each feature against its stored reference
    distribution (captured at training time -- see
    src/monitoring/reference_capture.py) via PSI. Skips (ticker, feature)
    pairs with no reference on file yet, rather than failing the whole
    check -- a partially-trained universe (e.g. only some tickers'
    references captured so far) is a normal, expected state, not an error.

    Meant to be called by a scheduled job (see scripts/refresh_and_check_drift.py)
    -- see that script's docstring for why this can't yet be a GitHub
    Actions cron job for a local-only deployment.
    """
    results = {}
    for ticker in TICKERS:
        try:
            history = data_provider(ticker)
        except Exception as e:
            results[ticker] = {"error": f"failed to fetch data: {e}"}
            continue

        featured = add_technical_indicators(history).dropna(subset=FEATURE_COLUMNS)
        if len(featured) < DRIFT_CHECK_WINDOW_DAYS:
            results[ticker] = {"error": f"insufficient recent history ({len(featured)} rows)"}
            continue

        current_window = featured.tail(DRIFT_CHECK_WINDOW_DAYS)
        ticker_results = {}
        for feature in FEATURE_COLUMNS:
            reference = crud.get_reference_distribution(db, ticker=ticker, feature_name=feature)
            if reference is None:
                ticker_results[feature] = {"status": "no_reference_captured"}
                continue

            psi_score = compute_psi(
                reference.bin_edges, reference.reference_proportions, current_window[feature].values
            )
            crud.log_drift_check(db, ticker=ticker, feature_name=feature, psi_score=psi_score, threshold=DEFAULT_THRESHOLD)
            ticker_results[feature] = {
                "psi_score": psi_score,
                "is_drifted": is_drifted(psi_score, DEFAULT_THRESHOLD),
            }
        results[ticker] = ticker_results

    return {"checked_at": datetime.now(timezone.utc).isoformat(), "results": results}


@app.get("/drift/{ticker}")
def drift_status(ticker: str, db: Session = Depends(get_db)):
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'. Known: {TICKERS}")

    checks = crud.get_latest_drift_status(db, ticker=ticker)
    return [
        {
            "feature_name": c.feature_name,
            "psi_score": c.psi_score,
            "threshold": c.threshold,
            "is_drifted": c.is_drifted,
            "checked_at": c.checked_at.isoformat(),
        }
        for c in checks
    ]
