"""
Tests the API's routing, validation, DB writes, and response shapes using
stub model/data providers injected via FastAPI's dependency_overrides --
no torch, no MLflow registry, no network access to Yahoo Finance needed.
This exercises the real route logic end to end; only the "load an actual
trained model" and "fetch real market data" pieces are stubbed, and both
of those are already independently verified elsewhere (registry.py's
mechanics in the manual test scripts, fetch.py's contract, and
inference.py's forecasting logic all have their own real tests).
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from src.api.main import app, get_model_provider, get_data_provider
from src.data.features import FEATURE_COLUMNS, add_technical_indicators
from src.monitoring.drift import build_reference_distribution
from src.storage.database import engine, Base, get_db, SessionLocal
from src.storage import crud


@pytest.fixture(autouse=True)
def clean_tables():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def make_stub_history(n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    price = 130.0
    closes = [price]
    for _ in range(n_days - 1):
        price *= 1 + rng.normal(0, 0.01)
        closes.append(price)
    return pd.DataFrame(
        {
            "Open": [c * 0.999 for c in closes],
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, n_days),
        },
        index=dates,
    )


def make_stub_scaler(history_df):
    featured = add_technical_indicators(history_df).dropna(subset=FEATURE_COLUMNS)
    scaler = StandardScaler()
    scaler.fit(featured[FEATURE_COLUMNS].values)
    return scaler


@pytest.fixture
def stub_history():
    return make_stub_history()


@pytest.fixture
def client(stub_history):
    stub_scaler = make_stub_scaler(stub_history)

    def stub_model_provider(model_type: str):
        def stub_predict_fn(scaled_window: np.ndarray) -> float:
            return 0.1  # arbitrary fixed scaled prediction
        return stub_predict_fn, {"NVDA": stub_scaler}, 3  # pretend version 3 is in prod

    def stub_data_provider(ticker: str):
        return stub_history

    app.dependency_overrides[get_model_provider] = lambda: stub_model_provider
    app.dependency_overrides[get_data_provider] = lambda: stub_data_provider

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_ticker_and_model(client):
    resp = client.get("/predict/NVDA", params={"model_type": "lstm"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "NVDA"
    assert body["model_type"] == "lstm"
    assert body["model_version"] == 3
    assert isinstance(body["predicted_close"], float)
    assert isinstance(body["last_known_close"], float)
    assert "prediction_for_date" in body


def test_predict_unknown_ticker_returns_404(client):
    resp = client.get("/predict/NOT_A_TICKER", params={"model_type": "lstm"})
    assert resp.status_code == 404


def test_predict_unknown_model_type_returns_400(client):
    resp = client.get("/predict/NVDA", params={"model_type": "not_a_real_model"})
    assert resp.status_code == 400


def test_predict_writes_to_prediction_log(client):
    client.get("/predict/NVDA", params={"model_type": "lstm"})
    client.get("/predict/NVDA", params={"model_type": "qlstm"})

    resp = client.get("/predictions/NVDA")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert {row["model_type"] for row in body} == {"lstm", "qlstm"}


def test_predict_missing_scaler_for_ticker_returns_500(client, stub_history):
    def broken_model_provider(model_type: str):
        def stub_predict_fn(scaled_window):
            return 0.1
        return stub_predict_fn, {"SOME_OTHER_TICKER": None}, 1  # NVDA missing on purpose

    app.dependency_overrides[get_model_provider] = lambda: broken_model_provider
    resp = client.get("/predict/NVDA", params={"model_type": "lstm"})
    assert resp.status_code == 500


def test_forecast_returns_correct_horizon_length(client):
    resp = client.get("/forecast/NVDA", params={"model_type": "lstm", "horizon_days": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["forecast"]) == 5
    assert all("date" in row and "predicted_close" in row for row in body["forecast"])


def test_forecast_rejects_horizon_out_of_range(client):
    resp = client.get("/forecast/NVDA", params={"model_type": "lstm", "horizon_days": 100})
    assert resp.status_code == 400

    resp = client.get("/forecast/NVDA", params={"model_type": "lstm", "horizon_days": 0})
    assert resp.status_code == 400


def test_prediction_history_empty_before_any_predictions(client):
    resp = client.get("/predictions/NVDA")
    assert resp.status_code == 200
    assert resp.json() == []


def test_prediction_history_unknown_ticker_returns_404(client):
    resp = client.get("/predictions/NOT_A_TICKER")
    assert resp.status_code == 404


def test_list_tickers(client):
    resp = client.get("/tickers")
    assert resp.status_code == 200
    assert "NVDA" in resp.json()


def test_price_history_returns_requested_range(client):
    resp = client.get("/history/NVDA", params={"days": 30})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 30
    assert all("date" in row and "close" in row for row in body)
    # chronological order
    dates = [row["date"] for row in body]
    assert dates == sorted(dates)


def test_price_history_unknown_ticker_returns_404(client):
    resp = client.get("/history/NOT_A_TICKER")
    assert resp.status_code == 404


def test_price_history_rejects_out_of_range_days(client):
    resp = client.get("/history/NVDA", params={"days": 0})
    assert resp.status_code == 400
    resp = client.get("/history/NVDA", params={"days": 1000})
    assert resp.status_code == 400


def test_admin_endpoint_rejects_missing_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-token")
    resp = client.post("/admin/refresh-and-check-drift")
    assert resp.status_code == 401


def test_admin_endpoint_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-token")
    resp = client.post(
        "/admin/refresh-and-check-drift", headers={"Authorization": "Bearer wrong-token"}
    )
    assert resp.status_code == 401


def test_admin_endpoint_rejects_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    resp = client.post(
        "/admin/refresh-and-check-drift", headers={"Authorization": "Bearer anything"}
    )
    assert resp.status_code == 503


def test_admin_endpoint_with_valid_token_and_no_references(client, monkeypatch):
    """
    Before any model has been trained, no reference distributions exist
    yet -- the endpoint should report that clearly per feature rather
    than erroring out.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-token")
    resp = client.post(
        "/admin/refresh-and-check-drift", headers={"Authorization": "Bearer secret-token"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "NVDA" in body["results"]
    for feature_result in body["results"]["NVDA"].values():
        assert feature_result["status"] == "no_reference_captured"


def test_admin_endpoint_computes_and_logs_drift_when_reference_exists(client, stub_history, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-token")

    # seed a real reference distribution for NVDA/return_1d, built from
    # the same stub history the data provider will return as "current"
    featured = add_technical_indicators(stub_history).dropna(subset=FEATURE_COLUMNS)
    edges, proportions = build_reference_distribution(featured["return_1d"].values, n_bins=10)

    db = SessionLocal()
    crud.upsert_reference_distribution(
        db, ticker="NVDA", feature_name="return_1d",
        training_run_id="test-run", bin_edges=edges, reference_proportions=proportions,
    )
    db.close()

    resp = client.post(
        "/admin/refresh-and-check-drift", headers={"Authorization": "Bearer secret-token"}
    )
    assert resp.status_code == 200
    body = resp.json()
    nvda_return_result = body["results"]["NVDA"]["return_1d"]
    assert "psi_score" in nvda_return_result
    # same distribution compared against itself -- should NOT register as drifted
    assert nvda_return_result["is_drifted"] is False

    # and it should be logged, readable via the drift status endpoint
    status_resp = client.get("/drift/NVDA")
    assert status_resp.status_code == 200
    feature_names = {row["feature_name"] for row in status_resp.json()}
    assert "return_1d" in feature_names


def test_drift_status_empty_before_any_checks(client):
    resp = client.get("/drift/NVDA")
    assert resp.status_code == 200
    assert resp.json() == []


def test_drift_status_unknown_ticker_returns_404(client):
    resp = client.get("/drift/NOT_A_TICKER")
    assert resp.status_code == 404
