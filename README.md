# Quantum Stock MLOps

Trains a classical LSTM (and a hybrid QLSTM, architecture built but not
yet trained -- see Status below) to predict next-day stock returns for a
10-ticker tech universe, tracked through MLflow's experiment tracking and
Model Registry, served through a FastAPI backend, with PSI-based drift
monitoring backed by Postgres. Runs entirely locally via `docker-compose`.

<figure>
  <img src="./docs/dashboard.png" alt="Screenshot of frontend dashboard.">
  <figcaption>Screenshot of interactive dashboard displaying predictions from
  the latest model version pushed to production.</figcaption>
</figure>

## Architecture

```
GitHub Actions (CI)          -- test + lint on every push, Docker build+publish on merge to main
        |
docker-compose (local)
        |
        +-- postgres          -- monitoring schema (predictions, drift checks, reference distributions)
        |                        + a second database as MLflow's own backend store
        +-- mlflow             -- tracking server + Model Registry (mlflow.Dockerfile)
        +-- backend            -- FastAPI serving layer (Dockerfile)
```

A locally-run script (`scripts/refresh_and_check_drift.py`) calls the
backend's admin endpoint to refresh data and check for drift -- see that
script's docstring for why this can't be a GitHub Actions scheduled job
yet (GitHub's runners can't reach `localhost` on your machine; that
becomes possible once this is deployed somewhere with a real URL).

## What's here

```
quantum-stock-mlops/
├── src/
│   ├── data/
│   │   ├── fetch.py                yfinance ingestion
│   │   ├── features.py             ratio/return-based technical indicators (see "Why returns" below)
│   │   └── dataset.py              chronological split, per-ticker scaling, sequence windowing
│   ├── models/
│   │   ├── lstm.py, qlstm.py       model architectures
│   │   └── factory.py              shared model instantiation (train.py + registry.py both use this)
│   ├── monitoring/
│   │   ├── drift.py                PSI (Population Stability Index) computation
│   │   └── reference_capture.py    captures reference distributions at training time
│   ├── storage/
│   │   ├── models.py               PredictionLog, DriftCheck, FeatureReferenceDistribution
│   │   ├── database.py, crud.py
│   ├── api/main.py                 FastAPI: /predict, /forecast, /drift, /admin/refresh-and-check-drift
│   ├── registry.py                 MLflow Model Registry (custom PyFunc wrapper, alias-based promotion)
│   ├── promote_model.py            CLI: promote a registered version to production
│   ├── train.py                    CLI training entrypoint
│   └── inference.py                predict_next_close, forecast_horizon
├── scripts/refresh_and_check_drift.py   local scheduled-job script (see Architecture above)
├── notebooks/demo.ipynb            trains, registers, promotes, visualizes -- run this first
├── tests/                          62 tests; all but test_models.py run without torch
├── Dockerfile, mlflow.Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml
```

## Setup

```bash
git clone <this repo> && cd quantum-stock-mlops
cp .env.example .env   # then edit if you want a non-default ADMIN_API_TOKEN
docker compose up --build
```

This starts Postgres (with both the monitoring schema's database and
MLflow's backend-store database, created automatically via
`docker/postgres-init/`), the MLflow tracking server on `localhost:5000`,
and the FastAPI backend on `localhost:8000`.

Check it's up: `curl localhost:8000/health` -> `{"status":"ok"}`. Browse
the MLflow UI at `localhost:5000`. Interactive API docs at
`localhost:8000/docs`.

## Train and promote a model

Requires normal internet access (to reach Yahoo Finance) and a working
local `torch` install -- see the CPU-wheel note below.

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export MLFLOW_TRACKING_URI=http://localhost:5000
export MONITORING_DATABASE_URL="postgresql+psycopg2://ccr_user:ccr_password@localhost:5432/quantum_stock_mlops"

python -m src.train --model lstm --epochs 30
# prints: "Registered stock-lstm version 1" and "NOT promoted to production. To promote: ..."

python -m src.promote_model --model-name stock-lstm --version 1
```

Or run `notebooks/demo.ipynb`, which does all of this (train, register,
promote, visualize) in one place.

## Use the API

```bash
curl "http://localhost:8000/predict/NVDA?model_type=lstm"
curl "http://localhost:8000/forecast/NVDA?model_type=lstm&horizon_days=10"
curl "http://localhost:8000/predictions/NVDA"          # prediction history
curl "http://localhost:8000/drift/NVDA"                # latest drift status
curl "http://localhost:8000/models"                    # what's currently in production
```

## Check for drift

```bash
export ADMIN_API_TOKEN=local-dev-token   # must match docker-compose's value
python scripts/refresh_and_check_drift.py
```

Fetches fresh data for every ticker, compares each feature against the
reference distribution captured at training time (PSI), logs results to
Postgres, and exits with code 2 if anything crossed the drift threshold
(useful for wiring into cron/alerting later).

## Why the target is next-day RETURN, not next-day price

This was a real bug, caught and fixed mid-project, worth documenting
rather than glossing over: predicting raw price *level* let the LSTM
achieve near-zero training loss almost instantly by learning "next Close
≈ last Close" -- trivially available since Close-derived values are
inputs at the window's last timestep, and day-to-day price levels barely
move relative to their scale. This is the single most common illusion in
"LSTM predicts stock prices" tutorials: charts that track the actual price
closely look impressive but reflect a lagged copy, not real forecasting.
Confirmed empirically (reproduced with synthetic data, see project
history) that this specifically breaks down when a stock's price level
drifts between train and test periods -- train/test target statistics
came back wildly mismatched (train mean≈0/std≈1 vs. test mean≈-0.54/
std≈2.71 in one reproduction). Switching to next-day log return as the
target, and to ratio/return-based input features (`close_to_ma21`,
`macd_pct`, `bb_percent_b`, etc., instead of raw dollar-denominated
indicators), fixed this directly -- train/test statistics came back
nearly identical once returns replaced levels. The API still returns
predicted *prices* to the caller (`last_close * exp(predicted_return)`);
only the model's internal training target changed.

**Known finding, not a bug:** baselining against a zero-return predictor
and a plain linear regression showed the LSTM's MSE isn't meaningfully
better than either. This is consistent with published literature on
short-horizon equity return prediction from technical indicators alone
(weak-form market efficiency) -- R² and directional accuracy are more
appropriate metrics than raw MSE for detecting a small-but-real effect
here (MSE is dominated by noise variance and can look identical between
"genuine small signal" and "no signal" -- see project history for a
worked demonstration). Richer features (correlated assets, sentiment,
Fourier trend decomposition) are a promising next step, intentionally
deferred in favor of finishing the MLOps pipeline first.

## What's actually been verified vs. what hasn't

- **Data pipeline, feature engineering, sequence windowing**: real pytest
  runs against synthetic data, including a hand-worked indexing check for
  the return-target's day-alignment (an easy off-by-one to get wrong).
- **QLSTM's quantum circuit**: verified in isolation against real
  PennyLane, including a gradient-flow check.
- **PSI drift detection**: real pytest runs, including a regression test
  for a genuine bug caught during development (the initial implementation
  used literal `-inf`/`+inf` bin edges, which numpy happily produces but
  standard JSON -- and Postgres's `json` column type -- reject as invalid
  tokens; fixed with a large finite sentinel instead).
- **MLflow Model Registry mechanics** (register, alias, promote, load-by-
  alias): verified against a real local MLflow tracking store using a
  stub model, including confirming that registering a new version does
  NOT silently move the production alias -- a fresh training run should
  never auto-replace what's currently serving.
- **Storage layer and FastAPI routes** (including the admin/drift
  endpoints' auth): real integration tests against Postgres, and stub-
  provider tests exercising real route logic (validation, DB writes, error
  handling) without needing torch or network access.
- **NOT executed in the sandbox this was built in**: the actual LSTM/QLSTM
  training loop, and `docker compose up` itself. That sandbox can't get a
  working CPU-only `torch` install (the default PyPI wheel demands ~5GB of
  CUDA libraries even for CPU use) and doesn't have Docker available at
  all. The Dockerfiles and compose config follow patterns already verified
  working in an earlier project in this same environment (course
  recommender's docker-compose, which *was* tested), but you're the first
  to actually run this specific compose file end to end -- watch for it.

## Known gaps / next steps

- **QLSTM hasn't been trained yet** (deliberately deferred -- longer
  training time, and the classical LSTM was the priority for reaching a
  working MVP). `python -m src.train --model qlstm --epochs 15` should
  work today; it just hasn't been run.
- **No live deployment yet** -- everything above is local-only by design
  for now. When ready: Dockerfiles are already written, so deploying to
  Render/Fly.io/ECS Fargate is a hosting-target decision, not a rewrite.
- **No frontend/dashboard yet** -- deliberately last. The API already
  returns everything a dashboard would need (`/predictions/{ticker}` for
  live prediction history, `/drift/{ticker}` for monitoring).
- **Richer features** (correlated assets, sentiment, Fourier trend
  decomposition) intentionally deferred until after the pipeline works
  end to end.
