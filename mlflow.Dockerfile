FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow>=2.15 psycopg2-binary

# MLflow >=3.5.0 validates the Host header on every request (CVE-2025-14279
# fix, DNS-rebinding protection) and rejects anything not explicitly
# allowed -- the default allowlist only covers localhost patterns. Inside
# docker-compose, the backend reaches this container via the Docker
# service name "mlflow" (e.g. Host: mlflow:5000), which isn't localhost,
# so it gets rejected without this flag. Also allow localhost:* for the
# case of connecting directly to the host-mapped port (e.g. from
# train.py run outside Docker, or opening the UI in a browser via
# localhost:5000).
#
# --artifacts-destination (NOT --default-artifact-root) is equally
# important, and easy to get backwards: --default-artifact-root pointed
# at a raw local path (e.g. "/mlflow-artifacts") makes every experiment's
# artifact_location a literal filesystem path, which requires whoever
# LOADS a model (the backend container) to have direct filesystem access
# to that exact path -- broken here, since backend and mlflow are
# separate containers with separate filesystems. --artifacts-destination
# instead tells the server where to physically store artifacts WHILE
# routing all upload/download/list requests through the tracking
# server's HTTP API (the "mlflow-artifacts:/" proxy scheme, on by
# default via --serve-artifacts) -- verified end to end: real PUT/GET
# traffic through /api/2.0/mlflow-artifacts/artifacts/... for both
# logging and loading a model, no shared volume required between
# backend and mlflow at all.
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://ccr_user:ccr_password@postgres:5432/mlflow", \
     "--artifacts-destination", "/mlflow-artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow,mlflow:*,localhost,localhost:*"]
