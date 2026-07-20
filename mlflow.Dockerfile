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
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://ccr_user:ccr_password@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlflow-artifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow,mlflow:*,localhost,localhost:*"]
