FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow>=2.15 psycopg2-binary

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://ccr_user:ccr_password@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlflow-artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
