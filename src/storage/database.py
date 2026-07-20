"""
Database engine + session management for the monitoring/serving layer.

Note this is separate from MLflow's own backend store -- MLflow tracks
experiments, runs, params, and metrics; this database tracks the things
MLflow has no concept of: a log of every live prediction served, drift
check results over time, and the reference feature distributions drift
gets compared against. Both point at the same RDS Postgres instance in
production, but they own different tables and different concerns.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get(
    "MONITORING_DATABASE_URL",
    "postgresql+psycopg2://ccr_user:ccr_password@localhost:5432/quantum_stock_mlops",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a session, always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Creates all tables if they don't already exist. create_all is
    naturally idempotent (checks for each table's existence before
    creating it), so this is safe to call on every process startup rather
    than needing a separate one-time migration step -- appropriate while
    the schema is still moving; switch to Alembic once it stabilizes.

    Called from every entry point that touches this database (the API at
    startup, and src/monitoring/reference_capture.py before training
    writes reference distributions) rather than relying on the API having
    started first -- training should work standalone against a fresh
    Postgres volume, not depend on start-up ordering with another service.
    """
    Base.metadata.create_all(bind=engine)
