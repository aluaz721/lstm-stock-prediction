"""
Trains StockLSTM or StockQLSTM on the pooled 10-ticker universe, tracking
everything through MLflow.

Model artifacts are logged as state_dict + hyperparameters rather than via
mlflow.pytorch.log_model's automatic pickling: QLSTM's PennyLane QNode
objects (and their underlying simulator devices) are not reliably
picklable, so relying on automatic full-object serialization would work
for StockLSTM but silently be fragile for StockQLSTM. Logging state_dict +
the params needed to reconstruct the architecture is the standard MLflow
pattern for custom architectures anyway, and treats both models the same
way -- one code path, not two.

Usage:
    python -m src.train --model lstm --epochs 30
    python -m src.train --model qlstm --epochs 15 --n-qubits 4
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.fetch import fetch_universe, TICKERS
from src.data.features import build_features_for_universe, FEATURE_COLUMNS
from src.data.dataset import (
    chronological_split,
    fit_scalers_per_ticker,
    transform_with_scalers,
    build_sequences,
)
from src.models.factory import build_model
from src.registry import log_and_register_model, promote_to_production
from src.monitoring.reference_capture import capture_reference_distributions


def build_datasets(seq_len: int, train_frac: float, start: str):
    universe = fetch_universe(TICKERS, start=start)
    featured = build_features_for_universe(universe)
    split = chronological_split(featured, train_frac=train_frac)
    scalers = fit_scalers_per_ticker(split.train)

    train_scaled = transform_with_scalers(split.train, scalers)
    test_scaled = transform_with_scalers(split.test, scalers)

    X_train, y_train, _, _, _ = build_sequences(train_scaled, seq_len=seq_len)
    X_test, y_test, tickers_test, dates_test, last_close_test = build_sequences(test_scaled, seq_len=seq_len)

    return (
        (X_train, y_train),
        (X_test, y_test, tickers_test, dates_test, last_close_test),
        scalers,
        split.split_date,
        split.train,
    )


def evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb, reduction="sum")
            total_loss += loss.item()
            n += xb.shape[0]
    return total_loss / n


def run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    log_to_mlflow: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Runs the epoch loop and returns per-epoch history as a list of dicts,
    so callers (the CLI script, or a notebook) can plot loss curves
    without re-implementing the loop. Assumes an MLflow run is already
    active if log_to_mlflow=True -- this function doesn't open one itself,
    so it composes cleanly inside `with mlflow.start_run():` from the CLI
    or a notebook cell.
    """
    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
            n += xb.shape[0]

        train_mse = epoch_loss / n
        test_mse = evaluate(model, test_loader)
        history.append({"epoch": epoch, "train_mse": train_mse, "test_mse": test_mse})

        if log_to_mlflow:
            mlflow.log_metrics({"train_mse": train_mse, "test_mse": test_mse}, step=epoch)
        if verbose:
            print(f"epoch {epoch:03d}  train_mse={train_mse:.5f}  test_mse={test_mse:.5f}")

    return history


def log_model_artifacts(model: torch.nn.Module, arch_params: dict) -> None:
    """Logs state_dict + reconstruction params to the active MLflow run's 'model' path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        torch.save(model.state_dict(), tmp_path / "state_dict.pt")
        (tmp_path / "arch_params.json").write_text(json.dumps(arch_params, indent=2))
        mlflow.log_artifacts(str(tmp_path), artifact_path="model")


def log_scaler_artifacts(scalers: dict) -> None:
    """Logs per-ticker StandardScaler mean/scale to the active MLflow run's 'scalers' path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for ticker, scaler in scalers.items():
            np.savez(tmp_path / f"scaler_{ticker}.npz", mean=scaler.mean_, scale=scaler.scale_)
        mlflow.log_artifacts(str(tmp_path), artifact_path="scalers")


def train(args: argparse.Namespace):
    (
        (X_train, y_train),
        (X_test, y_test, _tickers_test, _dates_test, _last_close_test),
        scalers,
        split_date,
        raw_train_df,
    ) = build_datasets(seq_len=args.seq_len, train_frac=args.train_frac, start=args.start)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    arch_params = {
        "model_type": args.model,
        "num_features": len(FEATURE_COLUMNS),
        "hidden_size": args.hidden_size,
        **({"n_qubits": args.n_qubits, "n_qlayers": args.n_qlayers} if args.model == "qlstm" else {}),
    }
    model = build_model(arch_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    registered_model_name = args.registered_model_name or f"stock-{args.model}"

    mlflow.set_experiment("quantum-stock-mlops")
    with mlflow.start_run(run_name=f"{args.model}-{args.hidden_size}h"):
        mlflow.log_params(
            {
                "seq_len": args.seq_len,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "train_frac": args.train_frac,
                "split_date": str(split_date),
                "n_tickers": len(TICKERS),
                **arch_params,
            }
        )

        run_training_loop(model, train_loader, test_loader, optimizer, args.epochs)
        log_scaler_artifacts(scalers)

        model_info = log_and_register_model(model, arch_params, registered_model_name)
        version = int(model_info.registered_model_version)
        print(f"Registered {registered_model_name} version {version}")

        run_id = mlflow.active_run().info.run_id
        capture_reference_distributions(raw_train_df, run_id)
        print(f"Captured drift reference distributions for {raw_train_df['ticker'].nunique()} ticker(s)")

        if args.promote:
            promote_to_production(registered_model_name, version)
            print(f"Promoted {registered_model_name} v{version} to production")
        else:
            print(
                f"NOT promoted to production. To promote: "
                f"python -m src.promote_model --model-name {registered_model_name} --version {version}"
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "qlstm"], required=True)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-frac", type=float, default=0.65)
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-qlayers", type=int, default=1)
    p.add_argument(
        "--registered-model-name", type=str, default=None,
        help="Defaults to 'stock-{model}', e.g. 'stock-lstm'",
    )
    p.add_argument(
        "--promote", action="store_true",
        help="Immediately promote this run to production. Omit to register without promoting "
        "(the safer default -- see src/promote_model.py to promote after reviewing metrics).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
