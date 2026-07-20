"""
Real MLflow Model Registry integration -- an upgrade from the earlier
state_dict-artifact-only approach in train.py.

Why a custom PyFunc wrapper instead of mlflow.pytorch.log_model: PennyLane
QNode objects (inside StockQLSTM) aren't reliably picklable, which is
exactly the problem mlflow.pytorch.log_model's automatic serialization
would hit. StockModelWrapper sidesteps this by never pickling the QNode at
all -- load_context() reconstructs a fresh nn.Module from arch_params.json
and loads state_dict.pt into it, the same way you'd load any PyTorch
checkpoint. This gets us a real registered model (versioned, loadable via
`models:/name@production`) while keeping the same safe artifact format
already used for both LSTM and QLSTM.

Uses the CURRENT MLflow Model Registry API (aliases), not the classic
Staging/Production "stages" concept -- `transition_model_version_stage`
has been formally deprecated since MLflow 2.9. `production` here is our
own naming convention for an alias, not an MLflow-reserved stage name.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

PRODUCTION_ALIAS = "production"


class StockModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch  # deferred: this class is only ever instantiated where torch exists
        from src.models.factory import build_model

        arch_params = json.loads(Path(context.artifacts["arch_params"]).read_text())
        self.model = build_model(arch_params)
        state_dict = torch.load(context.artifacts["state_dict"], map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._torch = torch

    def predict(self, context, model_input, params=None):
        """
        model_input: numpy array of shape (batch, seq_len, n_features), or
        a single (seq_len, n_features) window (auto-batched to size 1).
        Returns a numpy array of shape (batch,) -- scaled next-Close
        predictions, matching the PredictFn contract in src/inference.py.
        """
        import numpy as np

        if model_input.ndim == 2:
            model_input = model_input[None, :, :]

        with self._torch.no_grad():
            x = self._torch.tensor(model_input, dtype=self._torch.float32)
            out = self.model(x)
        return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def log_and_register_model(
    model,
    arch_params: dict,
    registered_model_name: str,
):
    """
    Call this from within an active MLflow run (after training). Logs the
    model via the PyFunc wrapper and registers it, returning the resulting
    ModelVersion. Does NOT promote it to production -- that's a separate,
    deliberate step (see promote_to_production), so a freshly trained
    model never silently starts serving live traffic.
    """
    import torch

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        state_dict_path = tmp_path / "state_dict.pt"
        arch_params_path = tmp_path / "arch_params.json"
        torch.save(model.state_dict(), state_dict_path)
        arch_params_path.write_text(json.dumps(arch_params, indent=2))

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=StockModelWrapper(),
            artifacts={
                "state_dict": str(state_dict_path),
                "arch_params": str(arch_params_path),
            },
            registered_model_name=registered_model_name,
        )
    return model_info


def promote_to_production(registered_model_name: str, version: int) -> None:
    """
    Points the 'production' alias at the given version. This is the
    single choke point that determines what the serving layer loads --
    see src/inference.py's future FastAPI integration.
    """
    client = MlflowClient()
    client.set_registered_model_alias(registered_model_name, PRODUCTION_ALIAS, version)


def get_production_version(registered_model_name: str) -> int | None:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(registered_model_name, PRODUCTION_ALIAS)
        return int(mv.version)
    except mlflow.exceptions.MlflowException:
        return None


def load_production_model(registered_model_name: str):
    """Loads the model currently aliased 'production'. Raises if none is set."""
    return mlflow.pyfunc.load_model(f"models:/{registered_model_name}@{PRODUCTION_ALIAS}")


def load_production_scalers(registered_model_name: str) -> dict:
    """
    Loads the per-ticker StandardScalers logged alongside whichever
    training run produced the CURRENT production model version -- not
    just any run's scalers. Scalers are logged to the run's artifact
    store (see train.py's log_scaler_artifacts), not to the registered
    model itself, so we look up the model version's run_id first and
    pull scalers from that specific run.
    """
    import tempfile
    from pathlib import Path

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(registered_model_name, PRODUCTION_ALIAS)
    run_id = model_version.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="scalers", dst_path=tmp_dir
        )
        scalers = {}
        for npz_path in Path(local_dir).glob("scaler_*.npz"):
            ticker = npz_path.stem.replace("scaler_", "")
            data = np.load(npz_path)
            scaler = StandardScaler()
            scaler.mean_ = data["mean"]
            scaler.scale_ = data["scale"]
            scalers[ticker] = scaler
    return scalers
