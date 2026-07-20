"""
Single source of truth for turning arch_params (a plain dict) into an
instantiated model. Used by train.py's CLI path and by the registry's
pyfunc load_context -- previously train.py had its own
argparse.Namespace-keyed version of this logic; centralizing it here means
a new model type only needs to be added in one place.
"""
from src.models.lstm import StockLSTM
from src.models.qlstm import StockQLSTM


def build_model(arch_params: dict):
    model_type = arch_params["model_type"]
    if model_type == "lstm":
        return StockLSTM(
            num_features=arch_params["num_features"],
            hidden_size=arch_params["hidden_size"],
        )
    elif model_type == "qlstm":
        return StockQLSTM(
            num_features=arch_params["num_features"],
            hidden_size=arch_params["hidden_size"],
            n_qubits=arch_params["n_qubits"],
            n_qlayers=arch_params["n_qlayers"],
        )
    raise ValueError(f"Unknown model type: {model_type}")
