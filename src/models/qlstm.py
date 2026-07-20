"""
QLSTM: each of the LSTM's four internal gates (forget, input, cell-update,
output) is computed by a variational quantum circuit (VQC) instead of a
plain linear layer, following the architecture used in both of your
reference sources (Chen, Yoo & Fang 2020; the SoftServe_QLSTM repo).

The circuit structure (gate ansatz, angle embedding via Hadamard + arctan
RY/RZ rotations) is verified against pure PennyLane in
tests/test_qlstm_circuit.py -- see that file for the isolated,
runnable-without-torch check of the quantum piece.

One deliberate deviation from the SoftServe_QLSTM reference repo: that
implementation reuses a SINGLE classical output layer across all four
gates (`self.clayer_out`). We give each gate its own output layer instead
-- there's no reason to force the forget/input/update/output gates to
share a linear readout, and the wider QLSTM literature (including the
original Chen et al. 2020 paper this all traces back to) uses independent
per-gate output layers. This roughly quadruples that specific layer's
parameter count but keeps the gates from being artificially coupled.
"""
import torch
from torch import nn
import pennylane as qml


def _make_vqc_gate(n_qubits: int, n_qlayers: int, n_vrotations: int, wires: list[str]):
    """Builds one gate's VQC as a PennyLane QNode wrapped for PyTorch autograd."""
    dev = qml.device("default.qubit", wires=wires)

    def ansatz(params, wires_type):
        # Entangling ring.
        for i in range(1, 3):
            for j in range(n_qubits):
                target = j + i if j + i < n_qubits else j + i - n_qubits
                qml.CNOT(wires=[wires_type[j], wires_type[target]])
        # Variational rotations.
        for i in range(n_qubits):
            qml.RX(params[0][i], wires=wires_type[i])
            qml.RY(params[1][i], wires=wires_type[i])
            qml.RZ(params[2][i], wires=wires_type[i])

    def vqc(features, weights, wires_type):
        # Angle embedding: arctan squashes unbounded classical features
        # into a stable rotation-angle range before entangling.
        ry_params = [torch.arctan(f) for f in features]
        for i in range(n_qubits):
            qml.Hadamard(wires=wires_type[i])
            qml.RY(ry_params[i], wires=wires_type[i])
            qml.RZ(ry_params[i], wires=wires_type[i])
        qml.layer(ansatz, n_qlayers, weights, wires_type=wires_type)

    def circuit(inputs, weights):
        vqc(inputs, weights, wires)
        return [qml.expval(qml.PauliZ(wires=w)) for w in wires]

    qnode = qml.QNode(circuit, dev, interface="torch")
    weight_shapes = {"weights": (n_qlayers, n_vrotations, n_qubits)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)


class QLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int = 4,
        n_qlayers: int = 1,
        n_vrotations: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        concat_size = input_size + hidden_size

        wires = {
            gate: [f"{gate}_{i}" for i in range(n_qubits)]
            for gate in ("forget", "input", "update", "output")
        }

        # Shared classical "downsample to qubit count" input layer -- this
        # one IS shared across gates in both reference sources, and sharing
        # it is reasonable: it's just a dimensionality reduction step
        # before the quantum circuits, not part of the gating logic itself.
        self.clayer_in = nn.Linear(concat_size, n_qubits)

        self.vqc_forget = _make_vqc_gate(n_qubits, n_qlayers, n_vrotations, wires["forget"])
        self.vqc_input = _make_vqc_gate(n_qubits, n_qlayers, n_vrotations, wires["input"])
        self.vqc_update = _make_vqc_gate(n_qubits, n_qlayers, n_vrotations, wires["update"])
        self.vqc_output = _make_vqc_gate(n_qubits, n_qlayers, n_vrotations, wires["output"])

        # Independent output layer per gate (see module docstring).
        self.clayer_out_forget = nn.Linear(n_qubits, hidden_size)
        self.clayer_out_input = nn.Linear(n_qubits, hidden_size)
        self.clayer_out_update = nn.Linear(n_qubits, hidden_size)
        self.clayer_out_output = nn.Linear(n_qubits, hidden_size)

    def forward(self, x: torch.Tensor, init_states=None):
        """x shape: (batch, seq_len, input_size)"""
        batch_size, seq_len, _ = x.shape
        device = x.device

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_states

        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            v_t = torch.cat([h_t, x_t], dim=1)
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out_forget(self.vqc_forget(y_t)))
            i_t = torch.sigmoid(self.clayer_out_input(self.vqc_input(y_t)))
            g_t = torch.tanh(self.clayer_out_update(self.vqc_update(y_t)))
            o_t = torch.sigmoid(self.clayer_out_output(self.vqc_output(y_t)))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)  # (batch, seq_len, hidden_size)
        return hidden_seq, (h_t, c_t)


class StockQLSTM(nn.Module):
    """Same regression head shape as StockLSTM, swapping in the QLSTM cell."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 32,
        n_qubits: int = 4,
        n_qlayers: int = 1,
    ):
        super().__init__()
        self.qlstm = QLSTMCell(
            input_size=num_features,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_t, _) = self.qlstm(x)
        return self.linear(h_t).squeeze(-1)
