"""
These tests require torch (and pennylane's torch interface for QLSTM),
which this sandbox can't install due to disk constraints on the CUDA
toolkit dependency chain (see conversation notes / README). They're
written to run for real in CI and on your own machine -- run
`pytest tests/test_models.py -v` after `pip install -r requirements.txt`.
"""
import pytest

torch = pytest.importorskip("torch")  # noqa: E402 -- must precede the torch-dependent imports below

from src.models.lstm import StockLSTM  # noqa: E402
from src.models.qlstm import StockQLSTM, QLSTMCell  # noqa: E402
from src.data.features import FEATURE_COLUMNS  # noqa: E402


BATCH_SIZE = 4
SEQ_LEN = 10
NUM_FEATURES = len(FEATURE_COLUMNS)  # was hardcoded to 11; drifted stale after the returns-target rewrite


@pytest.fixture
def sample_batch():
    return torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)


class TestStockLSTM:
    def test_forward_shape(self, sample_batch):
        model = StockLSTM(num_features=NUM_FEATURES, hidden_size=16)
        out = model(sample_batch)
        assert out.shape == (BATCH_SIZE,)

    def test_gradient_flows(self, sample_batch):
        model = StockLSTM(num_features=NUM_FEATURES, hidden_size=16)
        out = model(sample_batch)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} got no gradient"
            assert torch.any(param.grad != 0), f"{name} gradient is all zero"

    def test_single_training_step_reduces_loss_on_toy_data(self):
        # Overfit a single tiny batch -- if this doesn't work, something
        # is fundamentally wrong with the forward/backward wiring.
        torch.manual_seed(0)
        model = StockLSTM(num_features=NUM_FEATURES, hidden_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(8, SEQ_LEN, NUM_FEATURES)
        y = torch.randn(8)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]


class TestQLSTMCell:
    def test_forward_shape(self, sample_batch):
        cell = QLSTMCell(input_size=NUM_FEATURES, hidden_size=8, n_qubits=4, n_qlayers=1)
        hidden_seq, (h_t, c_t) = cell(sample_batch)
        assert hidden_seq.shape == (BATCH_SIZE, SEQ_LEN, 8)
        assert h_t.shape == (BATCH_SIZE, 8)
        assert c_t.shape == (BATCH_SIZE, 8)

    def test_gates_have_independent_parameters(self):
        cell = QLSTMCell(input_size=NUM_FEATURES, hidden_size=8, n_qubits=4, n_qlayers=1)
        forget_params = {id(p) for p in cell.clayer_out_forget.parameters()}
        input_params = {id(p) for p in cell.clayer_out_input.parameters()}
        assert forget_params.isdisjoint(input_params), (
            "gate output layers are sharing parameters -- "
            "see the deliberate-deviation note in qlstm.py"
        )


class TestStockQLSTM:
    def test_forward_shape(self, sample_batch):
        model = StockQLSTM(num_features=NUM_FEATURES, hidden_size=8, n_qubits=4)
        out = model(sample_batch)
        assert out.shape == (BATCH_SIZE,)

    def test_gradient_flows_through_quantum_layers(self, sample_batch):
        model = StockQLSTM(num_features=NUM_FEATURES, hidden_size=8, n_qubits=4)
        out = model(sample_batch)
        loss = out.sum()
        loss.backward()

        # specifically check a quantum layer's weights got gradients --
        # this is the part most likely to silently break (e.g. if a
        # QNode's inputs got detached from the autograd graph somewhere)
        quantum_params = list(model.qlstm.vqc_forget.parameters())
        assert len(quantum_params) > 0
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in quantum_params)

    def test_single_training_step_reduces_loss_on_toy_data(self):
        torch.manual_seed(0)
        model = StockQLSTM(num_features=NUM_FEATURES, hidden_size=8, n_qubits=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        x = torch.randn(4, SEQ_LEN, NUM_FEATURES)
        y = torch.randn(4)

        losses = []
        for _ in range(10):  # fewer steps: quantum sim is slow
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]
