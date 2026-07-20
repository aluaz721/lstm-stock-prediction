"""
Classical LSTM baseline, following the shape of ShallowRegressionLSTM in
the SoftServe_QLSTM reference repo: a single-layer LSTM whose FINAL hidden
state is passed through one linear layer to produce a single scalar
prediction (the next-day scaled Close price).
"""
import torch
from torch import nn


class StockLSTM(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, num_features) -> (batch_size,)"""
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))
        # hn shape: (num_layers, batch, hidden_size); take the last layer's
        # final hidden state, matching the reference repo's hn[0] for the
        # num_layers=1 case, generalized to hn[-1] for num_layers > 1.
        out = self.linear(hn[-1]).squeeze(-1)
        return out
