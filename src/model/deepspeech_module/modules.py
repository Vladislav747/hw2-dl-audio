import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.norm(x)


class NormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            dropout=dropout_rate,
            batch_first=False,
            bidirectional=True,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs, hs=None):
        """
        Forward pass through GRU and norm layer
        """
        x, hs = self.gru(inputs, hs)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
        batch_size, time_steps = x.shape[1], x.shape[0]
        x = x.view(batch_size * time_steps, -1)
        x = self.batch_norm(x)
        x = x.view(time_steps, batch_size, -1).contiguous()

        return x, hs