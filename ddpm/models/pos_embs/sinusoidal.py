import torch
import math


def position_encoding_1d(d_model: int, length: int, base: float = 10000, interval: float = 1):
    assert d_model % 2 == 0, f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})"

    pe = torch.zeros(int(length * interval), d_model)
    position = torch.arange(0, length, 1.0 / interval, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
