import torch.nn.functional as F
import torch
from torch import nn
from .pscan import pscan, pscan_rev


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        # return self.dropout(self.w2(F.silu(self.w1(x))))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, axis: int = -1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.axis = axis
        if axis != -1:
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        else:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(self.axis, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MinGRUBlock(nn.Module):

    def __init__(self, dim, hidden_dim=None, dropout=0.0, has_ffn=True, pre_norm=True, reverse=False):
        super().__init__()
        self.input_size = dim
        self.hidden_size = hidden_dim if hidden_dim is not None else dim
        self.dropout_p = dropout
        self.has_ffn = has_ffn
        self.pre_norm = pre_norm
        
        if reverse:
            self.scan_op = pscan_rev
        else:
            self.scan_op = pscan

        self.graph_gates = nn.Linear(
            self.input_size, self.hidden_size, bias=False
        )
        self.graph_can = nn.Linear(
            self.input_size, self.hidden_size, bias=False
        )

        if has_ffn:
            self.ffn_norm = RMSNorm(self.input_size)
            self.ff = FeedForward(
                dim=self.input_size,
                hidden_dim=self.hidden_size,
                multiple_of=1,
                dropout=self.dropout_p,
            )
        self.layer_norm = RMSNorm(self.input_size)
        # self.layer_norm = nn.LayerNorm(self.input_size)

    def forward(self, inp, hx_prev=None):
        """
        Args:
            inp (B, L, D):
            hx_prev (B, 1, D):
        Returns:
            out (B, L, D)
        """
        bsz, seq_len, dim = inp.shape

        if hx_prev is not None:
            assert inp.shape == hx_prev.shape

        if self.pre_norm:
            norm_inp = self.layer_norm(inp)
        else:
            norm_inp = inp

        # (B, L, D)
        beta = torch.sigmoid(self.graph_gates(norm_inp))

        # (B, L, D)
        hx_hat = self.graph_can(norm_inp)

        a = 1 - beta
        x = beta * hx_hat

        if hx_prev is not None:
            # (B, 1, D)
            hx_next = a * hx_prev + x
        else:
            hx_next = self.scan_op(a, x)  # a * hx_prev + x => (1 - beta) * hx_prev + beta * hx_hat

        # inp = inp + hx_next
        inp = hx_next
        if self.has_ffn:
            out = inp + self.ff(self.ffn_norm(inp))
        else:
            out = inp

        return out, hx_next
