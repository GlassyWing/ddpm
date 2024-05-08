import torch

from .sinusoidal import position_encoding_1d


def apply_rotary_position_embeddings(sinusoidal: torch.Tensor, *tensors):
    assert len(tensors) > 0, "at least one input tensor"

    # (cos(x), cos(y), ...) => (cos(x), cos(x), cos(y), cos(y), ...)
    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, 1)
    # (sin(x), sin(y), ...) => (sin(x), sin(x), sin(y), sin(y), ...)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, 1)

    cos_pos = cos_pos.expand_as(tensors[0])
    sin_pos = sin_pos.expand_as(tensors[0])

    outputs = []
    for t in tensors:
        t_r = torch.empty_like(t)
        t_r[..., 0::2] = -t[..., 1::2]
        t_r[..., 1::2] = t[..., 0::2]
        outputs.append(t * cos_pos + t_r * sin_pos)

    return outputs if len(tensors) > 1 else outputs[0]


class Rotary2D:

    def __init__(self, dim: int, base: float = 10000, resolution_multiplier: int = 1):
        self.dim = dim
        self.base = base
        self.pos_cached = None
        self.w_size_cached = None
        self.h_size_cached = None
        self.resolution_multiplier = resolution_multiplier

    def forward(self, x: torch.Tensor):
        H, W = x.size(2), x.size(3)
        assert H % 2 == 0
        assert W % 2 == 0
        if self.pos_cached is None or self.w_size_cached != W or self.h_size_cached != H:
            self.h_size_cached = H
            self.w_size_cached = W

            # (H, dim // 2) H x (sin(x), cos(x), ...)
            position_x = position_encoding_1d(self.dim // 2, H // self.resolution_multiplier,
                                              self.base, self.resolution_multiplier)
            # (W, dim // 2) W x (sin(y), cos(y), ...)
            position_y = position_encoding_1d(self.dim // 2, W // self.resolution_multiplier,
                                              self.base, self.resolution_multiplier)

            position_x = position_x.reshape(H, -1, 2)
            position_y = position_y.reshape(W, -1, 2)

            self.pos_cached = torch.empty(H * W, self.dim, dtype=x.dtype, device=x.device)
            for i in range(H):
                for j in range(W):
                    emb = torch.cat([
                        position_x[i, 0::2],
                        position_y[j, 0::2],
                        position_x[i, 1::2],
                        position_y[j, 1::2]
                    ], 0).flatten(-2) # sin(x), cos(x), sin(y), cos(y)
                    # emb = torch.cat([position_x[i], position_y[j]], 0)
                    self.pos_cached[i * W + j] = emb.to(x.dtype).to(x.device)
        return self.pos_cached
