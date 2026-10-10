"""Unfused PyTorch reference implementations: what a framework runs without fusion.

Each is written as separate passes on purpose; `torch.compile` of these functions is
one of the benchmark baselines, `torch.softmax` / `F.rms_norm` (already fused) another.
"""

from __future__ import annotations

import torch


def softmax_unfused(x: torch.Tensor) -> torch.Tensor:
    m = x.max(dim=-1, keepdim=True).values      # pass 1: read x
    e = torch.exp(x - m)                        # pass 2: read x, write e
    s = e.sum(dim=-1, keepdim=True)             # pass 3: read e
    return e / s                                # pass 4: read e, write y


def rmsnorm_unfused(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)  # pass 1: read x
    y = xf * torch.rsqrt(var + eps)             # pass 2: read x, write y
    return (y * weight.float()).to(x.dtype)     # pass 3: read y and w, write out
