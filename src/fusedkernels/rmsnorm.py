"""Row-wise fused RMSNorm in Triton: y = x / sqrt(mean(x^2) + eps) * w, one read, one write."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_xr, stride_xc, stride_yr, stride_yc, n_cols, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    # Accumulate in fp32: x^2 overflows fp16 for |x| > 256 (CONCEPTS.md section 6).
    x = tl.load(x_ptr + row * stride_xr + offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(var + eps) * w
    tl.store(y_ptr + row * stride_yr + offs * stride_yc, y.to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int | None = None) -> torch.Tensor:
    assert x.dim() == 2 and weight.shape == (x.shape[1],)
    n_rows, n_cols = x.shape
    block = block_size or triton.next_power_of_2(n_cols)
    assert block >= n_cols
    y = torch.empty_like(x)
    _rmsnorm_kernel[(n_rows,)](x, weight, y, x.stride(0), x.stride(1), y.stride(0), y.stride(1), n_cols, eps,
                               BLOCK=block, num_warps=max(1, min(16, block // 256)))
    return y
