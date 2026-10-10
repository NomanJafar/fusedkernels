"""Row-wise fused softmax in Triton. One program instance per row; the whole row is
loaded once into a block, reduced on-chip, and written once. See CONCEPTS.md sections 2-4."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, stride_xr, stride_xc, stride_yr, stride_yc, n_cols,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    # Padding with -inf makes the masked lanes contribute exp(-inf) = 0 to the sum
    # and never win the max.
    x = tl.load(x_ptr + row * stride_xr + offs * stride_xc, mask=mask, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride_yr + offs * stride_yc, y.to(y_ptr.dtype.element_ty), mask=mask)


def _num_warps(block: int) -> int:
    return max(1, min(16, block // 256))


def softmax(x: torch.Tensor, block_size: int | None = None) -> torch.Tensor:
    assert x.dim() == 2, "2-D input (rows x cols); reshape higher-rank inputs first"
    n_rows, n_cols = x.shape
    block = block_size or triton.next_power_of_2(n_cols)
    assert block >= n_cols, f"block_size {block} smaller than row length {n_cols}: use the online variant"
    y = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](x, y, x.stride(0), x.stride(1), y.stride(0), y.stride(1), n_cols,
                               BLOCK=block, num_warps=_num_warps(block))
    return y
