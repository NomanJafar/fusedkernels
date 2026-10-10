"""Triton kernels vs PyTorch. Skipped where Triton is unavailable (macOS); run via
scripts/test-cpu.sh (interpreter) or on the GPU."""

import os

import pytest
import torch

triton = pytest.importorskip("triton")

from fusedkernels.rmsnorm import rmsnorm
from fusedkernels.softmax import softmax

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if DEVICE == "cpu" and not INTERPRET:
    pytest.skip("no GPU and TRITON_INTERPRET not set", allow_module_level=True)

ROW_LENGTHS = [1, 7, 64, 100, 1000, 1024, 1025, 4096]


@pytest.mark.parametrize("n_cols", ROW_LENGTHS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_softmax_matches_torch_for_power_of_two_and_ragged_rows(n_cols, dtype):
    x = torch.randn(5, n_cols, device=DEVICE, dtype=dtype) * 4
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=2e-3, rtol=2e-2)


def test_softmax_is_stable_for_large_inputs():
    x = torch.full((2, 256), 1e4, device=DEVICE, dtype=torch.float16)
    y = softmax(x)
    assert torch.isfinite(y).all() and torch.allclose(y.sum(-1).float(), torch.ones(2, device=DEVICE), atol=1e-2)


def test_softmax_handles_non_contiguous_input():
    x = torch.randn(64, 32, device=DEVICE).t()  # strides (1, 32)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1))


@pytest.mark.parametrize("n_cols", ROW_LENGTHS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_matches_torch(n_cols, dtype):
    x = torch.randn(5, n_cols, device=DEVICE, dtype=dtype) * 3
    w = torch.randn(n_cols, device=DEVICE, dtype=dtype)
    ref = torch.nn.functional.rms_norm(x.float(), (n_cols,), weight=w.float(), eps=1e-6).to(dtype)
    torch.testing.assert_close(rmsnorm(x, w, eps=1e-6), ref, atol=2e-2, rtol=2e-2)


def test_rmsnorm_fp16_does_not_overflow_in_the_square():
    x = torch.full((2, 512), 300.0, device=DEVICE, dtype=torch.float16)  # 300^2 > fp16 max
    w = torch.ones(512, device=DEVICE, dtype=torch.float16)
    y = rmsnorm(x, w, eps=1e-6)
    assert torch.isfinite(y).all() and torch.allclose(y.float(), torch.ones_like(y).float(), atol=1e-2)


@pytest.mark.parametrize("block", [1024, 2048, 4096])
def test_softmax_block_size_override_gives_same_result(block):
    x = torch.randn(4, 777, device=DEVICE)
    torch.testing.assert_close(softmax(x, block_size=block), torch.softmax(x, dim=-1))


def test_softmax_rejects_block_smaller_than_row():
    x = torch.randn(2, 777, device=DEVICE)
    with pytest.raises(AssertionError, match="online variant"):
        softmax(x, block_size=256)
