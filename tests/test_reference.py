"""Reference (PyTorch) implementations and the benchmark bookkeeping, testable without Triton."""

import pytest
import torch

from fusedkernels.bench import SHAPES, achieved_bandwidth_gbs, bytes_moved
from fusedkernels.reference import rmsnorm_unfused, softmax_unfused


def test_softmax_unfused_matches_torch():
    x = torch.randn(8, 100)
    torch.testing.assert_close(softmax_unfused(x), torch.softmax(x, dim=-1))


def test_rmsnorm_unfused_matches_torch():
    x, w = torch.randn(8, 100), torch.randn(100)
    torch.testing.assert_close(rmsnorm_unfused(x, w, eps=1e-6),
                               torch.nn.functional.rms_norm(x, (100,), weight=w, eps=1e-6))


def test_bytes_moved_counts_one_read_and_one_write():
    assert bytes_moved(n_rows=4096, n_cols=1024, elem_bytes=2, extra_vectors=0) == 2 * 4096 * 1024 * 2
    # rmsnorm also reads the weight vector once
    assert bytes_moved(4096, 1024, 2, extra_vectors=1) == 2 * 4096 * 1024 * 2 + 1024 * 2


def test_bandwidth_is_bytes_over_seconds():
    assert achieved_bandwidth_gbs(bytes_total=320e9, ms=1000.0) == pytest.approx(320.0)


def test_shapes_cover_short_and_long_rows():
    cols = [d for _, d in SHAPES]
    assert min(cols) <= 512 and max(cols) >= 32768 and len(SHAPES) == 6
