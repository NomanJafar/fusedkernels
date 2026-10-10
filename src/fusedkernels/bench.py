"""Benchmark bookkeeping: shapes, bytes moved, achieved bandwidth. The timing loop
lives in `run_benchmarks` and needs a GPU; everything else is testable anywhere."""

from __future__ import annotations

import json
from pathlib import Path

import torch

# (rows, cols): short rows stress launch overhead and occupancy; long rows the single-block limit.
SHAPES: list[tuple[int, int]] = [(16384, 512), (4096, 1024), (4096, 4096), (1024, 8192), (256, 32768), (64, 65536)]

T4_PEAK_GBS = 320.0


def bytes_moved(n_rows: int, n_cols: int, elem_bytes: int, extra_vectors: int = 0) -> int:
    """One read of the input, one write of the output, plus any per-column vectors (rmsnorm weight)."""
    return 2 * n_rows * n_cols * elem_bytes + extra_vectors * n_cols * elem_bytes


def achieved_bandwidth_gbs(bytes_total: float, ms: float) -> float:
    return bytes_total / (ms / 1000) / 1e9


def _time_ms(fn, warmup: int = 10, rep: int = 100) -> float:
    """Median kernel time in ms using triton's do_bench (flushes L2 between runs)."""
    import triton.testing

    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def _cases(x, w, d, compiled_softmax, compiled_rmsnorm, softmax, rmsnorm) -> dict:
    """Bound closures for one shape (a plain dict of lambdas inside the loop would all see the last shape)."""
    return {
        ("softmax", "torch.softmax"): lambda: torch.softmax(x, dim=-1),
        ("softmax", "torch.compile(unfused)"): lambda: compiled_softmax(x),
        ("softmax", "triton"): lambda: softmax(x),
        ("rmsnorm", "F.rms_norm"): lambda: torch.nn.functional.rms_norm(x, (d,), weight=w, eps=1e-6),
        ("rmsnorm", "torch.compile(unfused)"): lambda: compiled_rmsnorm(x, w),
        ("rmsnorm", "triton"): lambda: rmsnorm(x, w),
    }


def run_benchmarks(dtype: torch.dtype = torch.float16, shapes: list[tuple[int, int]] | None = None,
                   peak_gbs: float = T4_PEAK_GBS, out: Path | str | None = None) -> list[dict]:
    """Time torch eager, torch.compile(unfused) and the Triton kernels across shapes. GPU only."""
    from fusedkernels.reference import rmsnorm_unfused, softmax_unfused
    from fusedkernels.rmsnorm import rmsnorm
    from fusedkernels.softmax import softmax

    assert torch.cuda.is_available(), "benchmarks need a GPU"
    compiled_softmax = torch.compile(softmax_unfused)
    compiled_rmsnorm = torch.compile(rmsnorm_unfused)
    elem = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for n, d in shapes or SHAPES:
        x = torch.randn(n, d, device="cuda", dtype=dtype)
        w = torch.randn(d, device="cuda", dtype=dtype)
        cases = _cases(x, w, d, compiled_softmax, compiled_rmsnorm, softmax, rmsnorm)
        for (op, impl), fn in cases.items():
            ms = _time_ms(fn)
            nbytes = bytes_moved(n, d, elem, extra_vectors=1 if op == "rmsnorm" else 0)
            gbs = achieved_bandwidth_gbs(nbytes, ms)
            rows.append({"op": op, "impl": impl, "rows": n, "cols": d, "dtype": str(dtype).split(".")[-1],
                         "ms": ms, "gb_s": gbs, "pct_of_peak": 100 * gbs / peak_gbs})
            print(f"{op:8s} {impl:24s} {n:>6}x{d:<6} {ms:8.3f} ms  {gbs:7.1f} GB/s  {100 * gbs / peak_gbs:5.1f}%")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"peak_gbs": peak_gbs, "gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=1))
    return rows


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="benchmark fused softmax / rmsnorm vs PyTorch")
    p.add_argument("--out", default="results/bench.json")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--peak-gbs", type=float, default=T4_PEAK_GBS)
    a = p.parse_args(argv)
    run_benchmarks(dtype=getattr(torch, a.dtype), peak_gbs=a.peak_gbs, out=a.out)


if __name__ == "__main__":
    main()
