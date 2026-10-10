import json

from fusedkernels.plots import plot_bench


def test_plot_bench_writes_one_chart_per_op(tmp_path):
    rows = []
    for op in ("softmax", "rmsnorm"):
        for impl in ("torch", "triton"):
            for n, d in ((4096, 1024), (256, 32768)):
                rows.append({"op": op, "impl": impl, "rows": n, "cols": d, "ms": 0.1, "gb_s": 200.0, "pct_of_peak": 62.5})
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"peak_gbs": 320, "gpu": "T4", "rows": rows}))
    paths = plot_bench(p, tmp_path)
    assert sorted(x.name for x in paths) == ["rmsnorm_bandwidth.png", "softmax_bandwidth.png"]
    assert all(x.stat().st_size > 1000 for x in paths)
