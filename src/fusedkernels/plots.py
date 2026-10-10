"""Bandwidth-vs-shape chart from results/bench.json."""

from __future__ import annotations

import json
from pathlib import Path


def plot_bench(bench_path: str | Path, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(bench_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for op in ("softmax", "rmsnorm"):
        rows = [r for r in data["rows"] if r["op"] == op]
        impls = list(dict.fromkeys(r["impl"] for r in rows))
        shapes = list(dict.fromkeys(f"{r['rows']}x{r['cols']}" for r in rows))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.8 / len(impls)
        for i, impl in enumerate(impls):
            ys = [next(r["pct_of_peak"] for r in rows if r["impl"] == impl and f"{r['rows']}x{r['cols']}" == s) for s in shapes]
            ax.bar([j + i * width for j in range(len(shapes))], ys, width, label=impl)
        ax.set_xticks([j + width * (len(impls) - 1) / 2 for j in range(len(shapes))])
        ax.set_xticklabels(shapes, rotation=20)
        ax.set_ylabel(f"% of {data['peak_gbs']:.0f} GB/s peak")
        ax.set_title(f"{op}: achieved memory bandwidth on {data.get('gpu', 'GPU')}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"{op}_bandwidth.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths
