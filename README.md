# fusedkernels

Fused softmax and RMSNorm kernels written in Triton from scratch, benchmarked against PyTorch baselines, with a step-by-step walkthrough.

Project 7 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Write fused softmax and RMSNorm kernels in Triton and benchmark them against the PyTorch baselines across shapes.

## Why it matters

Kernel intuition separates infra engineers from API callers.

## Status

Kernels, tests and benchmark harness complete; GPU benchmark pending. Correctness is proven on CPU under Triton's interpreter (46 tests); bandwidth numbers need the T4.

## What is in here

| Module | Purpose |
|---|---|
| `fusedkernels.softmax` | Row-wise fused softmax: one program per row, whole row in one block, masked loads with `-inf`, FP32 accumulation |
| `fusedkernels.rmsnorm` | Row-wise fused RMSNorm: one read, one write, FP32 accumulation so `x^2` cannot overflow FP16 |
| `fusedkernels.reference` | The unfused PyTorch versions (3-4 passes), used as the `torch.compile` baseline |
| `fusedkernels.bench` | Six shapes x three implementations, median time with L2 flushed, achieved GB/s and % of the T4's 320 GB/s |
| `fusedkernels.plots` | Bandwidth-vs-shape bar charts |

## Running the tests

Triton has no macOS wheels, so the kernel tests run in a Linux container under `TRITON_INTERPRET=1` (exact, slow) or on a GPU:

```bash
./scripts/test-cpu.sh              # builds a small python:3.12 image with torch (CPU) + triton, runs pytest
pytest                             # on a Linux box with triton installed, or on a GPU
```

Tests cover power-of-two and ragged row lengths (1, 7, 100, 1025, ...), FP16 and FP32, large-input stability, non-contiguous inputs, and explicit block-size overrides.

## Benchmark

```bash
python -m fusedkernels.bench --out results/bench.json      # GPU only
```

`notebooks/kaggle_fusedkernels.ipynb` runs the tests on the GPU, the benchmark, and a block-size / `num_warps` sweep on one shape. Results will be added here once that run lands.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/fusedkernels/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
