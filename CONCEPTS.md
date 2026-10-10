# Concepts: why a fused kernel is faster, and how Triton lets you write one

`fusedkernels` implements the forward pass of softmax and RMSNorm in Triton and benchmarks them against PyTorch. Both are small enough to write in an afternoon and important enough to appear in every transformer layer. The point is not to beat PyTorch (its kernels are already fused); the point is to understand *why* fusion wins, by building one and measuring it.

## 1. The GPU memory hierarchy, in the numbers that matter

A T4 has:

| Level | Size | Bandwidth | Latency |
|---|---|---|---|
| Registers | 256 KB per SM (65,536 × 32-bit) | effectively free | ~0 |
| Shared memory / L1 | 64 KB per SM, configurable split | ~10 TB/s aggregate | ~30 cycles |
| L2 cache | 4 MB | ~1–2 TB/s | ~200 cycles |
| HBM / GDDR6 (global memory) | 16 GB | **320 GB/s** | ~500+ cycles |

and ~65 TFLOPS of FP16 tensor-core compute (8 TFLOPS FP32). The ratio matters: at 320 GB/s the card can move 160 billion FP16 values per second, but compute 65 trillion operations per second. **A kernel that does fewer than ~400 FLOPs per value loaded is bound by memory bandwidth, not compute.** Softmax does about 5 FLOPs per element. RMSNorm does about 4. Both are memory-bound by two orders of magnitude; their runtime is the time to read the input and write the output, and nothing else.

That single fact drives everything in this project.

## 2. What "fused" means and why it wins

Unfused softmax, the way a naive framework would run it, is four separate kernels:

```
m = x.max(dim=-1)          # read x, write m
e = exp(x - m)             # read x and m, write e
s = e.sum(dim=-1)          # read e, write s
y = e / s                  # read e and s, write y
```

Count the global-memory traffic for an N×D matrix: x is read three times, e is written once and read twice, y written once: roughly **7 N·D element transfers**. The arithmetic is trivial; each kernel launch also costs ~5–10 µs of fixed overhead.

A fused kernel loads one row of x into registers or shared memory **once**, computes max, exp, sum and the division while the data is on-chip, and writes y once: **2 N·D transfers**. On a memory-bound operation that is a 3.5× reduction in the only cost that matters. PyTorch's `torch.softmax` is already fused; `torch.compile` fuses chains of elementwise ops automatically. The benchmark will show the Triton kernel landing near both, and the write-up explains any gap.

RMSNorm is the same story: `y = x / sqrt(mean(x²) + eps) * w`. Unfused is three passes; fused is one read and one write.

## 3. The Triton programming model

CUDA makes you think in threads: each thread handles one or a few elements, and you manage shared memory, synchronisation and warp-level tricks by hand. Triton moves the unit of work up one level: **a program instance handles a block of elements**, and the compiler decides how to map that block onto threads, registers and shared memory.

The pieces:

- **Grid.** You launch a grid of program instances. For a row-wise op on an N×D matrix the natural grid is N instances, one per row: `kernel[(N,)](...)`.
- **`tl.program_id(axis)`** tells an instance which row it owns.
- **Block of offsets.** `offs = tl.arange(0, BLOCK)` is a vector of indices; `BLOCK` must be a power of two and is a compile-time constant (`tl.constexpr`).
- **Masked loads and stores.** Rows are rarely a power of two long. `tl.load(ptr + offs, mask=offs < D, other=-inf)` loads only the valid elements and fills the rest, so the reduction ignores them. Choosing `other` correctly per operation (−∞ for max, 0 for sum) is where correctness bugs live.
- **Reductions over the block.** `tl.max(x, axis=0)`, `tl.sum(x, axis=0)`. The compiler turns these into tree reductions across threads and warps.
- **Elementwise math.** `tl.exp`, `tl.sqrt`, `tl.rsqrt`, operating on the whole block.
- **Strides.** Pointers are computed from strides, so the kernel works for non-contiguous inputs: `x_ptr + row * stride_row + offs * stride_col`.

A softmax row kernel is about 15 lines. The compiler does the register allocation, the shared-memory staging and the warp shuffles that would be a few hundred lines of careful CUDA.

## 4. Two kernel shapes: one block per row vs. looping

If `BLOCK ≥ D`, one block covers the row: load the whole row, reduce, store. Simplest and fastest, as long as the row fits in registers (a few thousand FP32 elements per instance is fine; beyond that the compiler spills to local memory and performance collapses).

For long rows (D = 32k, as in a vocabulary-sized softmax), loop over the row in chunks with an **online** algorithm: keep a running max and a running sum, rescaling the sum when the max changes. This is the same trick FlashAttention uses to avoid materialising the attention matrix; the single-row version is a good way to understand it. This project implements the single-block version first and the online version only if the benchmark shapes need it.

## 5. What `num_warps` and `BLOCK` do

Each Triton program instance runs on `num_warps × 32` threads. More warps means more parallelism within a row and more registers in total, but fewer concurrent instances per SM. For memory-bound row ops the goal is simply to keep enough loads in flight to saturate bandwidth; a common heuristic is `num_warps = max(1, min(16, BLOCK // 256))`. `triton.autotune` can search over configurations; this project tries three block sizes by hand and reports all three rather than hiding the search.

## 6. Precision

Inputs are FP16 (what the model uses); accumulate in FP32 (`x.to(tl.float32)` after the load). Softmax in FP16 overflows `exp` for inputs above ~11 unless the max is subtracted first, and even then summing hundreds of FP16 values loses precision. RMSNorm's `mean(x²)` in FP16 overflows for |x| above 256. The PyTorch reference kernels accumulate in FP32 internally; so must ours, or the correctness test fails at the tolerances PyTorch meets.

## 7. How to benchmark a kernel honestly

- **Warm up** (first launch compiles the kernel; Triton caches by shape and dtype).
- **Time with CUDA events or `triton.testing.do_bench`**, which flushes the L2 cache between runs so a 1 MB input is not served from cache at 10× real bandwidth.
- **Report achieved bandwidth**, not just time: `bytes_moved / seconds`, where bytes moved is 2·N·D·2 for FP16 in and out. The fraction of the 320 GB/s peak is the number that says whether the kernel is good; a time alone says nothing.
- **Sweep shapes**: (N, D) in {(4096, 1024), (4096, 4096), (1024, 8192), (256, 32768), (16384, 512), (64, 65536)}: short rows stress launch overhead and occupancy, long rows stress the single-block limit.
- **Compare three things**: PyTorch eager (`torch.softmax`, `torch.nn.functional.rms_norm`), `torch.compile` of the unfused expression, and the Triton kernel.

## 8. Testing without a GPU

`TRITON_INTERPRET=1` runs Triton kernels on the CPU through NumPy, slowly but exactly. Tests compare against PyTorch at FP16-appropriate tolerances for several row lengths including non-power-of-two ones, so masking bugs show up on a laptop. The interpreter does not measure performance; the benchmark runs on the T4.

## 9. What the experiment will show

1. Correctness on CPU under the interpreter, then on the T4 against PyTorch.
2. Achieved bandwidth vs shape for all three implementations, as a fraction of 320 GB/s.
3. The effect of block size and `num_warps` on the Triton kernel.
4. Where Triton loses: tiny rows (launch overhead dominates) and very long rows (register spill), with the online variant as the fix if needed.

## Further reading

- Triton tutorials: *Fused Softmax* and *Layer Normalization* (triton-lang.org). This project's kernels are written from the programming model above and then compared with those.
- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018).
- Dao et al., *FlashAttention* (2022), section 3, for the same trick applied to attention.
- NVIDIA T4 datasheet for the bandwidth and compute figures used here.
