## Fused tiling strategy

```
                                                    +--------------+
                                                    |//////////////|
                CTileN                              +--------------+
                +---+----------------+              |              |
                |///| QTileK         |              |              |
                +---+----------------+              |              |
                |///|                |              |              |
                |///|                |              |              |
                |///|                |              |              |
QTileK          +---+----------------+              +--------------+
+-------------+ +---+----------------+  ^           +--------------+
|/////////////| |   | --->           |  |           |              |
+-------------+ +---+----------------+ num_heads    +--------------+
|             | |                    |  |           |              |
+-------------+ +--------------------+  v           +--------------+
<---HeadDim---> <-----seq_length----->

|   \
|    \          |   \
|     \         |    \
+------+        +----+      ^
|//////|        |    |      |
|//////|        |    |      |
+------+        +----+   QTileM=32
|//////|        |    |      |
|//////|        |    |      |
+------+        +----+      v

<------>        <---->
QTileK=16      CTileN=8
```

HeadDim = 576

## Worklog

Benchmarks for `seq_length=4096`, `head_dim=576` and `num_heads=128` with thread blocks of size `256`.

|kernel|time (ms)|
|:-----|--------:|
|pytorch eager                                         |0.315|
||
|naive (non fused gemm [`QM=QK=CN=16`] + softmax)      | 1.61|
||
|fused (fully tiled)  [`QM=QK=CN=16`]                  |  ~30|
| + (no C vertical tiling) [`QM=32 QK=16 CN=8`]        |19.29|
| + (no Q horizontal tiling) [`QM=32 QK=16 CN=8`]      |14.27|
| + [`QM=QK=CN=16`]                                    | 7.21|
| + [`QM=8 QK=16 CN=32`]                               | 3.72|
| + allow unrolling Q and C gmem->shmem loops          | 2.64|
| + slice-k QC^T product with vertical coarsening      | 1.18|
| + warp-parallel PC product with vertical coarsening  | 1.02|
| + vectorized C load with unroll 4                    |0.964|
| + padding Cs to avoid bank conflicts                 |0.751|

Moving to ncu execution time (more precises, flushes caches at each iteration).
|kernel|time (ms)| peak flops (fp32) |
|:-----|---:|---:|
|pytorch eager                                                               |0.434|48%|
|pytorch eager (head_dim_v fix)                                              |0.446|48%|
||
| (fused) vectorized C load with unroll 4                                    |1.190|13%|
| + padding Cs to avoid bank conflicts                                       |0.934|16%|
| + 2-stage pipeline to load C                                               |0.795|19%|
| + 3-stage pipeline to load C                                               |0.769|20%|
| + two cols coarsening for PC product                                       |0.664|22%|
| + two cols coarsening for QC^T product [`QTileK=8`]                        |0.641|24%|
| + head_dim_v fix :P                                                        |0.566|25%|
| + `QTileM=CTileN=16` + 4 col coarse PC + O reg tiling + force Ps vec load |*0.440|31%|

(*excluding splitkv reduce kernel)

- Using shmem to accumulate output results degrades performance apparently.
- I think I've to quickly move to fp16, otherwise shmem isn't enough!
- When the number of threads in the block is fixed to `256`, increasing CTileN while reducing QTileM is better (at least without split-kv), I guess because it exposes more block parallelism and reduces the number of iterations along C. It could also be due to less shmem bank conflicts. I initially chose a taller Q block since it could be better in terms of AI (right?).
- Vectorizing gmem->shmem loads doesn't seem to impact performance, at least when the compiler can unroll their loops.

- It looks like vectorized shmem loads (LDS.128) broadcast require two wavefronts (and not 4?), that's why
they appear like they are producing bank conflicts.

- The docs state that cp.async achieves best performance when both src and dst are 128 bytes aligned, shmem can be aligned with `extern __shared__ char shmem[] alignas(128);`, while gmem allocations should already be 256 bytes aligned, but using `QTileK=8`, only 4/8 warps load C tiles at 128 bytes boundaries.
Weirdly enough, the uncoalesced gmem accesses are caused only by the presence of 16 byte padding in Cs (in shared memory!), which causes tiles rows to be split across 32 byte sectors, it is sufficient to use 32 byte padding to avoid them. Perhaps if the cp.async shmem store has to be split in multiple accesses, also the gmem access has to be performed in multiple transactions.
The uncoalesced shared accesses are instead caused by Cs tiles misalignment to 128 byte boundaries, which can be fixed by explicitly aligning shmem buffers, using `QTileK=16` and avoid padding Cs.
Note here that we are using 16 byte cp.async instructions, using the 8 byte variant the 16 byte padding of Cs doesn't generate excessive global loads, but in that casee it seems like there's no way to avoid the excessive shared wavefronts (perhaps that's why they say that best performance is achieved with the 16 byte version). Perhaps only the 16 byte version suffers from the coupling between shmem tiles aligment and gmem excessive accesses.
TLDR; prefer `QTileK=16` when using two col coarsening, use `alignas(128)` on shmem buffer and either avoid padding for Cs, or use a padding multiple of 32 bytes.

- I can swizzle within each tile; with two col coarsening, each warp accesses a tile (16, QTileK), one column after the other. The swizzle has to keep 128 bit words intact, so it has to be performed at the granularity of float4. Note that for 128 bit shared requests, the access is split in 4 cycles, each serving 8 consecutive threads in the warp. It is then sufficient to avoid bank conflicts for 128 bit words within each quarter warp (0-7, 8-15, 16-23, 24-31). Two col coarsening is fine here, since half warps are already avoiding bank conflicts with 128 bit accesses.

- An attempt to reduce the barrier stalls: For the QC^T product, using two col coarsening only for the first 512 columns, such that the last 64 ones can be processed by all warps (with `QTileK=8`) increases the sync overhead (due to an additional `__syncthreads()`), bringing no performance increase (even when removing the need of `__synchtreads()` by changing the prefetch behavior such that only a `syncwarp()` is required).

- Four col coarsening for PC product reduces short scoreboard and MIO throttle, but 4 warps remain idle, increasing barrier stalls (no increase in performance). With `QTileM=16`, those 4 warps could be used! What if C gets loaded again for the PC product? I could increase `QTileM` then, but note that split-k will be a must. I could load Cs in two chunks, for the PC product the second chunk could be immediately reused while the other could be asynchronously copied.
- Four col coarsening for QC^T product shifts the stalls from short scoreboard and MIO throttle to long scoreboard and LG throttle (even when reducing `QTileK` such that each warp always prefetches a (`QTileM`, 32) tile), bringing no performance improvement. I guess increasing `QTileM` would be beneficial to reduce gmem access penalties; from there four col coarsening could actually help (or at least allow to not reduce `QTileK`, such that more of `HeadDim` could be processed in parallel by the warp).

- Revert to `QTileM=CTileN=16`. It would be great not to reduce `CTileN`, but there wouldn't be enough shmem. This configuration should increase the AI of gmem access wrt to `QTileM=8`, `CTileN=32`. Note that the AI wrt to gmem access is determined only by the first gemm, since for the second one, everything is in shmem. But we could say that throughput of computing the Ps tile is a form of bandwidth that could limit the throughput of the second gemm, so we could talk about AI even in that case. Most notably, the width of the Ps tile (`CTileN`) has no effect on this AI, since it is the k dimension of the gemm, while the height (`QTileM`) does.
It is easy to visualize why a larger `QTileM` would be beneficial, the same Cs tile can be reused for a taller tile of the output. Doubling the size of `QTileM`, C will be fetched half the number of times as before. The size of `CTileN` influences only the AI at the shmem access level I guess, since at the end of the day each block has to fetch all of C from gmem, differently from a standard gemm. Initially I moved to smaller `QTileM` values since anything more than 8 would not use all of the SMs, and looking back at the speedup obtained, it is less than 2x, which fits with the idea that this speedup was only due to the usage of all of the SMs (from 8 to 16). A smaller `CTileN` size can help hide better the latency of loading Cs (with increased `QTileM`).

- Move the prefetching of C before the final `__syncthreads()`, or check if you can let each warp work only on a specific subset of chunks from Cs, for both products, in order to avoid block level synchronization.

## TODO

- Avoid Cs padding.
- Choose a number of warps that divides evenly Cs size -> 9 should be good, but it doesn't seem to reduce overall sync stalls for some reason.. only for the PC loop. I guess sync latency could be hidden by fitting two blocks on each SM.

- Slice-k with more than 8 warps -> tried with 9, sync latency after QC^T loop increases.
- 2D coarsening -> increase QTileM.
- Move to fp16 to reduce shmem requirements and fit two blocks in each SM.
- Split-k.
- Try to pipeline shmem->reg loads (how the hell?)

## Roadmap

- naive fused
- flash-attn optimizations
- gemm optimizations
  - cutlass hierarchy
  - vector loads
  - shmem bank conflicts
  - mma
  - sw pipelining
- fp8

- KV cache update?

Compare against flash-attention decoding, FlashMLA doesn't support Ada.

## References

- https://x.com/karpathy/status/1789666350878601581

- https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://docs.pytorch.org/cppdocs/index.html
- https://pybind11.readthedocs.io/en/stable/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
- https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions

- https://siboehm.com/articles/22/CUDA-MMM
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://www.spatters.ca/mma-matmul
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
- https://gau-nernst.github.io/fa-5090/

- https://github.com/NVIDIA/cutlass#documentation
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
. https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/
- https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/

- https://mlc.ai/modern-gpu-programming-for-mlsys/index.html

- https://www.youtube.com/watch?v=Y-o545eYjXM
- https://www.youtube.com/@GPUMODE
- https://www.youtube.com/watch?v=0VLAoVGf_74

- https://arxiv.org/abs/1805.02867
- https://arxiv.org/abs/2205.14135
- https://arxiv.org/pdf/2307.08691
- https://arxiv.org/abs/2407.08608
- https://arxiv.org/abs/2405.04434
- https://crfm.stanford.edu/2023/10/12/flashdecoding.html

## Suggested reads

- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- https://blog.ezyang.com/2019/05/pytorch-internals/
- https://hazyresearch.stanford.edu/blog/2024-05-12-tk
- https://dl.acm.org/doi/10.1145/1356052.1356053
- https://www.cs.cmu.edu/~mgormley/courses/10423/schedule.html
- https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/
