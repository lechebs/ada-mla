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
|pytorch eager                                          |0.357|
||
|naive (non fused gemm [`QM=QK=CN=16`] + softmax)       | 1.91|
||
|fused (fully tiled)  [`QM=QK=CN=16`]                   |  ~30|
| + (no C vertical tiling) [`QM=32` `QK=16` `CN=8`]     |19.29|
| + (no Q horizontal tiling) [`QM=32` `QK=16` `CN=8`]   |14.27|
| + [`QM=QK=CN=16`]                                     | 7.21|
||
|split-kv|

- Using shmem to accumulate output results degrades performance apparently.
- I think I've to quickly move to fp16, otherwise shmem isn't enough!
- When the number of threads in the block is fixed to `256`, increasing CTileN while reducing QTileM is better (at least without split-kv), I guess because it exposes more block parallelism and reduces the number of iterations along C. It could also be due to less shmem bank conflicts. I initially chose a taller Q block since it could be better in terms of AI (right?).

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
