## TODO

- [ ] Find a way to measure execution time. From python or in C++? What about using ncu?

## Roadmap

- naive fused
- flash-attn optimizations
- matmul optimizations
- mma
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
