# Multihead Latent Attention decoding kernel for Ada GPUs

## Requirements

The code was developed and tested on PyTorch 2.12.1 and CUDA 13.2.

Install the dependencies with `pip install -r requirements.txt`

The code targets `sm_89`, but informally supports `sm_80` as well.

PyTorch CUDA extension, available inside the script as a Python module via PyBind11.

## Usage

The `mla.py` scripts contains some boilerplate to verify the correctness of the implementation and perform a quick benchmark with `torch.Timer`.

```
python3 mla.py --test --seq-length 4096 -k fused

python3 mla.py --benchmark -s 4096 8192 16384 -k torch split --fp16
```

## Multihead Latent Attention

Proposed in DeepSeekV2.. weight absorption math here.

### MLA in practice

Quite simple if you get past the math stage, softmax(QC^T)C

flash-attention, and why fusion.

## Kernels

The implemented kernels operate in the weight absorbed regimes.

PyTorch eager was used as baseline (or better, as target).

`torch.nn.attention.scaled_dot_product_attention` has multiple backends, the only two that are compatible with `head_dim=576` are the `MATH` backend and the `MEMORY_EFFICIENT` (a Triton implementation from the xformers library).

### Fused (FP32)

The fused kernel was later profiled in split-k manner, but formally lacks a combine.

### Split-K fused (FP16)

```
 - grid tiles:
                                                                                          512             64
                                                                               +-------------------------+--+
                                                                               |/////////////////////////|  |
                                                                               |/////////////////////////|  |
                                                    seq_len       C^t          |/////////////////////////|  |
                                     +-------------------------------------+   |/////////////////////////|  |  C
                                     |//////////////////|                  |   |/////////////////////////|  |
                                     |//////////////////|                  |   |/////////////////////////|  |
                                     |//////////////////|                  |   |/////////////////////////|  |
                                     |//////////////////|                  |   |-------------------------+  | seq_len
                                     |//////////////////|                  |   |                         |  |
                                576  |//////////////////|                  |   |                         |  |
                                     |//////////////////|                  |   |                         |  |
                                     |//////////////////|                  |   |                         |  |
                                     |//////////////////|                  |   |                         |  |
                                     |//////////////////|                  |   |                         |  |
                                     |//////////////////|                  |   |                         |  |
                                     +-------------------------------------+   +-------------------------+--+
                   Q
     +---------------------------+   +------------------+------------------+   +-------------------------+
     |///////////////////////////|   |///////cta0///////|       cta4       |   |//////////cta0///////////| 32
     |---------------------------|   +------------------+------------------+   |-------------------------|
     |                           |   |       cta1       |       cta5       |   |          cta1           |
 128 |---------------------------|   +------------------+------------------+   |-------------------------|  O
     |                           |   |       cta2       |       cta6       |   |          cta2           |
     |---------------------------|   +------------------+------------------+   |-------------------------|
     |                           |   |       cta3       |       cta7       |   |          cta3           |
     +---------------------------+   +------------------+------------------+   +-------------------------+
                  576                                   P                                   +
                                                                               +-------------------------+
                                                                               |          cta4           |
                                                                               |-------------------------|
                                                                               |          cta5           |
                                                                               |-------------------------|
                                                                               |          cta6           |
                                                                               |-------------------------|
                                                                               |          cta7           |
                                                                               +-------------------------+

    Q is split vertically into 32x576 thread block tiles, C is split across the
    sequence lenght dimension to saturate all SMs (two splits is enough for 16 SMs).
    Note that P tiles are not materialized in gmem. Splits partial results
    are later reduced with a separate kernel.

                                                          512
 - thread block tiles:                            +--------------+-+
                                                  |//////////////| |Cs
                                                  +--------------+-+
                                                  |              | |
                        Cs       seq_len/2        |              | |
                       +--+---------------------+ |              | |
                       |//|                     | |              | | seq_len/2
                       |//|                     | |              | |
                  576  |//|     -->             | |              | |
                       |//|                     | |              | |
                       |//|                     | |              | |
            Qs         +--+---------------------+ +--------------+-+
    +----------------+ +--+---------------------+ +--------------+
 32 |////////////////| |//| Ps  -->             | |//////////////| Or
    +----------------+ +--+---------------------+ +--------------+
           576          16

    each thread block is assigned a tile of Q that spans the entire head
    dimension and iterates over the tiles of a chunk of C.

 - warp tiles:

                                                  16
                                                +----+
                                                |/w0/| 16
                                                +----+
                                                | w1 |
                                                +----+
                                                | w2 |
                                                +----+
                                                | w3 |
                                                +----+ Cs
                                                |/w0/|
                                                +----+
                                                | w1 |
                                                +----+
                                                | .. |                128           Cs
                                                +----+             +-------+-------+-------+-------+--+
                                                | w3 |             |///////|       |       |       |  |
                                                +----+             +-------+-------+-------+-------+--+

    +----+----+----+----+----+----+----+----+   +----+     +----+  +-------+-------+-------+-------+
    |/w0/| w1 | w2 | w3 |/w0/| w1 | .. | w3 |   |/w0/|  =  |////|  |/warp0/| warp1 | warp2 | warp3 |  Or
 32 |////|    |    |    |////|    |    |    |   |////|     |////|  |///////|       |       |       |
    +----+----+----+----+----+----+----+----+   +----+     +----+  +-------+-------+-------+-------+
      16               Qs                         +          Ps                  512  
                                                +----+
                                                | w1 | Pr
                                                |    |
                                                +----+
                                                  +
                                                +----+
                                                | w2 |
                                                |    |
                                                +----+
                                                  +
                                                +----+
                                                | w3 |
                                                |    |
                                                +----+

    the QC^t product is performed as a slice-k gemm, each warp accumulates
    a partial product by performing a 32x144x16 gemm split across 9 32x16x16 slices,
    which is followed by a block wide reduction.


 - mma tiles:
                      8    8                             8        128
                  +----+----+                        +----+----+----+----+----+
                  |////|    |                        |////|    |    |    |    |
         C_frag   |////|    |               C_frag   |////|    |    | .. |    |
                  |////|    |                        |////|    |    |    |    |
                  |////|    |                        |////|    |    |    |    |
                  +----+----+                        +----+----+----+----+----+
       Q_frag                              16
    +---------+   +----+----+          +---------+   +----+----+----+----+----+
    |/////////|   |////|    |          |/////////|   |////|    |    |    |    |
 16 |/////////|   |mma0|mma1|          |/////////|   |mma0|mma2|mma4| .. | mma|
    |/////////|   |////|    |          |/////////|   |////|    |    |    | 31 |
    |/////////|   |////|    |          |/////////|   |////|    |    |    |    |
    +---------+   +----+----+          +---------+   +----+----+----+----+----+
    |         |   |    |    |          |         |   |    |    |    |    |    |
 16 |         |   |mma2|mma3|          |         |   |mma1|mma3|mma5| .. | mma|
    |         |   |    |    |          |         |   |    |    |    |    | 32 |
    |         |   |    |    |          |         |   |    |    |    |    |    |
    +---------+   +----+----+          +---------+   +----+----+----+----+----+
        16          P_frag               P_frag                O_frag

    each warp issues 4 m16n8k16 mma instructions per QC^t slice (36 mmas)
    and 32 m16n8k16 mma instructions for PC.
```

Uses Tensor Cores, beats PyTorch!

## Optimization worklog

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
|pytorch eager (head_dim_v fix)                                              |0.357|48%|
||
| (fused) vectorized C load with unroll 4                                    |1.190|13%|
| + padding Cs to avoid bank conflicts                                       |0.934|16%|
| + 2-stage pipeline to load C                                               |0.795|19%|
| + 3-stage pipeline to load C                                               |0.769|20%|
| + two cols coarsening for PC product                                       |0.664|22%|
| + two cols coarsening for QC^T product [`QTileK=8`]                        |0.641|24%|
| + head_dim_v fix :P                                                        |0.566|25%|
| + `QTileM=CTileN=16` + 4 col coarse PC + O reg tiling + force Ps vec load |*0.440|33%|
| + avoid Qs bank conflicts + avoid Ps store conditional + full Qr tiling   |*0.404|36%|
(*excluding splitkv reduce kernel)

times for fp16

|kernel|time (ms)| peak flops (tops) |
|splitk_mma|..||
|splitk

## Future works

- BF16 support should be fairly straightforward, given the templated nature of the code.
- FP8 quantized KV cache could be a good candidate for Ada, which has native support for FP8.

## Notes

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

-  When accessing Qs, broadcast/multicast 128 bit loads can be served in a minimum of 2 wavefronts. Bank conflicts occur in each half warp (so if threads 0-7 access the same bank as threads 8-15), but not across half warps. At least when each quarter warp is accessing in multicast the same 16 byte word.

- When loading C into Cs using cp.async, since each group of consecutive threads access two different cache lines (2 sectors from each, 2*QTileK spans 64 bytes), the store to shmem for each group of consecutive 8 threads can't be made in a single wavefront, that's why we get 8 wavefronts per LDGSTS.128 (instead of the expected 4, which we get if we access entire cache lines from gmem). The thing is that we don't get conflicts within each group of consecutive 8 threads, so that's the only explanation I can come up with. Remember that 128 bit loads from shmem ideally happen in 4 wavefront usually (when no broadcast/multicast is happening), so one should worry only about conflicts within quarter warps.
Anyhow, removing those conflicts doesn't increase performance currently, it just shifts the stalls somewhere else.

- I should check whether I am already accessing Cs in the best way possible, or if there are any conflicts. Apart from that I'd like to hide better the long scoreboard stalls.

- Can I start prefetching C while performing the PC product? I guess I could let each warp work on the same exact chunk of Cs, it would be easier. So prefetch immediately the (CTileN, 2 * QTileK) tile after having performed the PC product. Perhaps then writing to O would suffer a bit, but I can try.

- Use 4 warps instead of 8, it looks like it suffers less long scoreboard stalls. At that point you can increaseTN to 4. Then perhaps shmem->reg pipelining?
- try 8x8 for PC product at this point

- For shmem 128 bit multicast to require the minimum amount of wavefronts, contiguous threads have to access the same word!

- The best I can get is 2 wavefronts for Cs and 4 wavefronts for Qs or viceversa, multicast requiring contiguity of threads is a pain in the neck. I can't get the 4x4 thread tile for QC^T to perform better than the 8x2, bank conflicts are easier to deal with for the 8x2 version.

- Double buffering Cs and prefetching the entire Cs tile avoids long scoreboard stalls much better than the intra tile pipeline, which was basically being "flushed" for each iteration over the sequence length. The effective CTileN halvens though, I guess it's a good compromise for the fp16 version.

- Pipelining shmem to regs loads doesn't seem to have an effect on scoreboard stalls. The resulting ptx changes, but the SASS looks basically the same, as well as the number of registers used. __syncwarp() or cta.membar can be used to force the ordering between loads and mma loop, but then the MIO throttle stalls spike, so I guess ptxas is already doing some kind of pipelining by exploiting ILP from the unrolled loops. Note however that tiling explicitly an entire fragment into registers before consuming it in the mma loop can still be beneficial.

- Short scoreboard stalls are present even while performing in warp butterfly reduction to compute the softmax, I think having more
rows to process per warp would allow be beneficial in terms of ILP, which can be done for the fp16 version, were the block size is smaller wrt to the Ps tile.

- torch fp16 gemms appear memory bound wrt tensor cores flops. This could give an edge to the fused kernel, at least on a GPU with very limited bandwidth, like the RTX 500 Ada (128GB/s). The fact that the gemms are tall-and-skinny (lower AI) is now becoming apparent due to the higher peak flops of tensor cores. That's probably also why torch in fp32 can't reach more than 50% peak flops. 

- Loading naively with LDS each 8x8 tile for mma and using ldmatrix without any swizzling results in the same number of wavefronts, interestingly though ncu reports 128 bit access size for LDSM instructions, so perhaps not all threads are active during a LDSM instruction? The ISA requires threads 0-7 to provide the ptr to the start of rows 0-7 of the matrix, so perhaps only those threads actually access the shmem, each loading one entire row with a 128 bit access granularity. But what would be the point of ldmatrix then? Each group of 4 contiguous threads could load the corrisponding row in multicast with LDS.128, and then each thread could discard the values that it doesn't need. Perhaps x2 and x4 ldmatrix variants use respectively 16 and 32 threads, if each quarter warp is assigned to a 8x8 matrix. In that case I can see why ldmatrix would be superior, if it still maps to a single SASS instruction.
This intuiton is confirmed by the ptx docs: "When reading 8x8 matrices, a group of four consecutive threads loads 16 bytes." So to benefit from ldmatrix, apart from swizzling, it would be better to use the x4 version." and "When .num = .x2, the elements of the second matrix are loaded in the next destination register in each thread as per the layout in above table. Similarly, when .num = .x4, elements of the third and fourth matrices are loaded in the subsequent destination registers in each thread"

- I mentioned a while ago that smaller Cs tiles do not impact the AI, since the whole C has to be streamed, but they do have an effect on the AI wrt to shmem, since the Q tiles in regs can be reused for more output values, the number of iterations over C is also reduced.

- Lots of IMAD instructions between HMMA, see this: https://forums.developer.nvidia.com/t/the-number-of-imad-instructions-blow-up-after-changing-to-m16n8k16-mma/351316/4

- It could be interesting to see whether not using double buffering (with and without intra-tile cp.async) and having a Cs tile of double the size increases performance in the fp16 version.

- What about multi-stage pipelining for Cs? In that case one could attempt pipelining the QC^T gemm with the softmax computation. Warp specialization could play a role here, since manually interleaving softmax with the next gemm could be troublesome.

- Try to estimate how much peak flops can theoretically be achieved on the rtx 500 ada for the gemms in the mla computation, to have a more meaningful view on the obtained flops.

- Benchmark also with a bigger batch size.

- Compare against flash-attention decoding.

## References

Some useful and/or inspiring resources that I stumbled upon while working on this project.

- https://x.com/karpathy/status/1789666350878601581
- https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://docs.pytorch.org/cppdocs/index.html
- https://pybind11.readthedocs.io/en/stable/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
- https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions
- https://blog.ezyang.com/2019/05/pytorch-internals/
- https://siboehm.com/articles/22/CUDA-MMM
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://www.spatters.ca/mma-matmul
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
- https://gau-nernst.github.io/fa-5090/
- https://salykova.github.io/sgemm-gpu
- https://www.aleksagordic.com/blog/matmul
- https://github.com/NVIDIA/cutlass#documentation
- https://github.com/deepseek-ai/FlashMLA
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
. https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/
- https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/
- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- https://www.youtube.com/watch?v=Y-o545eYjXM
- https://www.youtube.com/@GPUMODE
- https://www.youtube.com/watch?v=0VLAoVGf_74
- https://arxiv.org/abs/1805.02867
- https://arxiv.org/abs/2205.14135
- https://arxiv.org/pdf/2307.08691
- https://arxiv.org/abs/2407.08608
- https://arxiv.org/abs/2405.04434
- https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- https://hazyresearch.stanford.edu/blog/2024-05-12-tk
- https://dl.acm.org/doi/10.1145/1356052.1356053
- https://www.cs.cmu.edu/~mgormley/courses/10423/schedule.html
- https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/
