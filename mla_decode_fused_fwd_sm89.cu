#include <cstdio>

#include "mla_decode.hpp"
#include "mla_decode_fused_fwd.cuh"

void run_mla_decode_fused(const float *query,
                          const float *cache,
                          float *out,
                          int num_heads,
                          int head_dim_c,
                          int seq_length)
{
    // TODO: Should I make head_dim_c known at compile time?

    const int q_tile_m = 16;
    const int q_tile_k = 8;
    const int c_tile_n = 16;

    const int num_stages = 3;

    const int num_sms = 16; // TODO: query device properties

    // NOTE: To check correctness with a different choice
    // of q_tile_m, you can override this to 1.
    const int num_splits = num_sms / (num_heads / q_tile_m);

    // WARNING: Reduce after split is missing!

    dim3 block(c_tile_n, q_tile_m);
    dim3 grid(num_splits, (num_heads - 1) / q_tile_m + 1);

    int num_warps = (block.x * block.y) / WARP_SIZE;

    int shmem_bytes = (q_tile_m * head_dim_c +
                       // +4 padding to avoid bank conflicts
                       // and guarantee float4 alignment (for vector loads)
                       c_tile_n * (head_dim_c + 4) +
                       q_tile_m * c_tile_n * num_warps) * 4;

    // Dynamic shmem is required for allocations > 48KB
    cudaError_t error = cudaFuncSetAttribute(
        mla_decode_fused<q_tile_m, q_tile_k, c_tile_n, 576, num_stages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_bytes);

    if (error != cudaSuccess) {
        printf("cudaFuncSetAttribute: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }

    mla_decode_fused<q_tile_m, q_tile_k, c_tile_n, 576, num_stages>
        <<<grid, block, shmem_bytes>>>(query, cache, out, seq_length);

    // TODO: Wrap error checking in macro
    if ((error = cudaGetLastError()) != cudaSuccess) {
        printf("cudaGetLastError: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }
}
