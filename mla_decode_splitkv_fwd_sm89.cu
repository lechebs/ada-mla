#include <cstdio>

#include "mla_decode.hpp"
#include "mla_decode_splitkv_fwd.cuh"

void run_mla_decode_splitkv(const float *query,
                            const float *cache,
                            float *out,
                            float *splits_max,
                            float *splits_sum,
                            float *splits_out,
                            int num_splits,
                            int num_heads,
                            int head_dim_c,
                            int seq_length)
{
    const int q_tile_m = 8;
    const int q_tile_k = 16;
    const int c_tile_n = 32;

    dim3 block(c_tile_n, q_tile_m);
    dim3 grid(num_splits, (num_heads - 1) / block.y + 1);

    // TODO: Compute size using input args
    int shmem_bytes = (q_tile_m * head_dim_c +
                       c_tile_n * head_dim_c +
                       q_tile_m * c_tile_n) * 4;

    // Dynamic shmem is required for allocations > 48KB
    cudaError_t error = cudaFuncSetAttribute(
        mla_decode_split_kv<q_tile_m, q_tile_k, c_tile_n, 576>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_bytes);

    if (error != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }

    mla_decode_split_kv<q_tile_m, q_tile_k, c_tile_n, 576>
        <<<grid, block, shmem_bytes>>>(
            query, cache, splits_out, splits_max, splits_sum,
            num_heads, seq_length, num_splits);

    //mla_decode_combine<>

    if ((error = cudaGetLastError()) != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }
}
