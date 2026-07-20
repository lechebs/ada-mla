#include "mla_decode.hpp"
#include "mla_decode_naive_fwd.cuh"

void run_mla_decode_naive(const float *query,
                          const float *cache,
                          float *attn,
                          float *out,
                          int num_heads,
                          int head_dim_c,
                          int seq_length)
{
    dim3 gemm_block(16, 16);
    dim3 gemm_grid((seq_length - 1) / gemm_block.x + 1,
                   (num_heads - 1) / gemm_block.y + 1);

     // Compute attn matrix A = softmax(QK^T)
    mla_decode_naive_gemm<16, 16, 16, true><<<gemm_grid, gemm_block>>>(
        query, cache, attn, num_heads,
        seq_length, head_dim_c, 1.0 / sqrt(head_dim_c));

    constexpr int softmax_block = 256;
    dim3 softmax_grid(1, num_heads); // One block per row
    mla_decode_naive_softmax<softmax_block><<<softmax_grid, softmax_block>>>(
        attn, seq_length);

    gemm_grid.x = (head_dim_c - 1) / gemm_block.x + 1;
    gemm_grid.y = (num_heads - 1) / gemm_block.y + 1;
    // Project values O = AV
    mla_decode_naive_gemm<16, 16, 16, false><<<gemm_grid, gemm_block>>>(
        attn, cache, out, num_heads, head_dim_c, seq_length);
}
