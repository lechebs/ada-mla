#include <cstdio>

#include "mla_decode.hpp"
#include "mla_decode_splitkv_fwd.cuh"

#include <ATen/ATen.h>

template<>
void run_mla_decode_splitkv<float>(const float *query,
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
    run_mla_decode_splitkv_<float>(
        query, cache, out, splits_max, splits_sum, splits_out,
        num_splits, num_heads, head_dim_c, seq_length);
}

template<>
void run_mla_decode_splitkv<at::Half>(const at::Half *query,
                                      const at::Half *cache,
                                      at::Half *out,
                                      at::Half *splits_max,
                                      at::Half *splits_sum,
                                      at::Half *splits_out,
                                      int num_splits,
                                      int num_heads,
                                      int head_dim_c,
                                      int seq_length)
{
    run_mla_decode_splitkv_<__half>(
        reinterpret_cast<const __half *>(query),
        reinterpret_cast<const __half *>(cache),
        reinterpret_cast<__half *>(out),
        reinterpret_cast<__half *>(splits_max),
        reinterpret_cast<__half *>(splits_sum),
        reinterpret_cast<__half *>(splits_out),
        num_splits,
        num_heads,
        head_dim_c,
        seq_length);
}
