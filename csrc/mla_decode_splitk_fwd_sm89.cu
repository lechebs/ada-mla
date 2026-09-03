#include <cstdio>

#include "mla_decode.hpp"
#include "mla_decode_splitk_fwd.cuh"

#include <ATen/ATen.h>

template<>
void run_mla_decode_splitk<float>(const float *query,
                                  const float *cache,
                                  float *out,
                                  float *splits_max,
                                  float *splits_sum,
                                  float *splits_out,
                                  int num_splits,
                                  int num_heads,
                                  int seq_length)
{
    /*
    run_mla_decode_splitkv_<float>(
        query, cache, out, splits_max, splits_sum, splits_out,
        num_splits, num_heads, seq_length);
    */
}

template<>
void run_mla_decode_splitk<at::Half>(const at::Half *query,
                                     const at::Half *cache,
                                     at::Half *out,
                                     at::Half *splits_max,
                                     at::Half *splits_sum,
                                     at::Half *splits_out,
                                     int num_splits,
                                     int num_heads,
                                     int seq_length)
{
    run_mla_decode_splitk_fp16(
        reinterpret_cast<const __half *>(query),
        reinterpret_cast<const __half *>(cache),
        reinterpret_cast<__half *>(out),
        reinterpret_cast<__half *>(splits_max),
        reinterpret_cast<__half *>(splits_sum),
        reinterpret_cast<__half *>(splits_out),
        num_splits,
        num_heads,
        seq_length);
}
