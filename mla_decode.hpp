#pragma once

void run_mla_decode_cpu(const float *query, // (num_heads, head_dim_c)
                        const float *cache, // (seq_length, head_dim_c)
                        float *out,         // (num_heads, head_dim_c)
                        int num_heads,
                        int head_dim_c,
                        int seq_length); // ctx_length

void run_mla_decode_naive(const float *query,
                          const float *cache,
                          float *attn,
                          float *out,
                          int num_heads,
                          int head_dim_c,
                          int seq_length);

void run_mla_decode_fused(const float *query,
                          const float *cache,
                          float *out,
                          int num_heads,
                          int head_dim_c,
                          int seq_length);

template<typename Scalar>
void run_mla_decode_splitkv(const Scalar *query,
                            const Scalar *cache,
                            Scalar *out,
                            Scalar *splits_max,
                            Scalar *splits_sum,
                            Scalar *splits_out,
                            int num_splits,
                            int num_heads,
                            int head_dim_c,
                            int seq_length);
