#include <cmath>

#include "mla_decode.hpp"

void run_mla_decode_cpu(const float *query, // (num_heads, head_dim_c)
                        const float *cache, // (seq_length, head_dim_c)
                        float *out,         // (num_heads, head_dim_c)
                        int num_heads,
                        int head_dim_c,
                        int seq_length) // ctx_length
{
    // Holds attn matrix
    float *tmp = new float[num_heads * seq_length];

    // Perform QK^T/sqrt(d)
    for (int i = 0; i < num_heads; ++i) {
        for (int j = 0; j < seq_length; ++j) {
            tmp[i * seq_length + j] = 0;
            for (int k = 0; k < head_dim_c; ++k) {
                // Note that K gets transposed
                tmp[i * seq_length + j] += query[i * head_dim_c + k] *
                                           cache[j * head_dim_c + k];
            }
            tmp[i * seq_length + j] /= sqrt(head_dim_c);
        }
    }

    // Compute row-wise softmax (without causal masking)
    // TODO: Use online softmax
    for (int i = 0; i < num_heads; ++i) {
        // Logits are translated by their max for stability
        float max = tmp[i * seq_length];
        for (int j = 0; j < seq_length; ++j) {
            float val = tmp[i * seq_length + j];
            if (val > max) {
                max = val;
            }
        }
        float sum = 0;
        for (int j = 0; j < seq_length; ++j) {
            sum += exp(tmp[i * seq_length + j] - max);
        }
        for (int j = 0; j < seq_length; ++j) {
            tmp[i * seq_length + j] =
                exp(tmp[i * seq_length + j] - max) / sum;
        }
    }

    // Project V using computed weights
    for (int i = 0; i < num_heads; ++i) {
        for (int j = 0; j < head_dim_c; ++j) {
            out[i * head_dim_c + j] = 0;
            for (int k = 0; k < seq_length; ++k) {
                out[i * head_dim_c + j] += tmp[i * seq_length + k] *
                                           cache[k * head_dim_c + j];
            }
        }
    }

    delete tmp;
}
