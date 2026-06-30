#include <torch/python.h>

void mla_decode_cpu(const float *query, // (num_heads, head_dim_c)
                    const float *cache, // (seq_length, head_dim_c)
                    float *out,         // (num_heads, head_dim_c)
                    const int num_heads,
                    const int head_dim_c,
                    const int seq_length) // ctx_length
{
    // Holds attn matrix
    float *tmp = new float[num_heads * seq_length];

    // Perform QK^T/sqrt(d)
    for (int i = 0; i < num_heads; ++i) {
        for (int j = 0; j < seq_length; ++j) {
            tmp[i * seq_length + j] = 0;
            for (int k = 0; k < head_dim_c; ++k) {
                /* Note that K gets transposed. */
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

    // Project V using computed weights.
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

at::Tensor mla_decode(const at::Tensor &query,
                      const at::Tensor &cache)
{
    // TODO: Use TORCH_CHECK on sizes, dtype and device
    c10::IntArrayRef q_size = query.sizes();
    at::Tensor out = at::empty(q_size);

    float *o_ptr = static_cast<float *>(out.data_ptr());
    // TODO: Perhaps call .contiguous() on the inputs
    const float *q_ptr = static_cast<float *>(query.data_ptr());
    const float *c_ptr = static_cast<float *>(cache.data_ptr());

    mla_decode_cpu(q_ptr, c_ptr, o_ptr, q_size[0], q_size[1], cache.size(0));

    return out;
}

PYBIND11_MODULE(ada_mla, m)
{
    m.def("mla_decode", &mla_decode);
}
