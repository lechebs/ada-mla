#include <torch/python.h>

#include <cmath>

#include "mla_decode_naive.cuh"
#include "mla_decode_fused.cuh"
//#include "mla_decode_split_kv.cuh"

void mla_decode_cpu(const float *query, // (num_heads, head_dim_c)
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

void run_mla_decode_naive(const float *query,
                          const float *cache,
                          float *out,
                          const at::Device &device,
                          int num_heads,
                          int head_dim_c,
                          int seq_length)
{
    // Materialize attn matrix
    auto options = at::TensorOptions().device(device);
    at::Tensor attn_mat = at::empty({ num_heads, seq_length }, options);
    float *attn = static_cast<float *>(attn_mat.data_ptr());

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

void run_mla_decode_fused(const float *query,
                          const float *cache,
                          float *out,
                          int num_heads,
                          int head_dim_c,
                          int seq_length)
{
    dim3 block(16, 16);
    dim3 grid(1, (num_heads - 1) / 32 + 1);

    int shmem_bytes = 92160 + 1024;

    // Dynamic shmem is required for allocations > 48KB
    cudaFuncSetAttribute(mla_decode_fused<32, 16, 8, 576>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_bytes);

    mla_decode_fused<32, 16, 8, 576><<<grid, block, shmem_bytes>>>(
        query, cache, out, num_heads, seq_length);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }

    // Why is this not needed?
    //cudaDeviceSynchronize();
}

at::Tensor mla_decode_naive_api(const at::Tensor &query,
                                const at::Tensor &cache)
{
    // TODO: Use TORCH_CHECK on sizes, dtype and device
    const c10::IntArrayRef q_size = query.sizes();
    const at::Device dev = query.device();
    // Set dtype and layout same as input tensors
    at::TensorOptions options = at::TensorOptions().device(dev);
    at::Tensor out = at::empty(q_size, options);

    float *o_ptr = static_cast<float *>(out.data_ptr());
    // TODO: Perhaps call .contiguous() on the inputs
    const float *q_ptr = static_cast<float *>(query.data_ptr());
    const float *c_ptr = static_cast<float *>(cache.data_ptr());

    // I need some kind of JIT if I want to specialize kernels at runtime

    //mla_decode_cpu(q_ptr, c_ptr, o_ptr, q_size[0], q_size[1], cache.size(0));

    run_mla_decode_naive(
        q_ptr, c_ptr, o_ptr, dev, q_size[0], q_size[1], cache.size(0));

    return out;
}

at::Tensor mla_decode_fused_api(const at::Tensor &query,
                                const at::Tensor &cache)
{
    // TODO: Refactor common logic with mla_decode_naive

    const c10::IntArrayRef q_size = query.sizes();
    const at::Device dev = query.device();
    at::TensorOptions options = at::TensorOptions().device(dev);
    at::Tensor out = at::empty(q_size, options);

    float *o_ptr = static_cast<float *>(out.data_ptr());
    const float *q_ptr = static_cast<float *>(query.data_ptr());
    const float *c_ptr = static_cast<float *>(cache.data_ptr());

    run_mla_decode_fused(
        q_ptr, c_ptr, o_ptr, q_size[0], q_size[1], cache.size(0));

    return out;
}

PYBIND11_MODULE(ada_mla, m)
{
    m.def("mla_decode_naive", &mla_decode_naive_api);
    m.def("mla_decode_fused", &mla_decode_fused_api);
}
