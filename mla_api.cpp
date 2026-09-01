#include <torch/python.h>

#include "mla_decode.hpp"

at::Tensor mla_decode_naive(const at::Tensor &query,
                            const at::Tensor &cache)
{
    // TODO: Use TORCH_CHECK on sizes, dtype and device
    const c10::IntArrayRef q_size = query.sizes();
    const at::Device dev = query.device();
    const at::ScalarType dtype = query.scalar_type();
    // Set dtype and layout same as input tensors
    at::TensorOptions options =
        at::TensorOptions().device(dev).dtype(dtype);
    at::Tensor out = at::empty(q_size, options);

    float *o_ptr = static_cast<float *>(out.data_ptr());
    // TODO: Perhaps call .contiguous() on the inputs
    const float *q_ptr = static_cast<float *>(query.data_ptr());
    const float *c_ptr = static_cast<float *>(cache.data_ptr());

    // I need to obtain the pointer type based on at::ScalarType

    // Materialize attn matrix
    at::Tensor attn = at::empty({ q_size[0], cache.size(0) }, options);
    float *a_ptr = static_cast<float *>(attn.data_ptr());

    run_mla_decode_naive(
        q_ptr, c_ptr, a_ptr, o_ptr, q_size[0], q_size[1], cache.size(0));

    return out;
}

at::Tensor mla_decode_fused(const at::Tensor &query,
                            const at::Tensor &cache)
{
    // todo: refactor common logic with mla_decode_naive

    const c10::IntArrayRef q_size = query.sizes();
    const at::Device dev = query.device();
    at::TensorOptions options = at::TensorOptions().device(dev);
    // TODO: Fill with zeros inside the kernel
    // NOTE: No need to fill with zeros if pc can be accumulated into registers
    at::Tensor out = at::empty({ q_size[0], q_size[1] - 64 }, options);

    float *o_ptr = static_cast<float *>(out.data_ptr());
    const float *q_ptr = static_cast<float *>(query.data_ptr());
    const float *c_ptr = static_cast<float *>(cache.data_ptr());

    run_mla_decode_fused(
        q_ptr, c_ptr, o_ptr, q_size[0], q_size[1], cache.size(0));

    return out;
}

template<typename Scalar>
void mla_decode_split_(const at::Tensor &query,
                       const at::Tensor &cache,
                       at::Tensor &out,
                       at::TensorOptions options,
                       int num_splits)
{
    Scalar *o_ptr = out.data_ptr<Scalar>();
    const Scalar *q_ptr = query.data_ptr<Scalar>();
    const Scalar *c_ptr = cache.data_ptr<Scalar>();

    int num_heads = query.size(0);
    int head_dim_c = query.size(1);
    int seq_length = cache.size(0);

    at::Tensor splits_max = at::empty({ num_splits, num_heads }, options);
    at::Tensor splits_sum = at::empty({ num_splits, num_heads }, options);
    // Each split needs its own output tensor (note that out could be reused)
    at::Tensor splits_out = at::empty(
        { num_splits, num_heads, head_dim_c - 64 }, options);

    Scalar *splits_max_ptr = splits_max.data_ptr<Scalar>();
    Scalar *splits_sum_ptr = splits_sum.data_ptr<Scalar>();
    Scalar *splits_out_ptr = splits_out.data_ptr<Scalar>();

    run_mla_decode_splitk<Scalar>(
        q_ptr, c_ptr, o_ptr, splits_max_ptr, splits_sum_ptr, splits_out_ptr,
        num_splits, num_heads, seq_length);
}

at::Tensor mla_decode_split(const at::Tensor &query,
                            const at::Tensor &cache)
{
    // todo: refactor common logic with mla_decode_naive

    at::TensorOptions options =
        at::TensorOptions().device(query.device()).dtype(query.scalar_type());
    at::Tensor out = at::empty({ query.size(0), query.size(1) - 64 }, options);

    const int num_splits = 4;

    // dispatch based on dtype
    switch (query.scalar_type()) {
        case at::kFloat:
            mla_decode_split_<float>(query, cache, out, options, num_splits);
            break;
        case at::kHalf:
            mla_decode_split_<at::Half>(
                query, cache, out, options, num_splits);
            break;
        default:
            break;
    }

    return out;
}

PYBIND11_MODULE(ada_mla, m)
{
    m.def("mla_decode_naive", &mla_decode_naive);
    m.def("mla_decode_fused", &mla_decode_fused);
    m.def("mla_decode_split", &mla_decode_split);
}
