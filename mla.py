import argparse

import torch

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from torch.utils.cpp_extension import load
from torch.utils.benchmark import Timer, Compare

module = load(
    name="ada_mla",
    sources=[
        "mla_api.cpp",
        "mla_decode_naive_fp32_cpu.cpp",
        "mla_decode_naive_fwd.cu",
        "mla_decode_fused_fwd_sm89.cu",
        "mla_decode_splitk_fwd_sm89.cu",
    ],
    extra_cuda_cflags=["-arch=sm_89", "-lineinfo", "--ptxas-options=-v"],
    verbose=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: These constants should be exported by the C++ module
NUM_HEADS = 128
HEAD_DIM_K = 576
HEAD_DIM_V = 512

mla_decode_naive = module.mla_decode_naive
mla_decode_fused = module.mla_decode_fused
mla_decode_split = module.mla_decode_split

# torch.set_float32_matmul_precision("high") # to use tensor cores for fp32

def mla_decode_torch(q: torch.Tensor,
                     c: torch.Tensor,
                     head_dim_k: int=HEAD_DIM_K,
                     head_dim_v: int=HEAD_DIM_V) -> torch.Tensor:

    return torch.softmax(
        q @ c.T / head_dim_k ** 0.5, dim=-1) @ c[:, :head_dim_v]

def mla_decode_sdpa(q: torch.Tensor,
                    c: torch.Tensor,
                    head_dim_v: int=HEAD_DIM_V) -> torch.Tensor:
    # flash_attn and cuddn attention are not compatible with current head_dim
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION): # xformers (mlsk) triton backend
        # EFFICIENT_ATTENTION is faster on fp16, MATH is faster on fp32
        # TODO: Benchmark against GQA with enable_qga=True
        return scaled_dot_product_attention(
            q[None, None, :, :],
            c[None, None, :, :],
            c[None, None, :, :head_dim_v])

MLA_DECODE_FUNCS = {
    "torch": mla_decode_torch,
    "sdpa": mla_decode_sdpa,
    #"naive": mla_decode_naive,
    "fused": mla_decode_fused,
    "split": mla_decode_split
}

def run_funcs(mla_funcs: list[callable],
              num_heads: int,
              seq_lengths: list[int],
              head_dim_k: int=HEAD_DIM_K,
              dtype: torch.dtype=torch.float32,
              num_iters=1):

    q = torch.normal(0, 1, size=(num_heads, head_dim_k),
                     dtype=dtype).to(DEVICE)

    for seq_len in seq_lengths:
        c = torch.normal(0, 1, size=(seq_len, head_dim_k),
                         dtype=dtype).to(DEVICE)

        for func in mla_funcs:
            for _ in range(num_iters):
                func(q, c)

def compute_error(mla_decode_func: callable,
                  q: torch.Tensor,
                  c: torch.Tensor) -> tuple[float, float]:
    # Uses mla_decode_torch as reference
    out = mla_decode_func(q, c)
    out_ref = mla_decode_torch(q, c)
    diff = torch.abs(out - out_ref)
    return torch.max(diff), torch.mean(diff)

def run_test(mla_funcs: list[callable],
             num_heads: int,
             seq_lengths: list[int],
             head_dim_k: int=HEAD_DIM_K,
             dtype: torch.dtype=torch.float32,
             ref_func: callable=mla_decode_torch):

    q = torch.normal(0, 1, size=(num_heads, head_dim_k),
                     dtype=dtype).to(DEVICE)

    for seq_len in seq_lengths:
        c = torch.normal(0, 1, size=(seq_len, head_dim_k),
                         dtype=dtype).to(DEVICE)

        out_ref = ref_func(q, c)

        for func in mla_funcs:
            diff = torch.abs(func(q, c) - out_ref)
            max_err, mean_err = torch.max(diff), torch.mean(diff)
            print(f"[{func.__name__}] seq_len={seq_len} max={max_err:.6g}"
                  f" mean={mean_err:.6g}")

def run_benchmark(mla_funcs: list[callable],
                  num_heads: int,
                  seq_lengths: list[int],
                  head_dim_k: int=HEAD_DIM_K,
                  dtype: torch.dtype=torch.float32,
                  num_iters: int=100):

    q = torch.normal(
        0, 1, size=(num_heads, head_dim_k), dtype=dtype).to(DEVICE)

    results = []

    for seq_len in seq_lengths:
        c = torch.normal(
            0, 1, size=(seq_len, head_dim_k), dtype=dtype).to(DEVICE)

        for func in mla_funcs:
            t = Timer(
                label=f"seq_len={seq_len}",
                sub_label=func.__name__,
                description="time",
                stmt=f"{func.__name__}(q, c)",
                setup=f"from __main__ import {func.__name__}",
                globals={"q": q, "c": c})

            results.append(t.timeit(number=num_iters))

    c = Compare(results)
    c.colorize()
    print(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mla")

    parser.add_argument("--for-ncu", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    parser.add_argument("-s", "--seq-length", type=int, nargs="+",
                        default=[4096])

    parser.add_argument("--fp16", action="store_true")
    # parser.add_argument("--bf16")

    parser.add_argument("-k", "--kernel", nargs="+",
                        default=MLA_DECODE_FUNCS.keys(),
                        choices=MLA_DECODE_FUNCS.keys())

    # parser.add_argument("--num-heads", default=128)
    # parser.add_argument("--head-dim", default=HEAD_DIM_C)

    args = parser.parse_args()

    chosen_funcs = [MLA_DECODE_FUNCS[name] for name in args.kernel]
    dtype = torch.float16 if args.fp16 else torch.float32

    if args.for_ncu:
        run_funcs(chosen_funcs,
                  num_heads=NUM_HEADS,
                  seq_lengths=args.seq_length,
                  dtype=dtype)

    if args.test:
        #funcs = [f for f in chosen_funcs if f != mla_decode_torch]
        run_test(chosen_funcs,
                 num_heads=NUM_HEADS,
                 seq_lengths=args.seq_length,
                 dtype=dtype)

    if args.benchmark:
        run_benchmark(chosen_funcs,
                      num_heads=NUM_HEADS,
                      seq_lengths=args.seq_length,
                      dtype=dtype,
                      num_iters=50)
