import argparse

import torch

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from torch.utils.cpp_extension import load
from torch.utils.benchmark import Timer, Compare

import triton.testing

module = load(
    name="ada_mla",
    sources=[
        "csrc/mla_api.cpp",
        "csrc/mla_decode_naive_fp32_cpu.cpp",
        "csrc/mla_decode_naive_fwd.cu",
        "csrc/mla_decode_fused_fwd_sm89.cu",
        "csrc/mla_decode_splitk_fwd_sm89.cu",
    ],
    extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                       "-gencode=arch=compute_80,code=sm_80", # supports Ampere
                       "-lineinfo"],
    verbose=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: These constants should be exported by the C++ module
NUM_HEADS = 128
HEAD_DIM = 128
HEAD_DIM_C = 512
HEAD_DIM_ROPE = 64
HEAD_DIM_K = HEAD_DIM_C + HEAD_DIM_ROPE

mla_decode_naive = module.mla_decode_naive
mla_decode_fused = module.mla_decode_fused
mla_decode_split = module.mla_decode_split

# torch.set_float32_matmul_precision("high") # uses tensor cores for torch fp32

torch.manual_seed(42)

@torch.compile
def mla_decode_torch(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ c.T / ((HEAD_DIM + HEAD_DIM_ROPE) ** 0.5),
                         dim=-1) @ c[:, :HEAD_DIM_C]

@torch.compile
def mla_decode_sdpa(q: torch.Tensor,
                    c: torch.Tensor,
                    # xformers (mlsk) triton backend
                    backend=SDPBackend.EFFICIENT_ATTENTION) -> torch.Tensor:
    # flash_attn and cuddn attention are not compatible with current head_dim
    with sdpa_kernel(backend):
        # EFFICIENT_ATTENTION is faster on fp16, MATH is faster on fp32
        # TODO: Benchmark against GQA with enable_qga=True
        return scaled_dot_product_attention(
            q[None, None, :, :],
            c[None, None, :, :],
            c[None, None, :, :HEAD_DIM_C],
            scale=1.0 / ((HEAD_DIM + HEAD_DIM_ROPE) ** 0.5))

mla_decode_sdpa_efficient = mla_decode_sdpa
mla_decode_sdpa_math = \
    lambda q, c: mla_decode_sdpa(q, c, backend=SDPBackend.MATH)

MLA_DECODE_FUNCS = {
    "torch": mla_decode_torch,
    "sdpa-math": mla_decode_sdpa_math,
    "sdpa-efficient": mla_decode_sdpa_efficient,
}

def run_funcs(mla_func_names: list[str],
              num_heads: int,
              seq_lengths: list[int],
              head_dim_k: int=HEAD_DIM_K,
              dtype: torch.dtype=torch.float32,
              num_iters=1):

    q = torch.randn(size=(num_heads, head_dim_k), dtype=dtype).to(DEVICE)

    for seq_len in seq_lengths:
        c = torch.randn(size=(seq_len, head_dim_k), dtype=dtype).to(DEVICE)

        for func_name in mla_func_names:
            func = MLA_DECODE_FUNCS[func_name]
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

def run_test(mla_func_names: list[str],
             num_heads: int,
             seq_lengths: list[int],
             head_dim_k: int=HEAD_DIM_K,
             dtype: torch.dtype=torch.float32,
             ref_func: callable=mla_decode_torch):

    q = torch.randn(size=(num_heads, head_dim_k), dtype=dtype).to(DEVICE)

    for seq_len in seq_lengths:
        c = torch.randn(size=(seq_len, head_dim_k), dtype=dtype).to(DEVICE)

        out_ref = ref_func(q, c)

        for func_name in mla_func_names:
            func = MLA_DECODE_FUNCS[func_name]
            diff = torch.abs(func(q, c) - out_ref)
            max_err, mean_err = torch.max(diff), torch.mean(diff)
            rel_mean_err = mean_err / torch.mean(torch.abs(out_ref))
            print(f"[{func_name}] seq_len={seq_len} max={max_err:.6g}"
                  f" mean={mean_err:.6g} rel={rel_mean_err:.6f}")

def triton_do_bench_mla(seq_len: int,
                        kernel_name: str,
                        dtype: torch.dtype) -> float:
    q = torch.randn((NUM_HEADS, HEAD_DIM_K), dtype=dtype, device=DEVICE)
    c = torch.randn((seq_len, HEAD_DIM_K), dtype=dtype, device=DEVICE)

    func = lambda: MLA_DECODE_FUNCS[kernel_name](q, c)

    return triton.testing.do_bench(func)

def run_benchmark(kernel_names: list[str],
                  seq_lengths: list[int],
                  dtype: torch.dtype):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=seq_lengths,
            line_arg="kernel_name",
            line_names=kernel_names,
            line_vals=kernel_names,
            ylabel="ms",
            plot_name="mla-bench",
            args={}
        )
    )
    def benchmark(seq_len: int, kernel_name: str):
        return triton_do_bench_mla(seq_len, kernel_name, dtype)

    benchmark.run(print_data=True, show_plots=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mla")

    parser.add_argument("--for-ncu", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")

    parser.add_argument("-s", "--seq-length", type=int, nargs="+",
                        default=[2048, 4096, 8192, 16384, 32768])

    parser.add_argument("--fp32", action="store_true")
    # parser.add_argument("--bf16")

    kernel_names = ["torch", "sdpa-math", "sdpa-efficient", "fused"]

    parser.add_argument("-k", "--kernel", nargs="+",
                        default=kernel_names,
                        choices=kernel_names)

    args = parser.parse_args()

    if args.fp32:
        dtype = torch.float32
        MLA_DECODE_FUNCS["fused"] = mla_decode_fused
    else:
        dtype = torch.float16
        MLA_DECODE_FUNCS["fused"] = mla_decode_split

    if args.for_ncu:
        run_funcs(args.kernel,
                  num_heads=NUM_HEADS,
                  seq_lengths=args.seq_length,
                  dtype=dtype)

    if args.test:
        run_test(args.kernel,
                 num_heads=NUM_HEADS,
                 seq_lengths=args.seq_length,
                 dtype=dtype)

    if args.bench:
        run_benchmark(args.kernel, args.seq_length, dtype=dtype)

