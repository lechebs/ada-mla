import math
import torch

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from torch.utils.cpp_extension import load
from torch.utils.benchmark import Timer

module = load(
    name="ada_mla",
    sources=["ada_mla.cu"],
    extra_cuda_cflags=["-arch=sm_89"],
    verbose=True)

NUM_HEADS = 128
HEAD_DIM_C = 576
SEQ_LENGTH = 4096

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# torch.set_float32_matmul_precision("high") # to use tensor cores

def mla_decode_torch(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ c.T / math.sqrt(HEAD_DIM_C), dim=-1) @ c

def mla_decode_sdpa(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # flash_attn and cuddn attention are not compatible with current head_dim
    with sdpa_kernel(SDPBackend.MATH):
        return scaled_dot_product_attention(q[None, None, :, :],
                                            c[None, None, :, :],
                                            c[None, None, :, :])

# TODO: Benchmark with different SEQ_LENGTH values

q = torch.normal(0, 1, size=(NUM_HEADS, HEAD_DIM_C)).to(DEVICE)
c = torch.normal(0, 1, size=(SEQ_LENGTH, HEAD_DIM_C)).to(DEVICE)

out_naive = module.mla_decode_naive(q, c)
out_fused = module.mla_decode_fused(q, c)
out_torch = mla_decode_torch(q, c)

diff_naive = torch.abs(out_naive - out_torch)
diff_fused = torch.abs(out_fused - out_torch)
print(f"[naive] max={torch.max(diff_naive)}, mean={torch.mean(diff_naive)}")
print(f"[fused] max={torch.max(diff_fused)}, mean={torch.mean(diff_fused)}")

t_naive = Timer(
    stmt="module.mla_decode_naive(q, c)",
    setup="from __main__ import module",
    globals={"q": q, "c": c})

t_fused = Timer(
    stmt="module.mla_decode_fused(q, c)",
    setup="from __main__ import module",
    globals={"q": q, "c": c})

t_torch = Timer(
    stmt="mla_decode_torch(q, c)",
    setup="from __main__ import mla_decode_torch",
    globals={"q": q, "c": c})

t_sdpa = Timer(
    stmt="mla_decode_sdpa(q, c)",
    setup="from __main__ import mla_decode_sdpa",
    globals={"q": q, "c": c})

print(t_torch.timeit(100))
print(t_sdpa.timeit(100))
print(t_naive.timeit(100))
print(t_fused.timeit(100))
