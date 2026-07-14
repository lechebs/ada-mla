import math
import torch

from torch.utils.cpp_extension import load
from torch.utils.benchmark import Timer

module = load(
    name="ada_mla",
    sources=["ada_mla.cu"],
    extra_cuda_cflags=["-arch=sm_89"],
    verbose=True)

NUM_HEADS = 128
HEAD_DIM_C = 512
SEQ_LENGTH = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mla_decode_ref(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ c.T / math.sqrt(HEAD_DIM_C), dim=-1) @ c

q = torch.rand((NUM_HEADS, HEAD_DIM_C)).to(DEVICE)
c = torch.rand((SEQ_LENGTH, HEAD_DIM_C)).to(DEVICE)

out = module.mla_decode(q, c)
out_ref = mla_decode_ref(q, c)

diff = torch.abs(out - out_ref)
print(f"max={torch.max(diff)}, mean={torch.mean(diff)}")

t_ada = Timer(
    stmt="module.mla_decode(q, c)",
    setup="from __main__ import module",
    globals={"q": q, "c": c})

t_ref = Timer(
    stmt="mla_decode_ref(q, c)",
    setup="from __main__ import mla_decode_ref",
    globals={"q": q, "c": c})

print(t_ada.timeit(100))
print(t_ref.timeit(100))


