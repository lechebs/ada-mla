import math
import torch

from torch.utils.cpp_extension import load

module = load(
    name="ada_mla",
    sources=["ada_mla.cpp"],
    verbose=True)

NUM_HEADS = 128
HEAD_DIM_C = 512
SEQ_LENGTH = 1024

def mla_decode_ref(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ c.T / math.sqrt(HEAD_DIM_C), dim=-1) @ c

q = torch.rand((NUM_HEADS, HEAD_DIM_C))
c = torch.rand((SEQ_LENGTH, HEAD_DIM_C))

out = module.mla_decode(q, c)
out_ref = mla_decode_ref(q, c)

diff = torch.abs(out - out_ref)
print(f"max={torch.max(diff)}, mean={torch.mean(diff)}")
