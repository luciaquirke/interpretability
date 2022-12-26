import copy
import torch
from torch import nn
from typing import List, TypeVar


T = TypeVar("T")


def clones(module: T, N: int) -> List[T]:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# We also modify the self-attention sub-layer in the decoder stack to
# prevent positions from attending to subsequent positions.  This
# masking, combined with fact that the output embeddings are offset by
# one position, ensures that the predictions for position $i$ can
# depend only on the known outputs at positions less than $i$.
def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
