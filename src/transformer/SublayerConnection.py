from torch import nn
from src.transformer.LayerNorm import LayerNorm


# The output of each sub-layer is $\mathrm{LayerNorm}(x +
# \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function
# implemented by the sub-layer itself.  We apply dropout
# [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the
# output of each sub-layer, before it is added to the sub-layer input
# and normalized.
#
# To facilitate these residual connections, all sub-layers in the
# model, as well as the embedding layers, produce outputs of dimension
# $d_{\text{model}}=512$.
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
