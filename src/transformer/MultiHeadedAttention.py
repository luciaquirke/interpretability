import math
import torch
import torch.nn as nn
from src.utils.utils import clones


# ### Attention
#
# An attention function can be described as mapping a query and a set
# of key-value pairs to an output, where the query, keys, values, and
# output are all vectors.  The output is computed as a weighted sum of
# the values, where the weight assigned to each value is computed by a
# compatibility function of the query with the corresponding key.
#
# We call our particular attention "Scaled Dot-Product Attention".
# The input consists of queries and keys of dimension $d_k$, and
# values of dimension $d_v$.  We compute the dot products of the query
# with all keys, divide each by $\sqrt{d_k}$, and apply a softmax
# function to obtain the weights on the values.
#
#
#
# ![](images/ModalNet-19.png)

#
# In practice, we compute the attention function on a set of queries
# simultaneously, packed together into a matrix $Q$.  The keys and
# values are also packed together into matrices $K$ and $V$.  We
# compute the matrix of outputs as:
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$


#
# The two most commonly used attention functions are additive
# attention [(cite)](https://arxiv.org/abs/1409.0473), and dot-product
# (multiplicative) attention.  Dot-product attention is identical to
# our algorithm, except for the scaling factor of
# $\frac{1}{\sqrt{d_k}}$. Additive attention computes the
# compatibility function using a feed-forward network with a single
# hidden layer.  While the two are similar in theoretical complexity,
# dot-product attention is much faster and more space-efficient in
# practice, since it can be implemented using highly optimized matrix
# multiplication code.
#
#
# While for small values of $d_k$ the two mechanisms perform
# similarly, additive attention outperforms dot product attention
# without scaling for larger values of $d_k$
# [(cite)](https://arxiv.org/abs/1703.03906). We suspect that for
# large values of $d_k$, the dot products grow large in magnitude,
# pushing the softmax function into regions where it has extremely
# small gradients (To illustrate why the dot products get large,
# assume that the components of $q$ and $k$ are independent random
# variables with mean $0$ and variance $1$.  Then their dot product,
# $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance
# $d_k$.). To counteract this effect, we scale the dot products by
# $\frac{1}{\sqrt{d_k}}$.
#
# ![](images/ModalNet-20.png)


# Multi-head attention allows the model to jointly attend to
# information from different representation subspaces at different
# positions. With a single attention head, averaging inhibits this.
#
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# Where the projections are parameter matrices $W^Q_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in
# \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in
# \mathbb{R}^{hd_v \times d_{\text{model}}}$.
#
# In this work we employ $h=8$ parallel attention layers, or
# heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$. Due
# to the reduced dimension of each head, the total computational cost
# is similar to that of single-head attention with full
# dimensionality.
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # -1 refers to the last dimension of query
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
