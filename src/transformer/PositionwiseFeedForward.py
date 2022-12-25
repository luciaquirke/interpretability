from torch import nn


# ## Position-wise Feed-Forward Networks
#
# In addition to attention sub-layers, each of the layers in our
# encoder and decoder contains a fully connected feed-forward network,
# which is applied to each position separately and identically.  This
# consists of two linear transformations with a ReLU activation in
# between.
#
# While the linear transformations are the same across different
# positions, they use different parameters from layer to
# layer. Another way of describing this is as two convolutions with
# kernel size 1.  The dimensionality of input and output is
# $d_{\text{model}}=512$, and the inner-layer has dimensionality
# $d_{ff}=2048$.
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
