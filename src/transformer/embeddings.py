import math
from torch import nn


# ## Embeddings and Softmax
#
# Similarly to other sequence transduction models, we use learned
# embeddings to convert the input tokens and output tokens to vectors
# of dimension $d_{\text{model}}$.  We also use the usual learned
# linear transformation and softmax function to convert the decoder
# output to predicted next-token probabilities.  In our model, we
# share the same weight matrix between the two embedding layers and
# the pre-softmax linear transformation, similar to
# [(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers,
# we multiply those weights by $\sqrt{d_{\text{model}}}$.
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
