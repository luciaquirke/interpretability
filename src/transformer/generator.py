from torch import nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # proj = project
        # Applies a linear transformation to an input of dimension d_model to create an output of dimension vocab
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)