import torch


# If Optimizer had an abstract base class we would inherit that instead
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        # lr = learning rate
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None