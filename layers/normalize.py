import torch

from torch import nn
from torch.nn import Parameter


class Normalize(nn.Module):
    def __init__(self, dim, unbiased=True, epsilon=0.00015):
        super(Normalize, self).__init__()
        self.dim = dim
        self.unbiased = unbiased
        self.epsilon = epsilon

    def forward(self, x):
        if self.training:
            """train stage"""
            self.mean_ = Parameter(torch.mean(x, dim=self.dim, keepdim=True), requires_grad=False)
            self.std_ = Parameter(torch.sqrt(torch.var(x, dim=self.dim, unbiased=self.unbiased, keepdim=True) + self.epsilon), requires_grad=False)
        else:
            """test stage"""
        transformed_x = (x - self.mean_) / self.std_
        return transformed_x
