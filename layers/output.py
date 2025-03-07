import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class Output(Module):
    """
    output layer for ridge regression
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        c: regularization coefficient
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, c: float) -> None:
        super(Output, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

    def forward(self, *args):
        """
        forward calculation of the output layer
        :param args: tensor tuple (x, y) training / x testing
        :return: tensor
        """
        if self.training:
            x, y = args
            """ridge regression"""
            pinv = (self.c * torch.eye(x.shape[1], device=x.device) + x.T @ x).inverse() @ x.T
            self.weight = Parameter(pinv @ y, requires_grad=False)
        else:
            """test stage"""
            x = args[0]

        return x @ self.weight

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
