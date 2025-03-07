
import numpy as np

import torch
from torch import Tensor
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter


class GROutput(Module):
    """
    output layer for graph regularization
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        c: regularization coefficient for L2 norm
        lam: regularization coefficient for graph regularization
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                in_features: int, 
                out_features: int, 
                c: float, 
                lam: float,
                ) -> None:
        
        super(GROutput, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.lam = lam
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, *args):
        """
        forward calculation of the output layer
        :param args: tensor tuple (x, y) training / x testing
        :return: tensor
        """
        if self.training:
            x, y = args
            L = self.laplacian_1(y)
            """ridge regression"""
            if x.shape[0] > x.shape[1]:
                self.pinvOfInput = self.pinv1(x, L, self.c, self.lam)
            else:
                self.pinvOfInput = self.pinv2(x, L, self.c, self.lam)
            weight = self.pinvOfInput @ y
            self.weight.data = weight
        else:
            """test stage"""
            x = args[0]

        return x @ self.weight
    
    def laplacian_1(self, y):
        '''
        Compute graph laplacian from labels y.
        Parameters
        ----------
        y : ndarray of shape (N,)
            The labels for X, where N is the number of instances.
        Returns
        -------
        L : ndarray of shape (N, N)
            The graph laplacian, where N is the number of instances.
        '''
        y = y.detach().numpy()
        y = np.argmax(y, axis=1)
        y_ = y[np.newaxis]
        W = np.zeros([len(y), len(y)])
        for c in np.unique(y):
            y_c = y_ == c
            W += np.dot(np.transpose(y_c), y_c)/np.sum(y_c)
            # W += np.dot(np.transpose(y_c), y_c)
        D = np.diag(np.sum(W, axis=1))  

        return torch.from_numpy(D-W)
    
    def laplacian_2(self, y):
        '''
        Compute graph laplacian from labels y.
        Parameters
        ----------
        y : ndarray of shape (N,)
            The labels for X, where N is the number of instances.
        Returns
        -------
        L : ndarray of shape (N, N)
            The graph laplacian, where N is the number of instances.
        '''
        y = y.detach().numpy()
        y = np.argmax(y, axis=1)
        y_ = y[np.newaxis]
        W = np.zeros([len(y), len(y)])
        for c in np.unique(y):
            y_c = y_ == c
            # W += np.dot(np.transpose(y_c), y_c)/np.sum(y_c)
            W += np.dot(np.transpose(y_c), y_c)
        W[W == 0] = -1
        D = np.diag(np.sum(W, axis=1))  

        return torch.from_numpy(D-W)

    @staticmethod
    def pinv1(A, L, lambda1, lambda2):
        return (A.T @ A + lambda1 * torch.eye(A.shape[1], device=A.device) + lambda2 * A.T @ L @ A).inverse() @ A.T

    @staticmethod
    def pinv2(A, L, lambda1, lambda2):
        return A.T @ (A @ A.T + lambda1 * torch.eye(A.shape[0], device=A.device) + lambda2 * L @ A @ A.T).inverse()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, c={}, lam={}'.format(
            self.in_features, self.out_features, self.c, self.lam
        )
