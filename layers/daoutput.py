
import numpy as np

import torch
from torch import Tensor
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter


class DAOutput(Module):
    """
    output layer for domain adaptation
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        c: regularization coefficient for L2 norm
        lam2: regularization coefficient for domain adapatation
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 c: float,
                 lam: float
                 ) -> None:

        super(DAOutput, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.lam = lam
        """instantiate the model parameters"""
        self.weight = Parameter(torch.Tensor(
            in_features, out_features), requires_grad=False)
        """initialize the model parameters"""
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
            Xs, Xt, Ys = args
            Ns, Nt = Xs.shape[0], Xt.shape[0]
            Y = torch.vstack([Ys, torch.zeros([Nt, Ys.shape[1]])])

            """MMD matrix"""
            V_00 = torch.ones([Ns, Ns]) / (Ns * Ns)
            V_01 = -torch.ones([Ns, Nt]) / (Ns * Nt)
            V_10 = -torch.ones([Nt, Ns]) / (Nt * Ns)
            V_11 = torch.ones([Nt, Nt]) / (Nt * Nt)
            V = torch.vstack([torch.hstack([V_00, V_01]), torch.hstack([V_10, V_11])]).double()
            """ridge regression"""
            if Ns + Nt > Xs.shape[1]:
                self.pinvOfInput = self.pinv1(Xs, Xt, V, self.c, self.lam)
            else:
                self.pinvOfInput = self.pinv2(Xs, Xt, V, self.c, self.lam)
            weight = self.pinvOfInput @ Y
            self.weight.data = weight

            return Xs @ self.weight
        else:
            """test stage"""
            Xt = args[0]

            return Xt @ self.weight

    def laplacian(self, y):
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

    @staticmethod
    def pinv1(As, At, V, lambda1, lambda3):
        A = torch.vstack([As, At])
        ns, nt = As.shape[0], At.shape[0]
        W_00 = torch.eye(ns)
        W_01 = torch.zeros([ns, nt])
        W_10 = torch.zeros([nt, ns])
        W_11 = torch.zeros([nt, nt])
        W = torch.vstack([torch.hstack([W_00, W_01]), torch.hstack([W_10, W_11])]).double()
        return (A.T @ W @ A + lambda1 * torch.eye(As.shape[1], device=As.device) + lambda3 * A.T @ V @ A).inverse() @ A.T

    @staticmethod
    def pinv2(As, At, V, lambda1, lambda3):
        A = torch.vstack([As, At])
        ns, nt = As.shape[0], At.shape[0]
        W_00 = torch.eye(ns)
        W_01 = torch.zeros([ns, nt])
        W_10 = torch.zeros([nt, ns])
        W_11 = torch.zeros([nt, nt])
        W = torch.vstack([torch.hstack([W_00, W_01]), torch.hstack([W_10, W_11])]).double()
        return A.T @ (W @ A @ A.T + lambda1 * torch.eye(A.shape[0], device=A.device) + lambda3 * V @ A @ A.T).inverse()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, c={}, lam={}'.format(
            self.in_features, self.out_features, self.c, self.lam
        )
