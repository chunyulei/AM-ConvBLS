from torch import nn
import torch


class SPP(nn.Module):
    def __init__(self, os1, os2, os3):
        super(SPP, self).__init__()
        self.os1 = os1
        self.os2 = os2
        self.os3 = os3

        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(os1, os1))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(os2, os2))
        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(os3, os3))

    def forward(self, x):
        x1 = self.pool1(x).reshape(x.shape[0], -1)
        x2 = self.pool2(x).reshape(x.shape[0], -1)
        x3 = self.pool3(x).reshape(x.shape[0], -1)
        return torch.cat([x1, x2, x3], dim=1)

if __name__ == '__main__':
    x = torch.rand((2, 512, 13, 13))
    f = SPP(1, 2, 3)
    print(f)
    out = f(x)
    print(out.shape)
