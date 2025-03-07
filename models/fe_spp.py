import torch
import utils

import torch.nn as nn

from layers.conv_enhanment import Conv2d_Enhancement
from layers.conv_parallel import Conv2d_Parallel
from layers.normalize import Normalize
from layers.spp import SPP

from torch.nn import functional as F



"""feature extractor with spatial pyramid pooling, parallel and single feature layer (with torch.save() function)"""


class FE_SPP_SFL_Parallel(nn.Module):
    def __init__(
            self, 
            in_channels, 
            N1, 
            N2, 
            N3, 
            KS, 
            activation,
            verbose
            ):
        super(FE_SPP_SFL_Parallel, self).__init__()

        """Convolutional Feature Layer"""
        self.cf = nn.ModuleList()
        for i in range(N2):
            self.cf.add_module('CF_{}'.format(i+1), Conv2d_Parallel(in_channels, N1, KS, padding='same', activation=activation, verbose=verbose))

        """Convolutional Enhancement Layer"""
        self.ce = nn.Sequential(
            Conv2d_Enhancement(N1 * N2, N3, kernel_size=(3, 3), padding='same'),
            nn.Tanh()
        )
        """Spatial Pyramid Pooling"""
        self.spp = SPP(1, 2, 3)

        """normalize layer"""
        self.normalize = Normalize(dim=0, epsilon=0.01)

    def forward(self, x):
        cf_out = None
        for idx, m in enumerate(self.cf):
            if idx == 0:
                cf_out = m(x)
            else:
                cf_out = torch.cat([cf_out, m(x)], dim=1)
        
        ce_out = self.ce(cf_out)
        """TSMS"""
        tsms_fea = self.spp(torch.cat([cf_out, ce_out], dim=1))
        """Z-SCORE Normalization"""
        out = self.normalize(tsms_fea).double()

        return out
