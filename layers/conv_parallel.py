"""
Convolutional Feature Layer
"""
import math
from requests import patch

import torch
import utils

from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from torch.nn.common_types import _size_2_t
from typing import Union
from torch.nn.modules.utils import _pair

import numpy as np
from progress.bar import IncrementalBar, PixelBar


class Conv2d_Parallel(Module):
    def __init__(
            self,
            in_channels: int,       # number of input channels
            out_channels: int,      # number of output channels
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,  # TODO: refine this type
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,  # TODO: refine this type
            groups: int = 1,   # using group convolution or not (default: 1 means no group convolution)
            bias: bool = False,
            padding_mode: str = 'zeros',  # TODO: refine this type
            pooling: bool = False,
            pooling_mode: str = 'average',
            activation: str = 'relu',
            verbose: bool = False
    ) -> None:
        super(Conv2d_Parallel, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            # if padding == 'same' and any(s != 1 for s in stride):
            #     raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        valid_pooling_modes = {'average', 'max', 'quarter'}
        if pooling_mode not in valid_pooling_modes:
            raise ValueError("pooling_mode must be one of {}, but got pooling_mode='{}'".format(
                valid_pooling_modes, pooling_mode))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.pooling = pooling
        self.pooling_mode = pooling_mode
        self.activation = activation
        self.verbose = verbose   # logging or not

        self.epoch = 10         # use for k-means algorithm
        self.epsilon = 0.00015  # hyper-parameter for normalization
        self.regu = 0.01        # hyper-parameter for ZCA whitening

        self.weight = Parameter(
            torch.randn(self.out_channels, self.in_channels // groups, self.kernel_size[0], self.kernel_size[1]),
            requires_grad=False)  # weights of the first convolutional layer

        if bias:
            self.bias = Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        self.reset_parameters()  # initialize weights and bias

        # TODO: revise to model paramater(PyTorch)
        self.zca = utils.ZCA_parallel(regularization=self.regu)

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def patches_preprocessing_parallel(self, patches, training=False, parallel=False, index=0):
        """
        normalize and whitening
        shape: (num_patches, groups, channels_per_group, height, width)
        """
        patches = utils.normalize(patches, dim=0, epsilon=self.epsilon)
        if training:
            patches = self.zca.fit_transform(patches)
        else:
            patches = self.zca.transform(patches, parallel, index)
        return patches

    def extract_random_patches_parallel(self, num_patches, rf_size, input):
        n_samples, channels, height, width = input.shape
        patches = torch.zeros([num_patches, channels, rf_size[0], rf_size[1]]).to(input.device)
        if self.verbose:
            bar = PixelBar('extract random patches', max = num_patches)
        for idx in range(num_patches):
            r = torch.randint(0, height - rf_size[0], (1,))
            c = torch.randint(0, width - rf_size[1], (1,))
            patch = input[idx % n_samples, :, r:r + rf_size[0], c:c + rf_size[1]]
            # to do list cv2.transpose(patch)
            patches[idx] = patch
            if self.verbose:
                bar.next()
        if self.verbose:
            bar.finish()
        patches = patches.reshape(num_patches, self.groups, self.in_channels // self.groups, *rf_size)
        return patches


    def run_kmeans_parallel(self, patches):
        num_patches, groups, channels_per_group, height, width = patches.shape
        centroids = torch.randn([groups, self.out_channels // groups, channels_per_group * height * width],
                                device=patches.device) * 0.1
        batch_size = 1000  # batch size for k-means
        if self.verbose:
            bar = PixelBar('training weights of CFLs using K-Means', max = self.epoch)
        for _ in range(self.epoch):
            c2 = 0.5 * torch.sum(torch.pow(centroids, 2), dim=2)
            summation = torch.zeros([groups, self.out_channels // groups, channels_per_group * height * width],
                                    device=patches.device)
            counts = torch.zeros([groups, self.out_channels // groups, 1], device=patches.device)
            for i in range(0, num_patches, batch_size):
                last_index = min(i + batch_size, num_patches)
                m = last_index - i  # current batch size
                matrix = torch.matmul(centroids, patches[i:last_index].view(m, groups, -1).permute(1, 0, 2).transpose(2,
                                                                                                                      1)) - c2.unsqueeze(
                    -1)
                val, labels = matrix.max(1)
                """clustering"""
                S = torch.zeros([groups, m, self.out_channels // groups], device=patches.device)
                for idx, st in enumerate(S):
                    pos = (np.array(range(m)), labels[idx])
                    S[idx][pos] = 1
                summation += S.transpose(2, 1) @ patches[i:last_index].permute(1, 0, 2, 3, 4).view(groups, m, -1)
                counts += torch.sum(S, dim=1, keepdim=True).transpose(2, 1)
            centroids = summation / counts
            """
            assert empty clusters exist or not
            """
            bad_index = torch.where(counts == 0)[1]
            centroids[:, bad_index, :] = 0
            if self.verbose:
                bar.next()
        centroids = centroids.reshape(self.out_channels, channels_per_group, height, width)
        if self.verbose:
            bar.finish()
        return centroids

    def _conv_forward_parallel(self, data_x, weight):
        """
        :param data_x: (n_sample, channels, height, width)
        :param weight: (in_channels, out_channels, kernel_height, kernel_width)
        :return: Tensor
        """
        n_sample, channels, height, width = data_x.shape
        out_channels, in_channels, kernel_height, kernel_width = weight.shape

        batch_size = 500
        if self.verbose:
            bar = PixelBar('extract CFL features', max=len(range(0, n_sample, batch_size)))
        for idx in range(0, n_sample, batch_size):
            last_index = min(idx + batch_size, n_sample)
            cur_batch_size = last_index - idx

            batch_x = data_x[idx:last_index]

            if self.padding == 'same':
                feature_map_size = (height, width)
                """padding"""
                # padding = ((height - 1) * self.stride[0] - height + kernel_height) / 2
                padding = ((height - 1) - height + kernel_height) / 2
                batch_x = F.pad(batch_x, (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding)))
            elif self.padding == 'valid':
                h = (height - kernel_height) // self.stride + 1
                w = (width - kernel_width) // self.stride + 1
                feature_map_size = (h, w)
            else:
                raise ValueError('Invalid padding string')
            """unfold input data according to kernel size and stride"""
            patches = F.unfold(input=batch_x, kernel_size=(kernel_height, kernel_width), stride=self.stride).float()  # (500, 108, 729)

            patches = patches.reshape(cur_batch_size, self.groups, in_channels * kernel_height * kernel_width,
                                      -1).permute(1, 0, 2, 3)

            """normalize and whitening"""
            patches = patches.permute(0, 1, 3, 2).reshape(self.groups, -1, in_channels, kernel_height,
                                                          kernel_width).permute(1, 0, 2, 3, 4)
            patches = self.patches_preprocessing_parallel(patches, parallel=True)
            patches = patches.permute(1, 0, 2, 3, 4)
            patches = patches.reshape(self.groups, cur_batch_size, -1,
                                      in_channels * kernel_height * kernel_width).permute(0, 1, 3, 2)

            """convolution (inner product)"""
            affine = patches.transpose(2, 3).matmul(
                weight.reshape(self.groups, -1, in_channels * kernel_height * kernel_width).transpose(2, 1).unsqueeze(
                    1)).transpose(3, 2)
            """fold the output to obtain resulting feature maps"""
            # out = F.fold(affine, (27, 27), (1, 1))
            patches = affine.permute(1, 0, 2, 3).reshape(cur_batch_size, out_channels, height // self.stride[0], width // self.stride[1])

            """activation function"""
            if self.activation == 'relu':
                patches = F.relu(patches)
            elif self.activation == 'sigmoid':
                patches = torch.sigmoid(patches)
            elif self.activation == 'tanh':
                patches = torch.tanh(patches)
            elif self.activation == 'none':
                patches = patches
            else:
                raise ValueError('invalid activation')
            """pooling"""
            if self.pooling:
                if self.pooling_mode == 'average':
                    patches = F.avg_pool2d(patches, kernel_size=2, stride=2)
                else:
                    raise ValueError('Invalid pooling mode.')

            out = patches if idx == 0 else torch.cat([out, patches], dim=0)
            if self.verbose:
                bar.next()
        if self.verbose:
            bar.finish()
        return out

    def forward(self, input: Tensor):
        """
        input shape:(batch size, channel, height, width)
        """
        if self.training:
            """training the first convolutional layer"""
            num_patches = 40000
            patches = self.extract_random_patches_parallel(num_patches=num_patches, rf_size=self.kernel_size,
                                                           input=input)
            patches = self.patches_preprocessing_parallel(patches, training=True, parallel=True)
            centroids = self.run_kmeans_parallel(patches)

            self.weight.data = centroids

        else:
            """test stage"""
            pass
        out = self._conv_forward_parallel(input, self.weight)

        return out

    def extra_repr(self) -> str:
        return '{}, {}, kernel_size={}, stride={}, padding={},groups={}, bias={}, pooling={}, pooling_mode={}, activation={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.groups,
            self.bias is not None, self.pooling, self.pooling_mode, self.activation)
    
