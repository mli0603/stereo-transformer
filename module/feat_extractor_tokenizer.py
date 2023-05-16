#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
from torch import nn, Tensor
from torchvision.models.densenet import _DenseBlock

from utilities.misc import center_crop


class TransitionUp(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2, kernel_size=None):
        super().__init__()

        #import ipdb;ipdb.set_trace()
        self.kernel_size = 11 if kernel_size is None else kernel_size
        self.padding = (kernel_size -3)//2
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=True, groups=in_channels)

    def forward(self, x: Tensor, skip: Tensor):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class DoubleConv(nn.Module):
    """
    Two conv2d-bn-relu modules
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Tokenizer(nn.Module):
    """
    Expanding path of feature descriptor using DenseBlocks
    """

    def __init__(self, block_config: int, hidden_dim: int, backbone_feat_channel: list, growth_rate: int, kernel_size=None):
        super(Tokenizer, self).__init__()

        backbone_feat_channel.reverse()  # reverse so we have high-level first (lowest-spatial res)

        self.block_config = block_config
        self.growth_rate = growth_rate
        self.hidden_dim = hidden_dim

        # 1/16
        self.bottle_neck = _DenseBlock(block_config, backbone_feat_channel[0], 4, drop_rate=0.0,
                                       growth_rate=growth_rate)
        up = []
        dense_block = []
        prev_block_channels = growth_rate * block_config

        # 1/8
        up.append(TransitionUp(prev_block_channels, prev_block_channels, kernel_size=kernel_size))
        cur_channels_count = prev_block_channels + backbone_feat_channel[1]
        dense_block.append(_DenseBlock(block_config, cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate))
        prev_block_channels = growth_rate * block_config

        # 1/4
        up.append(TransitionUp(prev_block_channels, prev_block_channels, kernel_size=kernel_size))
        cur_channels_count = prev_block_channels + backbone_feat_channel[2]
        dense_block.append(_DenseBlock(block_config, cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate))

        self.up = nn.ModuleList(up)
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, features: list):
        """
        :param features:
            list containing feature descriptors at different spatial resolution
                0: [2N, C0, H//4, W//4]
                1: [2N, C1, H//8, W//8]
                2: [2N, C2, H//16, W//16]
        :return: feature descriptor at full resolution [2N,C,H//4,W//4]
        """

        features.reverse()
        output = self.bottle_neck(features[0])
        output = output[:, -(self.block_config * self.growth_rate):]  # take only the new features

        for i in range(len(features) - 1):
            hs = self.up[i](output, features[i + 1])  # scale up and concat
            output = self.dense_block[i](hs)  # denseblock
            if i == len(features) - 2:
                output = output[:, -self.hidden_dim:]
            else:
                output = output[:, -(self.block_config * self.growth_rate):]  # take only the new features

        return output


def build_tokenizer(args, layer_channel):
    block_config = 4
    growth_rate = 16
    return Tokenizer(block_config, args.channel_dim, layer_channel, growth_rate, kernel_size=args.kernel_size)
