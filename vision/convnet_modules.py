import torch
from torch import nn


class DepthwiseConvBlock(nn.Module):
    def __init__(self):
        self.depthconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, groups=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.depthconv(x)

        return x
