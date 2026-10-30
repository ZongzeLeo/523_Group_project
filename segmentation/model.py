"""A compact four-level 2D U-Net; one grayscale input and one LV logit."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels, normalization="group"):
        if normalization != "group":
            raise ValueError("This model requires group normalization")
        norm = lambda: nn.GroupNorm(math.gcd(8, output_channels), output_channels)
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            norm(), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            norm(), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base_channels=24, normalization="group"):
        super().__init__()
        widths = [base_channels * 2**i for i in range(5)]
        self.encoder = nn.ModuleList([DoubleConv(1 if i == 0 else widths[i-1], width, normalization)
                                      for i, width in enumerate(widths)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i], widths[i-1], 2, stride=2) for i in range(4, 0, -1)])
        self.decoder = nn.ModuleList([DoubleConv(widths[i-1]*2, widths[i-1], normalization) for i in range(4, 0, -1)])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 16:
            raise ValueError("Expected N x 1 x H x W grayscale images with H,W >= 16")
        skips = []
        value = image
        for index, block in enumerate(self.encoder):
            value = block(value)
            if index < 4:
                skips.append(value)
                value = F.max_pool2d(value, 2)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            value = up(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat([skip, value], dim=1))
        return self.output(value)
