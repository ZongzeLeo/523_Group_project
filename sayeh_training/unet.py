"""Baseline 2D U-Net (Ronneberger et al., 2015) with batch norm."""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base=32, depth=5):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]  # e.g. 32,64,128,256,512
        self.down = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.down.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.up_conv.append(DoubleConv(2 * c, c))
            prev = c
        self.head = nn.Conv2d(prev, n_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, conv in zip(self.up, self.up_conv):
            x = up(x)
            x = conv(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
