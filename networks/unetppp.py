import torch
import torch.nn

from .base import *


class UNetPPP(nn.Module):
    '''
    UNet 3+: A full-scale connected unet for medical image segmentation
    <https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf>
    '''
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            bias: bool = False,
            n_filters: list = [32, 64, 128, 256, 512],
            deep_supervision: bool = False
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        C = n_filters[0]*5

        self.encoder1 = Block(in_channels, n_filters[0], 2, bias=bias)
        self.encoder2 = Block(n_filters[0], n_filters[1], 2, bias=bias)
        self.encoder3 = Block(n_filters[1], n_filters[2], 2, bias=bias)
        self.encoder4 = Block(n_filters[2], n_filters[3], 2, bias=bias)
        self.encoder5 = Block(n_filters[3], n_filters[4], 2, bias=bias)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.decoder1 = Block(C, C, 2, bias=bias)
        self.decoder2 = Block(C, C, 2, bias=bias)
        self.decoder3 = Block(C, C, 2, bias=bias)
        self.decoder4 = Block(C, C, 2, bias=bias)
        self.decoder5 = Block(C, C, 2, bias=bias)

        # Full-Scale Skip Connection
        self.e1_d1 = SkipConv(n_filters[0], n_filters[0], bias=bias)
        self.e1_d2 = SkipConv(n_filters[0], n_filters[0], True, False, 2, bias)
        self.e1_d3 = SkipConv(n_filters[0], n_filters[0], True, False, 4, bias)
        self.e1_d4 = SkipConv(n_filters[0], n_filters[0], True, False, 8, bias)
        self.e2_d2 = SkipConv(n_filters[1], n_filters[0], bias=bias)
        self.e2_d3 = SkipConv(n_filters[1], n_filters[0], True, False, 2, bias)
        self.e2_d4 = SkipConv(n_filters[1], n_filters[0], True, False, 4, bias)
        self.e3_d3 = SkipConv(n_filters[2], n_filters[0], bias=bias)
        self.e3_d4 = SkipConv(n_filters[2], n_filters[0], True, False, 2, bias)
        self.e4_d4 = SkipConv(n_filters[3], n_filters[0], bias=bias)

        self.e5_d1 = SkipConv(n_filters[4], n_filters[0], False, True, 16, bias)
        self.e5_d2 = SkipConv(n_filters[4], n_filters[0], False, True, 8, bias)
        self.e5_d3 = SkipConv(n_filters[4], n_filters[0], False, True, 4, bias)
        self.e5_d4 = SkipConv(n_filters[4], n_filters[0], False, True, 2, bias)

        self.d4_d1 = SkipConv(C, n_filters[0], False, True, 8, bias)
        self.d4_d2 = SkipConv(C, n_filters[0], False, True, 4, bias)
        self.d4_d3 = SkipConv(C, n_filters[0], False, True, 2, bias)

        self.d3_d1 = SkipConv(C, n_filters[0], False, True, 4, bias)
        self.d3_d2 = SkipConv(C, n_filters[0], False, True, 2, bias)

        self.d2_d1 = SkipConv(C, n_filters[0], False, True, 2, bias)

        # Output Conv Layer in the official code uses 3 as kernel_size
        if deep_supervision:
            self.out1 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
            self.out2 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
            self.out3 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
            self.out4 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
        else:
            self.out = nn.Conv2d(C, num_classes, 3, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        e5 = self.encoder5(self.pool(e4))

        # Fuse Decoder 4
        t1 = self.e1_d4(e1)
        t2 = self.e2_d4(e2)
        t3 = self.e3_d4(e3)
        t4 = self.e4_d4(e4)
        t5 = self.e5_d4(e5)
        fusion = torch.cat([t1, t2, t3, t4, t5], dim=1)
        d4 = self.decoder4(fusion)

        # Fuse Decoder 3
        t1 = self.e1_d3(e1)
        t2 = self.e2_d3(e2)
        t3 = self.e3_d3(e3)
        t4 = self.d4_d3(d4)
        t5 = self.e5_d3(e5)
        fusion = torch.cat([t1, t2, t3, t4, t5], dim=1)
        d3 = self.decoder3(fusion)

        # Fuse Decoder 2
        t1 = self.e1_d2(e1)
        t2 = self.e2_d2(e2)
        t3 = self.d3_d2(d3)
        t4 = self.d4_d2(d4)
        t5 = self.e5_d2(e5)
        fusion = torch.cat([t1, t2, t3, t4, t5], dim=1)
        d2 = self.decoder4(fusion)

        # Fuse Decoder 1
        t1 = self.e1_d1(e1)
        t2 = self.d2_d1(d2)
        t3 = self.d3_d1(d3)
        t4 = self.d4_d1(d4)
        t5 = self.e5_d1(e5)
        fusion = torch.cat([t1, t2, t3, t4, t5], dim=1)
        d1 = self.decoder1(fusion)

        if self.deep_supervision:
            out1 = self.out1(d1)
            out2 = self.out2(d2)
            out3 = self.out3(d3)
            out4 = self.out4(d4)
            return [out1, out2, out3, out4]
        else:
            return self.out(d1)


class SkipConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = False,
        up: bool = False,
        scale_factor: int = 2,
        bias: bool = False,
    ) -> None:
        '''Full-scale Skip Connections in the paper'''
        super().__init__()
        assert (pool and up) == False, "Skip connection should be only downsampling or upsampling"
        self.pool = pool
        self.up = up
        if pool:
            self.downsample = nn.MaxPool2d(scale_factor, scale_factor, ceil_mode=True)
        if up:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool:
            x = self.downsample(x)
        elif self.up:
            x = self.upsample(x)
        x = self.relu(self.bn(self.conv(x)))
        return x
