import torch
import torch.nn as nn

from .base import Block


class UNetPP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            down: str = 'pool',
            up: str = 'interpolate',
            bias: bool = False,
            n_filters: list = [32, 64, 128, 256, 512],
            deep_supervision: bool = False
        ) -> None:
        super(UNetPP, self).__init__()
        self._check_param(up, down, n_filters)
        self.deep_supervision = deep_supervision

        self.conv0_0 = Block(in_channels, n_filters[0], 2, bias=bias)
        self.conv0_1 = Block(n_filters[0]+n_filters[1], n_filters[0], 2, bias=bias)
        self.conv0_2 = Block(n_filters[0]*2+n_filters[1], n_filters[0], 2, bias=bias)
        self.conv0_3 = Block(n_filters[0]*3+n_filters[1], n_filters[0], 2, bias=bias)
        self.conv0_4 = Block(n_filters[0]*4+n_filters[1], n_filters[0], 2, bias=bias)

        self.conv1_0 = Block(n_filters[0], n_filters[1], 2, bias=bias)
        self.conv1_1 = Block(n_filters[1]+n_filters[2], n_filters[1], 2, bias=bias)
        self.conv1_2 = Block(n_filters[1]*2+n_filters[2], n_filters[1], 2, bias=bias)
        self.conv1_3 = Block(n_filters[1]*3+n_filters[2], n_filters[1], 2, bias=bias)

        self.conv2_0 = Block(n_filters[1], n_filters[2], 2, bias=bias)
        self.conv2_1 = Block(n_filters[2]+n_filters[3], n_filters[2], 2, bias=bias)
        self.conv2_2 = Block(n_filters[2]*2+n_filters[3], n_filters[2], 2, bias=bias)

        self.conv3_0 = Block(n_filters[2], n_filters[3], 2, bias=bias)
        self.conv3_1 = Block(n_filters[3]+n_filters[4], n_filters[3], 2, bias=bias)

        self.conv4_0 = Block(n_filters[3], n_filters[4], 2, bias=bias)

        if down =='pool':
            self.pool1 = nn.MaxPool2d(2, 2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.pool4 = nn.MaxPool2d(2, 2)
        else:
            self.pool1 = nn.Conv2d(n_filters[0],n_filters[0], 3, 2, 1, bias=bias)
            self.pool2 = nn.Conv2d(n_filters[1],n_filters[1], 3, 2, 1, bias=bias)
            self.pool3 = nn.Conv2d(n_filters[2],n_filters[2], 3, 2, 1, bias=bias)
            self.pool4 = nn.Conv2d(n_filters[3],n_filters[3], 3, 2, 1, bias=bias)


        if up == 'deconv':
            self.up4 = nn.ConvTranspose2d(n_filters[4], n_filters[4], 2, 2, bias=bias)
            self.up3 = nn.ConvTranspose2d(n_filters[3], n_filters[3], 2, 2, bias=bias)
            self.up2 = nn.ConvTranspose2d(n_filters[2], n_filters[2], 2, 2, bias=bias)
            self.up1 = nn.ConvTranspose2d(n_filters[1], n_filters[1], 2, 2, bias=bias)
        else:
            self.up4 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.up3 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.up1 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        if deep_supervision:
            self.out1 = nn.Conv2d(n_filters[0], num_classes, 1, bias=bias)
            self.out2 = nn.Conv2d(n_filters[0], num_classes, 1, bias=bias)
            self.out3 = nn.Conv2d(n_filters[0], num_classes, 1, bias=bias)
            self.out4 = nn.Conv2d(n_filters[0], num_classes, 1, bias=bias)
        else:
            self.out = nn.Conv2d(n_filters[0], num_classes, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool1(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool2(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool3(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool4(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3)], 1))

        if self.deep_supervision:
            out1 = self.out1(x0_4)
            out2 = self.out2(x0_3)
            out3 = self.out3(x0_2)
            out4 = self.out4(x0_1)
            return [out1, out2, out3, out4]
        else:
            return self.out(x0_4)

    def _check_param(self, up: str, down: str, n_filters: list) -> None:
        if up not in ["deconv", "interpolate"]:
            raise ValueError(f"Unsupported Upsample Method:{up}")
        if down not in ["pool", "conv"]:
            raise ValueError(f"Unsupported Downsample Method:{down}")
        assert len(n_filters) == 5, "length of n_filters should be 5!"
