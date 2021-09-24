import torch
import torch.nn as nn

from .base import Block


class UNet3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            down: str = 'pool',
            up: str = 'deconv',
            act: str = 'relu',
            norm: str = 'batch',
            bias: bool = False,
            n_filters: list = [32, 64, 128, 256, 512],
            deep_supervision: bool = False
        ) -> None:
        super(UNet3D, self).__init__()
        self._check_param(up, down, n_filters, act, norm)
        self.deep_supervision = deep_supervision

        self.conv0_0 = Block(in_channels, n_filters[0], 3, bias=bias, act=act, norm=norm)
        self.conv1_0 = Block(n_filters[0], n_filters[1], 3, bias=bias, act=act, norm=norm)
        self.conv2_0 = Block(n_filters[1], n_filters[2], 3, bias=bias, act=act, norm=norm)
        self.conv3_0 = Block(n_filters[2], n_filters[3], 3, bias=bias, act=act, norm=norm)
        self.conv4_0 = Block(n_filters[3], n_filters[4], 3, bias=bias, act=act, norm=norm)

        self.conv3_1 = Block(n_filters[4], n_filters[3], 3, bias=bias, act=act, norm=norm)
        self.conv2_2 = Block(n_filters[3], n_filters[2], 3, bias=bias, act=act, norm=norm)
        self.conv1_3 = Block(n_filters[2], n_filters[1], 3, bias=bias, act=act, norm=norm)
        self.conv0_4 = Block(n_filters[1], n_filters[0], 3, bias=bias, act=act, norm=norm)
        if down =='pool':
            self.pool1 = nn.MaxPool3d(2, 2)
            self.pool2 = nn.MaxPool3d(2, 2)
            self.pool3 = nn.MaxPool3d(2, 2)
            self.pool4 = nn.MaxPool3d(2, 2)
        else:
            self.pool1 = nn.Conv3d(n_filters[0],n_filters[0], 3, 2, 1, bias=bias)
            self.pool2 = nn.Conv3d(n_filters[1],n_filters[1], 3, 2, 1, bias=bias)
            self.pool3 = nn.Conv3d(n_filters[2],n_filters[2], 3, 2, 1, bias=bias)
            self.pool4 = nn.Conv3d(n_filters[3],n_filters[3], 3, 2, 1, bias=bias)


        if up == 'deconv':
            self.up4 = nn.ConvTranspose3d(n_filters[4], n_filters[3], 2, 2, bias=bias)
            self.up3 = nn.ConvTranspose3d(n_filters[3], n_filters[2], 2, 2, bias=bias)
            self.up2 = nn.ConvTranspose3d(n_filters[2], n_filters[1], 2, 2, bias=bias)
            self.up1 = nn.ConvTranspose3d(n_filters[1], n_filters[0], 2, 2, bias=bias)
        else:
            self.up4 = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
            self.up3 = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
            self.up1 = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)

        if deep_supervision:
            self.out1 = nn.Conv3d(n_filters[0], num_classes, 1, bias=bias)
            self.out2 = nn.Conv3d(n_filters[1], num_classes, 1, bias=bias)
            self.out3 = nn.Conv3d(n_filters[2], num_classes, 1, bias=bias)
            self.out4 = nn.Conv3d(n_filters[3], num_classes, 1, bias=bias)
        else:
            self.out = nn.Conv3d(n_filters[0], num_classes, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool1(x0_0))
        x2_0 = self.conv2_0(self.pool2(x1_0))
        x3_0 = self.conv3_0(self.pool3(x2_0))
        x4_0 = self.conv4_0(self.pool4(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up3(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up1(x1_3)], dim=1))
        if self.deep_supervision:
            out1 = self.out1(x0_4)
            out2 = self.out2(x1_3)
            out3 = self.out3(x2_2)
            out4 = self.out4(x3_1)
            return [out1, out2, out3, out4]
        else:
            return self.out(x0_4)

    def _check_param(self, up: str, down: str, n_filters: list, act: str, norm: str) -> None:
        if up not in ["deconv", "interpolate"]:
            raise ValueError(f"Unsupported Upsample Method:{up}")
        if down not in ["pool", "conv"]:
            raise ValueError(f"Unsupported Downsample Method:{down}")
        if act not in ["relu", "leaky"]:
            raise ValueError(f"Activation Type:{act} is unsupported!")
        if norm not in ["batch", "instance"]:
            raise ValueError(f"Normalization Type:{norm} is unsupported!")
        assert len(n_filters) == 5, "length of n_filters should be 5!"
