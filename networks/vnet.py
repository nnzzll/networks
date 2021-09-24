import torch
import torch.nn as nn

from .base import *


class VNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int = 3,
        base_filters: int = 16,
        up: str = 'deconv',
        down: str = 'conv',
        act: str = 'prelu',
        norm: str = 'batch',
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Sequential(
            CONV[dim](in_channels, base_filters, 5, 1, 2, bias=bias),
            ACT[act](),
            NORM[dim][norm](base_filters),
        )
        self.down1 = Down(base_filters, base_filters*2, 2, 3)
        self.down2 = Down(base_filters*2, base_filters*4, 3, 3)
        self.down3 = Down(base_filters*4, base_filters*8, 3, 3)
        self.down4 = Down(base_filters*8, base_filters*16, 3, 3)
        self.up4 = Up(base_filters*16, base_filters*16, 3, 3)
        self.up3 = Up(base_filters*16, base_filters*8, 3, 3)
        self.up2 = Up(base_filters*8, base_filters*4, 2, 3)
        self.up1 = Up(base_filters*4, base_filters*2, 1, 3)
        self.out_conv = nn.Sequential(
            CONV[dim](base_filters*2, num_classes, 1, 1, bias=bias),
            ACT[act](),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = x + self.init_conv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottle = self.down4(skip4)
        up4 = self.up4(bottle, skip4)
        up3 = self.up3(up4, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)
        out = self.out_conv(up1)
        return out


class Down(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            dim: int = 3,
            down: str = 'conv',
            act: str = 'prelu',
            norm: str = 'batch',
            bias: bool = False,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            CONV[dim](in_channels, out_channels, 2, 2, bias=bias),
            NORM[dim][norm](out_channels),
            ACT[act](),
        )
        conv_list = []
        for i in range(num_layer):
            conv_list.append(
                CONV[dim](out_channels, out_channels, 5, 1, 2, bias=bias))
            conv_list.append(NORM[dim][norm](out_channels))
            conv_list.append(ACT[act]())
        self.encoder = nn.Sequential(*conv_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x + self.encoder(x)


class Up(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            dim: int = 3,
            up: str = 'deconv',
            act: str = 'prelu',
            norm: str = 'batch',
            bias: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            DECONV[dim](in_channels, out_channels//2, 2, 2, bias=bias),
            NORM[dim][norm](out_channels//2),
            ACT[act](),
        )
        conv_list = []
        for i in range(num_layer-1):
            conv_list.append(CONV[dim](out_channels, out_channels, 5, 1, 2, bias=bias))
            conv_list.append(NORM[dim][norm](out_channels))
            conv_list.append(ACT[act]())
        self.decoder = nn.Sequential(*conv_list)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.up(x), skip], dim=1)
        return x + self.decoder(x)
