import torch
import torch.nn as nn

from .base import Block, CONV, DROPOUT, DECONV


class StageOne(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            dim: int = 3,
            base_filters: int = 16,
            dropout: float = 0.2,
            bias: bool = False,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Sequential(
            CONV[dim](in_channels, base_filters, 3, 1, 1, bias=bias),
            DROPOUT[dim](dropout, True),
        )
        self.encoder1 = Block(base_filters, base_filters, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True)
        self.encoder2 = nn.Sequential(
            Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
        )
        self.encoder3 = nn.Sequential(
            Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
        )
        self.bottleneck = nn.Sequential(
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
        )
        self.decoder3 = Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout,'relu', 'group', 8, True, True)
        self.decoder2 = Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout,'relu', 'group', 8, True, True)
        self.decoder1 = Block(base_filters, base_filters, dim, 3, 1, 1,  bias, dropout, 'relu', 'group', 8, True, True)

        self.out_conv = CONV[dim](base_filters, num_classes, 1, 1, bias=bias)

        self.down1 = CONV[dim](base_filters, base_filters*2, 3, 2, 1, bias=bias)
        self.down2 = CONV[dim](base_filters*2, base_filters*4, 3, 2, 1, bias=bias)
        self.down3 = CONV[dim](base_filters*4, base_filters*8, 3, 2, 1, bias=bias)

        self.up3 = nn.Sequential(
            CONV[dim](base_filters*8, base_filters*4, 1, 1, bias=bias),
            DECONV[dim](base_filters*4, base_filters*4, 2, 2, bias=bias)
        )
        self.up2 = nn.Sequential(
            CONV[dim](base_filters*4, base_filters*2, 1, 1, bias=bias),
            DECONV[dim](base_filters*2, base_filters*2, 2, 2, bias=bias)
        )
        self.up1 = nn.Sequential(
            CONV[dim](base_filters*2, base_filters, 1, 1, bias=bias),
            DECONV[dim](base_filters, base_filters, 2, 2, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.down1(skip1))
        skip3 = self.encoder3(self.down2(skip2))
        bottle = self.bottleneck(self.down3(skip3))
        out3 = self.decoder3(skip3+self.up3(bottle))
        out2 = self.decoder2(skip2+self.up2(out3))
        out1 = self.decoder1(skip1+self.up1(out2))
        out = self.out_conv(out1)
        return torch.sigmoid(out)


class StageTwo(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int = 3,
        base_filters: int = 32,
        dropout: float = 0.2,
        bias: bool = False,
        training_flag: bool = True,
    ) -> None:
        super().__init__()
        self.training_flag = training_flag
        self.init_conv = nn.Sequential(
            CONV[dim](in_channels, base_filters, 3, 1, 1, bias=bias),
            DROPOUT[dim](dropout, True),
        )
        self.encoder1 = Block(base_filters, base_filters, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True)
        self.encoder2 = nn.Sequential(
            Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
        )
        self.encoder3 = nn.Sequential(
            Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
        )
        self.bottleneck = nn.Sequential(
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
            Block(base_filters*8, base_filters*8, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True),
        )
        self.decoder3 = Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True)
        self.decoder2 = Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True)
        self.decoder1 = Block(base_filters, base_filters, dim, 3, 1, 1,  bias, dropout, 'relu', 'group', 8, True, True)

        self.out_conv = CONV[dim](base_filters, num_classes, 1, 1, bias=bias)

        self.down1 = CONV[dim](base_filters, base_filters*2, 3, 2, 1, bias=bias)
        self.down2 = CONV[dim](base_filters*2, base_filters*4, 3, 2, 1, bias=bias)
        self.down3 = CONV[dim](base_filters*4, base_filters*8, 3, 2, 1, bias=bias)

        self.up3 = nn.Sequential(
            CONV[dim](base_filters*8, base_filters*4, 1, 1, bias=bias),
            DECONV[dim](base_filters*4, base_filters*4, 2, 2, bias=bias)
        )
        self.up2 = nn.Sequential(
            CONV[dim](base_filters*4, base_filters*2, 1, 1, bias=bias),
            DECONV[dim](base_filters*2, base_filters*2, 2, 2, bias=bias)
        )
        self.up1 = nn.Sequential(
            CONV[dim](base_filters*2, base_filters, 1, 1, bias=bias),
            DECONV[dim](base_filters, base_filters, 2, 2, bias=bias)
        )

        if training_flag:
            self.decoder3_1 = Block(base_filters*4, base_filters*4, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True)
            self.decoder2_1 = Block(base_filters*2, base_filters*2, dim, 3, 1, 1, bias, dropout, 'relu', 'group', 8, True, True)
            self.decoder1_1 = Block(base_filters, base_filters, dim, 3, 1, 1,  bias, dropout, 'relu', 'group', 8, True, True)
            self.up3_1 = nn.Sequential(
                CONV[dim](base_filters*8, base_filters*4, 1, 1, bias=bias),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) if dim == 3 else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.up2_1 = nn.Sequential(
                CONV[dim](base_filters*4, base_filters*2, 1, 1, bias=bias),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) if dim == 3 else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.up1_1 = nn.Sequential(
                CONV[dim](base_filters*2, base_filters, 1, 1, bias=bias),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) if dim == 3 else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.out_conv_1 = CONV[dim](base_filters, num_classes, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.down1(skip1))
        skip3 = self.encoder3(self.down2(skip2))
        bottle = self.bottleneck(self.down3(skip3))
        out3 = self.decoder3(skip3+self.up3(bottle))
        out2 = self.decoder2(skip2+self.up2(out3))
        out1 = self.decoder1(skip1+self.up1(out2))
        out = self.out_conv(out1)
        if self.training_flag:
            out3_1 = self.decoder3_1(skip3+self.up3_1(bottle))
            out2_1 = self.decoder2_1(skip2+self.up2_1(out3_1))
            out1_1 = self.decoder1_1(skip1+self.up1_1(out2_1))
            out_1 = self.out_conv_1(out1_1)
            return [torch.sigmoid(out), torch.sigmoid(out_1)]
        return torch.sigmoid(out)


class CascadeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int = 3,
        dropout: float = 0.2,
        bias: bool = False,
        training_flag: bool = True,
    ) -> None:
        '''
        Two-Stage Cascaded U-Net: 1st Place Solution to BraTS Challenge 2019 Segmentation Task
        <https://doi.org/10.1007/978-3-030-46640-4_22>

        Args:
            in_channels: Number of channels in the input image.
            num_classes: Number of classes to segment.
            dim: Dimension. 2D or 3D segmentation.
            dropout: Dropout rate.
            training_flag: If true, there are 2 decoder path in the stage2 Unet.
        '''
        super().__init__()
        self.unet1 = StageOne(in_channels, num_classes, dim, 16, dropout, bias)
        self.unet2 = StageTwo(in_channels+num_classes, num_classes, dim, 32, dropout, bias, training_flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.unet1(x)
        out2 = self.unet2(torch.cat([out1, x], dim=1))
        if self.training:
            return out1, out2 # out1:Stage1 output,out2:deconv result and trilinear result
        return out2 # Only Outputs deconv result
