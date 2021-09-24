import torch
import torch.nn as nn
from typing import Dict, Type, Union


CONV: Dict[int, Union[Type[nn.Conv2d], Type[nn.Conv3d]]] = {2: nn.Conv2d, 3: nn.Conv3d}
DECONV: Dict[int, Union[Type[nn.ConvTranspose2d], Type[nn.ConvTranspose3d]]] = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
DROPOUT: Dict[int, Union[Type[nn.Dropout2d], Type[nn.Dropout3d]]] = {2: nn.Dropout2d, 3: nn.Dropout3d}
ACT: Dict[str, Union[Type[nn.ReLU],Type[nn.LeakyReLU],Type[nn.PReLU]]] = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'prelu': nn.PReLU}
NORM: Dict[int, Dict[str,Union[Type[nn.modules.batchnorm._BatchNorm],Type[nn.modules.instancenorm._InstanceNorm],Type[nn.GroupNorm]]]] = {
    2: {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
        'group': nn.GroupNorm,
    },
    3: {
        'batch': nn.BatchNorm3d,
        'instance': nn.InstanceNorm3d,
        'group': nn.GroupNorm,
    }
}


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        act: str = "relu",
        norm: str = "batch",
        num_group: int = 8,
        pre_norm: bool = False,
        res: bool = False,
    ) -> None:
        '''Base Block for UNet-Like network

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dim: 2D or 3D convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            bias: If ``True``, adds a learnable bias to the output.
            act: Activation function of Block.``relu``,``prelu`` or ``leaky``.
            norm: Normalization type.``batch``,``group`` or ``instance``.
            num_group: number of groups to separate the channels into
            pre_norm: If true, normalization->activation->convolution.
            res:If true, set residual-connection to block.
        '''
        super().__init__()
        self._check_param(dim, act, norm)
        self.res = res

        if pre_norm:
            self.conv_x2 = nn.Sequential(
                NORM[dim][norm](num_group, in_channels) if norm == 'group' else NORM[dim][norm](in_channels),
                ACT[act](inplace=True),
                CONV[dim](in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                NORM[dim][norm](num_group, out_channels) if norm == 'group' else NORM[dim][norm](out_channels),
                ACT[act](inplace=True),
                CONV[dim](out_channels, out_channels, kernel_size, stride, padding, bias=bias),
            )
        else:
            self.conv_x2 = nn.Sequential(
                CONV[dim](in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                NORM[dim][norm](num_group, out_channels) if norm == 'group' else NORM[dim][norm](out_channels),
                ACT[act](inplace=True),
                CONV[dim](out_channels, out_channels, kernel_size, stride, padding, bias=bias),
                NORM[dim][norm](num_group, out_channels) if norm == 'group' else NORM[dim][norm](out_channels),
                ACT[act](inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            return x + self.conv_x2(x)
        return self.conv_x2(x)

    def _check_param(self, dim: int, act: str, norm: str) -> None:
        if dim != 2 and dim != 3:
            raise ValueError(f"Convolution Dim:{dim} is unsupported!")
        if act not in ["relu", "leaky","prelu"]:
            raise ValueError(f"Activation Type:{act} is unsupported!")
        if norm not in ["batch", "instance", "group"]:
            raise ValueError(f"Normalization Type:{norm} is unsupported!")
