import torch
import torch.nn as nn

from .base import *
from torch.utils.checkpoint import checkpoint


class nnUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim: int,
        bias: bool = False,
        dropout: float = 0,
        down: str = 'pool',
        up: str = 'deconv',
        act: str = 'leaky',
        norm: str = 'instance',
        deep_supervision: bool = False,
        checkpoint:bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.checkpoint = checkpoint
        n_filters = [32, 64, 128, 256, 320, 320]
        self.in_conv = Block(in_channels,n_filters[0],dim,3,1,1,bias,dropout,act,norm)

        self.down1 = Block(n_filters[0],n_filters[1],dim,3,1,1,bias,dropout,act,norm,down=True)
        self.down2 = Block(n_filters[1],n_filters[2],dim,3,1,1,bias,dropout,act,norm,down=True)
        self.down3 = Block(n_filters[2],n_filters[3],dim,3,1,1,bias,dropout,act,norm,down=True)
        self.down4 = Block(n_filters[3],n_filters[4],dim,3,1,1,bias,dropout,act,norm,down=True)
        self.down5 = Block(n_filters[4],n_filters[5],dim,3,1,1,bias,dropout,act,norm,down=True)

        self.decoder5 = Block(n_filters[4]*2,n_filters[4],dim,3,1,1,bias,dropout,act,norm)
        self.decoder4 = Block(n_filters[3]*2,n_filters[3],dim,3,1,1,bias,dropout,act,norm)
        self.decoder3 = Block(n_filters[2]*2,n_filters[2],dim,3,1,1,bias,dropout,act,norm)
        self.decoder2 = Block(n_filters[1]*2,n_filters[1],dim,3,1,1,bias,dropout,act,norm)
        self.decoder1 = Block(n_filters[0]*2,n_filters[0],dim,3,1,1,bias,dropout,act,norm)

        self.up5 = DECONV[dim](n_filters[5],n_filters[4],2,2,bias=bias)
        self.up4 = DECONV[dim](n_filters[4],n_filters[3],2,2,bias=bias)
        self.up3 = DECONV[dim](n_filters[3],n_filters[2],2,2,bias=bias)
        self.up2 = DECONV[dim](n_filters[2],n_filters[1],2,2,bias=bias)
        self.up1 = DECONV[dim](n_filters[1],n_filters[0],2,2,bias=bias)

        self.out_conv = CONV[dim](n_filters[0],num_classes,1,1,bias=bias)
        if deep_supervision:
            self.ds = nn.ModuleList([CONV[dim](n_filters[i],num_classes,1,1,bias=bias) for i in range(3)])


    def forward(self, x: torch.Tensor)->Union[torch.Tensor,Sequence[torch.Tensor]]:
        e1 = self.in_conv(x) if not self.checkpoint else checkpoint(self.in_conv,x)
        e2 = self.down1(e1) if not self.checkpoint else checkpoint(self.down1,e1)
        e3 = self.down2(e2) if not self.checkpoint else checkpoint(self.down2,e2)
        e4 = self.down3(e3) if not self.checkpoint else checkpoint(self.down3,e3)
        e5 = self.down4(e4) if not self.checkpoint else checkpoint(self.down4,e4)
        bottleneck = self.down5(e5) if not self.checkpoint else checkpoint(self.down5,e5)
        
        d5 = self.decoder5(torch.cat([self.up5(bottleneck),e5],dim=1)) if not self.checkpoint else checkpoint(self.decoder5,torch.cat([self.up5(bottleneck),e5],dim=1))
        d4 = self.decoder4(torch.cat([self.up4(d5),e4],dim=1)) if not self.checkpoint else checkpoint(self.decoder4,torch.cat([self.up4(d5),e4],dim=1))
        d3 = self.decoder3(torch.cat([self.up3(d4),e3],dim=1)) if not self.checkpoint else checkpoint(self.decoder3,torch.cat([self.up3(d4),e3],dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d3),e2],dim=1)) if not self.checkpoint else checkpoint(self.decoder2,torch.cat([self.up2(d3),e2],dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2),e1],dim=1)) if not self.checkpoint else checkpoint(self.decoder1,torch.cat([self.up1(d2),e1],dim=1))
        
        if self.training and self.deep_supervision:
            out = [self.ds[0](d1),self.ds[1](d2),self.ds[2](d3)]
        else:
            out = self.out_conv(d1)
        return out