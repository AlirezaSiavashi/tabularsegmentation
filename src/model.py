from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .snake_conv import SnakeConv3D

# --------- Mamba or fallback ----------
class FallbackSSMBlock(nn.Module):
    """
    If mamba_ssm isn't available, this is a cheap global mixer:
    LayerNorm -> depthwise Conv1D -> gated pointwise -> residual.
    Not real Mamba, but keeps the code runnable.
    """
    def __init__(self, d_model: int, kernel_size: int = 9):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2, groups=d_model)
        self.pw = nn.Conv1d(d_model, 2*d_model, kernel_size=1)
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C)
        y = self.ln(x)
        y = y.transpose(1,2)       # (B,C,L)
        y = self.dw(y)
        y = self.pw(y)
        a, b = y.chunk(2, dim=1)
        y = torch.sigmoid(a) * torch.tanh(b)
        y = self.out(y).transpose(1,2)  # (B,L,C)
        return x + y

def build_mamba_block(d_model: int):
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
        return Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    except Exception:
        return FallbackSSMBlock(d_model=d_model)

class ResSnakeBlock(nn.Module):
    def __init__(self, ch: int, drop_p: float = 0.6):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(ch, affine=True)
        self.conv1 = SnakeConv3D(ch, ch, kernel_size=9, drop_p=drop_p)
        self.norm2 = nn.InstanceNorm3d(ch, affine=True)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(x))
        y = self.conv1(y)
        y = self.act(self.norm2(y))
        y = self.conv2(y)
        return x + y

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_p: float):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.block1 = ResSnakeBlock(out_ch, drop_p=drop_p)
        self.block2 = ResSnakeBlock(out_ch, drop_p=drop_p)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, drop_p: float):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.block1 = ResSnakeBlock(out_ch, drop_p=drop_p)
        self.block2 = ResSnakeBlock(out_ch, drop_p=drop_p)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        if x.shape[-3:] != skip.shape[-3:]:
            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2, dz//2, dz-dz//2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

class MambaTopologyBottleneck(nn.Module):
    def __init__(self, ch: int, depth: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(ch, ch)
        self.blocks = nn.ModuleList([build_mamba_block(ch) for _ in range(depth)])
        self.out_proj = nn.Linear(ch, ch)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,D,H,W) -> tokens (B,L,C)
        B, C, D, H, W = x.shape
        t = x.permute(0,2,3,4,1).reshape(B, D*H*W, C)  # (B,L,C)
        t = self.in_proj(t)
        for blk in self.blocks:
            t = blk(t)
        t = self.ln(t)
        t = self.out_proj(t)
        y = t.reshape(B, D, H, W, C).permute(0,4,1,2,3)
        return x + y

class MambaSnakeUNet3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 2,
        base: int = 32,
        drop_p: float = 0.6,
        mamba_depth: int = 4,
    ):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, base, kernel_size=3, padding=1)

        self.down1 = Down(base, base*2, drop_p=drop_p)
        self.down2 = Down(base*2, base*4, drop_p=drop_p)
        self.down3 = Down(base*4, base*8, drop_p=drop_p)

        self.mid_conv = nn.Conv3d(base*8, base*16, kernel_size=3, padding=1)
        self.mid_block1 = ResSnakeBlock(base*16, drop_p=drop_p)
        self.mid_mamba = MambaTopologyBottleneck(base*16, depth=mamba_depth)
        self.mid_block2 = ResSnakeBlock(base*16, drop_p=drop_p)

        self.up3 = Up(base*16, base*8, base*8, drop_p=drop_p)
        self.up2 = Up(base*8, base*4, base*4, drop_p=drop_p)
        self.up1 = Up(base*4, base*2, base*2, drop_p=drop_p)

        self.head = nn.Conv3d(base*2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)

        x = self.mid_conv(x)
        x = self.mid_block1(x)
        x = self.mid_mamba(x)
        x = self.mid_block2(x)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.head(x)
