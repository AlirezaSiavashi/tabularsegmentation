from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_base_grid(D: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """
    grid_sample expects grid[..., 3] in (x,y,z) normalized to [-1,1]
    returns (1, D, H, W, 3)
    """
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)[None, ...]  # (1,D,H,W,3)
    return grid

def _axis_step_norm(axis: int, step: int, D: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """
    axis: 0->z,1->y,2->x in volume index terms, but grid is (x,y,z)
    We'll return displacement in normalized (x,y,z).
    """
    # normalized step size depends on resolution in that axis:
    # x step is 2/(W-1), y is 2/(H-1), z is 2/(D-1)
    dx = 2.0 / max(1, W - 1)
    dy = 2.0 / max(1, H - 1)
    dz = 2.0 / max(1, D - 1)

    disp = torch.zeros((1,1,1,1,3), device=device, dtype=dtype)
    if axis == 2:      # x
        disp[..., 0] = step * dx
    elif axis == 1:    # y
        disp[..., 1] = step * dy
    elif axis == 0:    # z
        disp[..., 2] = step * dz
    else:
        raise ValueError(axis)
    return disp

@dataclass
class SnakeAxisConfig:
    kernel_size: int = 9  # must be odd
    drop_p: float = 0.5   # random dropping for multi-view fusion

class SnakeAxisConv3D(nn.Module):
    """
    1D snake conv along one axis with *cumulative* orthogonal offsets.

    For axis='x':
      - kernel positions are x + step (step in [-k//2, ..., +k//2])
      - learn incremental offsets in y and z for each step away from center
      - cumulative sum from center outward (snake-like constraint)
    Similar for y and z.
    """
    def __init__(self, in_ch: int, out_ch: int, axis: int, cfg: SnakeAxisConfig):
        super().__init__()
        assert cfg.kernel_size % 2 == 1, "kernel_size must be odd"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.axis = axis
        self.k = cfg.kernel_size
        self.rad = cfg.kernel_size // 2
        self.drop_p = float(cfg.drop_p)

        # Predict incremental offsets for steps 1..rad on both sides => 2*rad steps
        # Each step predicts 2 orth offsets (the two axes perpendicular to self.axis)
        self.off_ch = 2 * (2 * self.rad)

        self.offset_pred = nn.Conv3d(in_ch, self.off_ch, kernel_size=3, padding=1)

        # After sampling k points, we concatenate => (k*in_ch) then 1x1 conv
        self.fuse = nn.Conv3d(in_ch * self.k, out_ch, kernel_size=1)

    def _split_offsets(self, inc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inc: (B, off_ch, D,H,W)
        returns:
          inc_pos: (B, rad, 2, D,H,W) increments for +1..+rad
          inc_neg: (B, rad, 2, D,H,W) increments for -1..-rad
        """
        B, C, D, H, W = inc.shape
        rad = self.rad
        inc = torch.tanh(inc)  # constrain to [-1,1] like paper’s idea :contentReference[oaicite:4]{index=4}
        inc_pos = inc[:, : 2*rad, ...].reshape(B, rad, 2, D, H, W)
        inc_neg = inc[:, 2*rad : 4*rad, ...].reshape(B, rad, 2, D, H, W)
        return inc_pos, inc_neg

    def _cumulative(self, inc_pos: torch.Tensor, inc_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cumulative sums outward:
          cum_pos[t] = sum_{i=0..t} inc_pos[i]
          cum_neg[t] = sum_{i=0..t} inc_neg[i]
        shapes: (B, rad, 2, D,H,W)
        """
        cum_pos = torch.cumsum(inc_pos, dim=1)
        cum_neg = torch.cumsum(inc_neg, dim=1)
        return cum_pos, cum_neg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,D,H,W)
        """
        B, C, D, H, W = x.shape
        inc = self.offset_pred(x)
        inc_pos, inc_neg = self._split_offsets(inc)
        cum_pos, cum_neg = self._cumulative(inc_pos, inc_neg)

        base = _make_base_grid(D, H, W, device=x.device, dtype=x.dtype)  # (1,D,H,W,3)

        # Build grids for each of k positions (center + rad pos + rad neg)
        grids = []
        steps = list(range(-self.rad, self.rad + 1))
        # prepare mapping from step -> (dy,dz) or equivalent
        for step in steps:
            disp = _axis_step_norm(self.axis, step, D, H, W, x.device, x.dtype)  # (1,1,1,1,3)
            g = base + disp

            if step == 0:
                # no orth offset at center
                grids.append(g)
                continue

            # pick the two orth axes (volume index axes: z=0,y=1,x=2; but grid channels are x,y,z)
            # we express offsets in normalized grid coordinates:
            # If axis==2 (x), orth are y and z -> grid[...,1], grid[...,2]
            # If axis==1 (y), orth are x and z -> grid[...,0], grid[...,2]
            # If axis==0 (z), orth are x and y -> grid[...,0], grid[...,1]
            if step > 0:
                t = step - 1  # 0..rad-1
                off = cum_pos[:, t, ...]  # (B,2,D,H,W)
            else:
                t = (-step) - 1
                off = cum_neg[:, t, ...]

            # convert unit offsets (≈ voxels) to normalized offsets
            # same scaling as step sizes
            dx = 2.0 / max(1, W - 1)
            dy = 2.0 / max(1, H - 1)
            dz = 2.0 / max(1, D - 1)

            # off[:,0] and off[:,1] correspond to the two orth axes in voxel units (approx).
            # We add them into grid channels accordingly.
            gB = g.expand(B, -1, -1, -1, -1).clone()  # (B,D,H,W,3)

            if self.axis == 2:  # x-axis snake => offsets in y,z
                gB[..., 1] = gB[..., 1] + off[:, 0, ...] * dy
                gB[..., 2] = gB[..., 2] + off[:, 1, ...] * dz
            elif self.axis == 1:  # y-axis snake => offsets in x,z
                gB[..., 0] = gB[..., 0] + off[:, 0, ...] * dx
                gB[..., 2] = gB[..., 2] + off[:, 1, ...] * dz
            else:  # z-axis snake => offsets in x,y
                gB[..., 0] = gB[..., 0] + off[:, 0, ...] * dx
                gB[..., 1] = gB[..., 1] + off[:, 1, ...] * dy

            grids.append(gB)

        # Stack (k, ...) -> (B*k,D,H,W,3) for efficient grid_sample
        # Some grids are (1,...) some are (B,...); normalize them:
        grids2 = []
        for g in grids:
            if g.shape[0] == 1:
                grids2.append(g.expand(B, -1, -1, -1, -1))
            else:
                grids2.append(g)
        grid = torch.stack(grids2, dim=1)  # (B,k,D,H,W,3)

        # Multi-view dropping (paper’s idea: random dropping templates) :contentReference[oaicite:5]{index=5}
        if self.training and self.drop_p < 1.0:
            keep = max(1, int(math.floor(self.k * self.drop_p)))
            # random subset per batch (cheap, stable)
            idx = torch.randperm(self.k, device=x.device)[:keep]
            grid = grid[:, idx, ...]
            k_eff = keep
        else:
            k_eff = grid.shape[1]

        x_rep = x.repeat_interleave(k_eff, dim=0)                 # (B*k_eff,C,D,H,W)
        grid_rs = grid.reshape(B * k_eff, D, H, W, 3)             # (B*k_eff,D,H,W,3)
        samp = F.grid_sample(x_rep, grid_rs, mode="bilinear", padding_mode="border", align_corners=True)
        samp = samp.reshape(B, k_eff, C, D, H, W).permute(0, 2, 1, 3, 4, 5)  # (B,C,k_eff,D,H,W)
        samp_cat = samp.reshape(B, C * k_eff, D, H, W)

        out = self.fuse(samp_cat)
        return out

class SnakeConv3D(nn.Module):
    """
    Combine 3 snake axes (x,y,z) and fuse.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9, drop_p: float = 0.6):
        super().__init__()
        cfg = SnakeAxisConfig(kernel_size=kernel_size, drop_p=drop_p)
        self.x = SnakeAxisConv3D(in_ch, out_ch, axis=2, cfg=cfg)
        self.y = SnakeAxisConv3D(in_ch, out_ch, axis=1, cfg=cfg)
        self.z = SnakeAxisConv3D(in_ch, out_ch, axis=0, cfg=cfg)
        self.fuse = nn.Conv3d(out_ch * 3, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.x(x)
        fy = self.y(x)
        fz = self.z(x)
        return self.fuse(torch.cat([fx, fy, fz], dim=1))
