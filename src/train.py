#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Mamba-Snake 3D tubular segmentation (stable, sparse-aware, deep-supervised, bidir-mamba, xyz-snake)

Main upgrades:
- Bidirectional Mamba (forward + reverse) + optional multi-axis scanning at bottleneck
- Snake refinement in x,y,z (not just z)
- Deep supervision heads (/1, /2, /4, /8) with weighted losses
- Sparse-aware patch sampling: --min-fg-voxels retries until patch has enough vessel voxels
- Warmup + cosine LR
- Scheduled CE foreground weight + scheduled clDice iterations + scheduled clDice weight
- NaN guards (skip bad batches), optional grad clip
- BF16 autocast recommended on A100

Cache format expected:
  img: [1,D,H,W] float16/float32 in [0,1]
  lab: [D,H,W] uint8 (0..num_classes-1)
"""

from __future__ import annotations

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Iterable, Set
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# -------------------------
# Repro / Perf flags
# -------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_perf_flags(tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


# -------------------------
# Patch ops
# -------------------------
def _pad_to_min_shape_torch(vol: torch.Tensor, target_dhw: Tuple[int, int, int], pad_value: float = 0.0) -> torch.Tensor:
    # vol: [D,H,W] or [C,D,H,W]
    if vol.ndim == 3:
        D, H, W = vol.shape
        has_c = False
    elif vol.ndim == 4:
        _, D, H, W = vol.shape
        has_c = True
    else:
        raise RuntimeError(f"Unexpected vol ndim={vol.ndim}")

    td, th, tw = target_dhw
    pd0 = max(0, td - D)
    ph0 = max(0, th - H)
    pw0 = max(0, tw - W)
    if pd0 == 0 and ph0 == 0 and pw0 == 0:
        return vol

    pd_before, pd_after = pd0 // 2, pd0 - (pd0 // 2)
    ph_before, ph_after = ph0 // 2, ph0 - (ph0 // 2)
    pw_before, pw_after = pw0 // 2, pw0 - (pw0 // 2)

    pad = (pw_before, pw_after, ph_before, ph_after, pd_before, pd_after)
    if has_c:
        return F.pad(vol, pad, mode="constant", value=pad_value)
    else:
        return F.pad(vol.unsqueeze(0), pad, mode="constant", value=pad_value).squeeze(0)


def _crop_around_center_torch(vol: torch.Tensor, center_dhw: Tuple[int, int, int], patch_dhw: Tuple[int, int, int]) -> torch.Tensor:
    # vol: [D,H,W] or [C,D,H,W]
    if vol.ndim == 3:
        D, H, W = vol.shape
        has_c = False
    else:
        _, D, H, W = vol.shape
        has_c = True

    pd, ph, pw = patch_dhw
    cd, ch, cw = center_dhw

    sd = int(np.clip(cd - pd // 2, 0, max(0, D - pd)))
    sh = int(np.clip(ch - ph // 2, 0, max(0, H - ph)))
    sw = int(np.clip(cw - pw // 2, 0, max(0, W - pw)))

    if has_c:
        return vol[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
    else:
        return vol[sd:sd + pd, sh:sh + ph, sw:sw + pw]


def _random_center_from_label(label: torch.Tensor, want_fg: bool, rng: np.random.Generator) -> Tuple[int, int, int]:
    if want_fg:
        idx = (label > 0).nonzero(as_tuple=False)
    else:
        idx = (label == 0).nonzero(as_tuple=False)

    if idx.numel() == 0:
        D, H, W = label.shape
        return (int(rng.integers(0, D)), int(rng.integers(0, H)), int(rng.integers(0, W)))

    pick = idx[int(rng.integers(0, idx.shape[0]))]
    return (int(pick[0].item()), int(pick[1].item()), int(pick[2].item()))


def _augment_patch(img: torch.Tensor, lab: torch.Tensor, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    # flips
    for axis in (1, 2, 3):  # img dims: C,D,H,W
        if rng.random() < 0.5:
            img = torch.flip(img, dims=(axis,))
            lab = torch.flip(lab, dims=(axis - 1,))

    # intensity jitter
    if rng.random() < 0.25:
        scale = float(rng.uniform(0.9, 1.1))
        shift = float(rng.uniform(-0.05, 0.05))
        img = torch.clamp(img * scale + shift, 0.0, 1.0)

    # gaussian noise
    if rng.random() < 0.20:
        noise = torch.from_numpy(rng.normal(0.0, 0.01, size=img.shape).astype(np.float32))
        img = torch.clamp(img + noise.to(img.device), 0.0, 1.0)

    return img, lab


# -------------------------
# Case listing (cache)
# -------------------------
def list_cases_from_cache(cache_dir: Path) -> List[Dict[str, str]]:
    cases_json = cache_dir / "cases.json"
    if not cases_json.exists():
        raise RuntimeError(f"cases.json not found in cache-dir: {cases_json}")
    js = json.loads(cases_json.read_text())
    uids = js["uids"]
    images = js["images"]
    labels = js["labels"]
    items: List[Dict[str, str]] = []
    for uid in uids:
        items.append({"uid": uid, "image_pt": images[uid], "label_pt": labels[uid]})
    if not items:
        raise RuntimeError(f"No cached items found in {cases_json}")
    return items


def split_train_val(items: List[Dict[str, str]], val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = max(1, int(len(items) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in range(len(items)) if i in val_idx]
    return train, val


class _LRUVolCache:
    def __init__(self, max_items: int = 0):
        self.max_items = int(max_items)
        self.data: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str):
        if self.max_items <= 0:
            return None
        v = self.data.get(key, None)
        if v is None:
            return None
        self.data.move_to_end(key)
        return v

    def put(self, key: str, value: Any):
        if self.max_items <= 0:
            return
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.max_items:
            self.data.popitem(last=False)


def _resolve_cached_path(uid: str, p: str, cache_dir: Path, kind: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)
    # fallback canonical
    if kind == "image":
        alt = cache_dir / "images" / f"{uid}.pt"
    else:
        alt = cache_dir / "labels" / f"{uid}.pt"
    return str(alt)


class CachedPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[Dict[str, str]],
        cache_dir: Path,
        patch_size: Tuple[int, int, int],
        training: bool,
        pos_ratio: float = 0.97,
        seed: int = 42,
        ram_cache_items: int = 0,
        min_fg_voxels: int = 256,
        max_center_tries: int = 32,
    ):
        self.items = items
        self.cache_dir = Path(cache_dir)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.training = bool(training)
        self.pos_ratio = float(pos_ratio)
        self.seed = int(seed)
        self.min_fg_voxels = int(min_fg_voxels) if training else 0
        self.max_center_tries = int(max_center_tries)

        self.cache_img = _LRUVolCache(ram_cache_items)
        self.cache_lab = _LRUVolCache(ram_cache_items)

    def __len__(self) -> int:
        return len(self.items)

    def _load_pair(self, uid: str, img_path: str, lab_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.cache_img.get(uid)
        lab = self.cache_lab.get(uid)
        if img is None or lab is None:
            img_path2 = _resolve_cached_path(uid, img_path, self.cache_dir, "image")
            lab_path2 = _resolve_cached_path(uid, lab_path, self.cache_dir, "label")
            img = torch.load(img_path2, map_location="cpu")  # [1,D,H,W]
            lab = torch.load(lab_path2, map_location="cpu")  # [D,H,W]
            if img.dtype not in (torch.float16, torch.float32):
                img = img.float()
            if lab.dtype != torch.uint8:
                lab = lab.to(torch.uint8)
            self.cache_img.put(uid, img)
            self.cache_lab.put(uid, lab)
        return img, lab

    def _choose_center(self, lab: torch.Tensor, rng: np.random.Generator) -> Tuple[int, int, int]:
        if not self.training:
            D, H, W = lab.shape
            return (D // 2, H // 2, W // 2)

        want_fg = (rng.random() < self.pos_ratio)
        if self.min_fg_voxels <= 0:
            return _random_center_from_label(lab, want_fg=want_fg, rng=rng)

        best_c = None
        best_fg = -1
        for _ in range(self.max_center_tries):
            c = _random_center_from_label(lab, want_fg=want_fg, rng=rng)
            lab_p = _crop_around_center_torch(lab, c, self.patch_size)
            fg = int((lab_p > 0).sum().item())
            if fg >= self.min_fg_voxels:
                return c
            if fg > best_fg:
                best_fg = fg
                best_c = c
        return best_c if best_c is not None else _random_center_from_label(lab, want_fg=True, rng=rng)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        uid = it["uid"]
        base = self.seed + idx * 10007
        if self.training:
            base += int(time.time() * 1000) % 100000
        rng = np.random.default_rng(base)

        img, lab = self._load_pair(uid, it["image_pt"], it["label_pt"])  # img [1,D,H,W], lab [D,H,W]
        img = _pad_to_min_shape_torch(img, self.patch_size, pad_value=0.0)
        lab = _pad_to_min_shape_torch(lab, self.patch_size, pad_value=0)

        center = self._choose_center(lab, rng)
        img_p = _crop_around_center_torch(img, center, self.patch_size)
        lab_p = _crop_around_center_torch(lab, center, self.patch_size)

        if self.training:
            img_p, lab_p = _augment_patch(img_p, lab_p.long(), rng=rng)
        else:
            lab_p = lab_p.long()

        return {"image": img_p.contiguous(), "label": lab_p.contiguous(), "uid": uid}


# -------------------------
# Losses (Dice/Tversky/clDice + scheduled CE weight + optional boundary)
# -------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = float(smooth)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
        C = logits.shape[1]
        probs = torch.softmax(logits.float(), dim=1)
        onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
        if (not self.include_background) and C > 1:
            probs = probs[:, 1:]
            onehot = onehot[:, 1:]
        dims = tuple(range(2, probs.ndim))
        inter = torch.sum(probs * onehot, dim=dims)
        den = torch.sum(probs + onehot, dim=dims)
        dice = (2.0 * inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5, include_background: bool = False):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
        C = logits.shape[1]
        probs = torch.softmax(logits.float(), dim=1)
        onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
        if (not self.include_background) and C > 1:
            probs = probs[:, 1:]
            onehot = onehot[:, 1:]
        dims = tuple(range(2, probs.ndim))
        tp = torch.sum(probs * onehot, dim=dims)
        fp = torch.sum(probs * (1 - onehot), dim=dims)
        fn = torch.sum((1 - probs) * onehot, dim=dims)
        t = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - t.mean()


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return torch.nan_to_num(skel, nan=0.0, posinf=0.0, neginf=0.0)


class SoftclDiceLoss(nn.Module):
    def __init__(self, iters: int = 6, smooth: float = 1e-5):
        super().__init__()
        self.iters = int(iters)
        self.smooth = float(smooth)

    def set_iters(self, iters: int) -> None:
        self.iters = int(iters)

    def forward(self, probs_fg: torch.Tensor, targets_fg: torch.Tensor) -> torch.Tensor:
        # fp32
        probs_fg = probs_fg.float()
        targets_fg = targets_fg.float()
        skel_p = _soft_skel(probs_fg, self.iters)
        skel_t = _soft_skel(targets_fg, self.iters)
        tprec = (torch.sum(skel_p * targets_fg) + self.smooth) / (torch.sum(skel_p) + self.smooth)
        tsens = (torch.sum(skel_t * probs_fg) + self.smooth) / (torch.sum(skel_t * probs_fg) + self.smooth)
        cl = (2 * tprec * tsens) / (tprec + tsens + self.smooth)
        cl = torch.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)
        return 1.0 - cl


class BoundaryLoss(nn.Module):
    """
    Lightweight boundary penalty using morphological gradient.
    Helps reduce vessel thickening without expensive distance transforms.
    """
    def __init__(self):
        super().__init__()

    def forward(self, probs_fg: torch.Tensor, targets_fg: torch.Tensor) -> torch.Tensor:
        # probs_fg/targets_fg: [B,1,D,H,W] in fp32
        p = probs_fg.float()
        t = targets_fg.float()
        t_d = _soft_dilate(t)
        t_e = _soft_erode(t)
        t_edge = torch.clamp(t_d - t_e, 0.0, 1.0)

        p_d = _soft_dilate(p)
        p_e = _soft_erode(p)
        p_edge = torch.clamp(p_d - p_e, 0.0, 1.0)

        return F.l1_loss(p_edge, t_edge)


class TubularLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_tversky: float,
        w_dice: float,
        w_cldice: float,
        w_ce: float,
        w_bnd: float,
        alpha: float,
        beta: float,
        include_background: bool,
        cldice_iters: int,
        ce_weight_fg: float,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.w_tversky = float(w_tversky)
        self.w_dice = float(w_dice)
        self.w_cldice = float(w_cldice)
        self.w_ce = float(w_ce)
        self.w_bnd = float(w_bnd)

        self.tv = TverskyLoss(alpha=alpha, beta=beta, include_background=include_background)
        self.dice = SoftDiceLoss(include_background=include_background)
        self.cldice = SoftclDiceLoss(iters=cldice_iters)
        self.bnd = BoundaryLoss()

        # CE weights (binary expected here)
        if self.num_classes == 2:
            w = torch.tensor([1.0, float(ce_weight_fg)], dtype=torch.float32)
            self.ce = nn.CrossEntropyLoss(weight=w)
        else:
            self.ce = nn.CrossEntropyLoss()

    def set_ce_weight_fg(self, w_fg: float) -> None:
        if self.num_classes == 2:
            w = torch.tensor([1.0, float(w_fg)], dtype=torch.float32, device=self.ce.weight.device if self.ce.weight is not None else None)
            self.ce = nn.CrossEntropyLoss(weight=w)

    def set_cldice_iters(self, iters: int) -> None:
        self.cldice.set_iters(iters)

    def set_w_cldice(self, w: float) -> None:
        self.w_cldice = float(w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)

        loss = logits.new_tensor(0.0)

        if self.w_tversky > 0:
            loss = loss + self.w_tversky * self.tv(logits, targets)
        if self.w_dice > 0:
            loss = loss + self.w_dice * self.dice(logits, targets)
        if self.w_ce > 0:
            loss = loss + self.w_ce * self.ce(logits.float(), targets.long())

        # clDice + boundary in fp32
        if (self.w_cldice > 0) or (self.w_bnd > 0):
            probs = torch.softmax(logits.float(), dim=1)
            probs_fg = 1.0 - probs[:, 0:1]
            tgt_fg = (targets > 0).float().unsqueeze(1)
            if self.w_cldice > 0:
                loss = loss + self.w_cldice * self.cldice(probs_fg, tgt_fg)
            if self.w_bnd > 0:
                loss = loss + self.w_bnd * self.bnd(probs_fg, tgt_fg)

        return torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)


# -------------------------
# Model blocks: GN convs + BiMamba (multi-axis) + XYZ-Snake
# -------------------------
def _gn(ch: int, groups: int = 16) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch, gn_groups),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch, gn_groups),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 16):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            _gn(out_ch, gn_groups),
            nn.SiLU(),
        )
        self.block = ConvBlock(out_ch, out_ch, gn_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


def _make_base_grid_3d(B: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    zz = torch.linspace(-1, 1, D, device=device)
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)  # (D,H,W,3)
    return grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)


class SnakeRefine3DXYZ(nn.Module):
    """
    XYZ snake refinement using grid_sample.
    NOTE: This is expensive at full resolution; enable it only for lower-res decoder stages.
    """
    def __init__(self, channels: int, K: int = 3, offset_scale: float = 0.25):
        super().__init__()
        assert K >= 3 and K % 2 == 1
        self.K = int(K)
        self.half = self.K // 2
        self.offset_scale = float(offset_scale)

        # 3*(K-1) offsets: (dx,dy,dz) for each step
        self.offset_pred = nn.Conv3d(channels, 3 * (self.K - 1), kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * self.K, channels, kernel_size=1, bias=False),
            _gn(channels, 16),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        delta = torch.tanh(self.offset_pred(x)) * self.offset_scale
        delta = delta.view(B, 3, self.K - 1, D, H, W)  # [B,3,K-1,D,H,W]
        base = _make_base_grid_3d(B, D, H, W, x.device)
        grids = [base]

        # positive direction cumulative offsets
        cum = torch.zeros((B, 3, 1, D, H, W), device=x.device, dtype=delta.dtype)
        for s in range(1, self.half + 1):
            cum = cum + delta[:, :, s - 1:s, ...]
            g = base.clone()
            # x,y,z in grid are 0,1,2
            g[..., 0] = g[..., 0] + (cum[:, 0, 0] * (2.0 / max(1, W - 1)))
            g[..., 1] = g[..., 1] + (cum[:, 1, 0] * (2.0 / max(1, H - 1)))
            g[..., 2] = g[..., 2] + (cum[:, 2, 0] * (2.0 / max(1, D - 1)))
            grids.append(g)

        # negative direction cumulative offsets
        cum = torch.zeros((B, 3, 1, D, H, W), device=x.device, dtype=delta.dtype)
        for s in range(1, self.half + 1):
            cum = cum + delta[:, :, s - 1:s, ...]
            g = base.clone()
            g[..., 0] = g[..., 0] - (cum[:, 0, 0] * (2.0 / max(1, W - 1)))
            g[..., 1] = g[..., 1] - (cum[:, 1, 0] * (2.0 / max(1, H - 1)))
            g[..., 2] = g[..., 2] - (cum[:, 2, 0] * (2.0 / max(1, D - 1)))
            grids.append(g)

        sampled = [F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True) for g in grids]
        y = torch.cat(sampled, dim=1)
        y = self.fuse(y)
        return x + y


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_snake: bool, snake_k: int, gn_groups: int = 16):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, gn_groups)
        self.snake = SnakeRefine3DXYZ(out_ch, K=snake_k) if use_snake else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad/crop to match skip
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return self.snake(x)


def _parse_stages(s: str) -> Set[str]:
    out: Set[str] = set()
    for t in s.split(","):
        t = t.strip()
        if t:
            out.add(t)
    return out


def _parse_mamba_axes(s: str) -> List[str]:
    # e.g. "dhw,hwd,wdh"
    axes = []
    for t in s.split(","):
        t = t.strip().lower()
        if t:
            if set(t) != set("dhw") or len(t) != 3:
                raise ValueError(f"Bad mamba axis order: {t} (use permutations of 'd','h','w')")
            axes.append(t)
    return axes


def _permute_for_axis(x: torch.Tensor, order: str) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    # x: [B,C,D,H,W]
    # return x_perm, inv permutation to restore
    axes = {'d': 2, 'h': 3, 'w': 4}
    perm = [0, 1, axes[order[0]], axes[order[1]], axes[order[2]]]
    inv = [0, 1, 0, 0, 0]
    for i, p in enumerate(perm):
        inv[p] = i
    x_p = x.permute(*perm).contiguous()
    # return inv spatial mapping for restore (we only need spatial inverse indices)
    # inv gives positions of original dims in permuted tensor
    return x_p, (inv[2], inv[3], inv[4])


class BiMambaSSM(nn.Module):
    """
    Bidirectional Mamba block.
    Optional multi-axis scanning by permuting (D,H,W) orderings at bottleneck.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0, axes: List[str] = ["dhw"]):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.axes = axes

        from mamba_ssm import Mamba  # must work

        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def _run_seq(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B,L,C]
        y = self.norm(seq)
        y_f = self.mamba_fwd(y)
        y_r = torch.flip(self.mamba_bwd(torch.flip(y, dims=(1,))), dims=(1,))
        y = y_f + y_r
        y = self.dropout(y)
        return seq + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,C,D,H,W]
        B, C, D, H, W = x.shape
        outs = []

        for ax in self.axes:
            x_p, inv_sp = _permute_for_axis(x, ax)  # [B,C,D',H',W']
            _, _, Dp, Hp, Wp = x_p.shape
            seq = x_p.permute(0, 2, 3, 4, 1).reshape(B, Dp * Hp * Wp, C)  # [B,L,C]
            seq = self._run_seq(seq)
            y = seq.reshape(B, Dp, Hp, Wp, C).permute(0, 4, 1, 2, 3).contiguous()
            # restore to original D,H,W
            # inv_sp are indices of original spatial dims inside permuted tensor
            # current y dims: [B,C,sp0,sp1,sp2] where sp0 corresponds to ax[0], etc
            # we need permute back so that dims become [B,C,D,H,W]
            # Build restore perm:
            # y currently has spatial dims in order ax[0],ax[1],ax[2]
            # We want order d,h,w => compute indices in current:
            cur = {'d': 2 + ax.index('d'), 'h': 2 + ax.index('h'), 'w': 2 + ax.index('w')}
            y = y.permute(0, 1, cur['d'], cur['h'], cur['w']).contiguous()
            outs.append(y)

        out = torch.stack(outs, dim=0).mean(dim=0)
        return out


class MambaSnakeUNet3D(nn.Module):
    """
    4-down U-Net:
      x0 full
      x1 /2
      x2 /4
      x3 /8
      x4 /16 (bottleneck grid)
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        base: int = 32,
        gn_groups: int = 16,
        mamba_layers: int = 4,
        mamba_axes: List[str] = ["dhw", "hwd", "wdh"],
        snake_stages: Iterable[str] = ("u3", "u2"),  # lower-res only by default (VRAM!)
        snake_k: int = 3,
        checkpoint_bottleneck: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.checkpoint_bottleneck = bool(checkpoint_bottleneck)

        self.stem = ConvBlock(in_ch, base, gn_groups)
        self.d1 = Down(base, base * 2, gn_groups)
        self.d2 = Down(base * 2, base * 4, gn_groups)
        self.d3 = Down(base * 4, base * 8, gn_groups)
        self.d4 = Down(base * 8, base * 16, gn_groups)

        bott_dim = base * 16
        self.bott_pre = ConvBlock(bott_dim, bott_dim, gn_groups)
        self.mamba = nn.ModuleList([BiMambaSSM(bott_dim, axes=mamba_axes) for _ in range(int(mamba_layers))])
        self.bott_post = ConvBlock(bott_dim, bott_dim, gn_groups)

        stages = set(snake_stages)
        self.u4 = Up(bott_dim, base * 8,  base * 8,  use_snake=("u4" in stages), snake_k=snake_k, gn_groups=gn_groups)  # /8
        self.u3 = Up(base * 8, base * 4,  base * 4,  use_snake=("u3" in stages), snake_k=snake_k, gn_groups=gn_groups)  # /4
        self.u2 = Up(base * 4, base * 2,  base * 2,  use_snake=("u2" in stages), snake_k=snake_k, gn_groups=gn_groups)  # /2
        self.u1 = Up(base * 2, base,      base,      use_snake=("u1" in stages), snake_k=snake_k, gn_groups=gn_groups)  # /1

        self.head = nn.Conv3d(base, num_classes, 1)

        # deep supervision aux heads
        self.aux2 = nn.Conv3d(base * 2, num_classes, 1)  # /2
        self.aux3 = nn.Conv3d(base * 4, num_classes, 1)  # /4
        self.aux4 = nn.Conv3d(base * 8, num_classes, 1)  # /8

    def _run_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bott_pre(x)
        if self.checkpoint_bottleneck and self.training:
            for blk in self.mamba:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.mamba:
                x = blk(x)
        x = self.bott_post(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.stem(x)     # /1
        x1 = self.d1(x0)      # /2
        x2 = self.d2(x1)      # /4
        x3 = self.d3(x2)      # /8
        x4 = self.d4(x3)      # /16

        b = self._run_bottleneck(x4)

        y4 = self.u4(b, x3)   # /8
        y3 = self.u3(y4, x2)  # /4
        y2 = self.u2(y3, x1)  # /2
        y1 = self.u1(y2, x0)  # /1

        out = {"logits": self.head(y1)}
        if self.training and self.deep_supervision:
            out["aux2"] = self.aux2(y2)  # /2
            out["aux3"] = self.aux3(y3)  # /4
            out["aux4"] = self.aux4(y4)  # /8
        return out


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def dice_fg(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5) -> float:
    # foreground dice for class 1 (binary)
    if logits.shape[2:] != targets.shape[-3:]:
        logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
    probs = torch.softmax(logits.float(), dim=1)
    pred = (torch.argmax(probs, dim=1) > 0).float()
    tgt = (targets > 0).float()
    inter = (pred * tgt).sum()
    den = pred.sum() + tgt.sum()
    return float(((2 * inter + eps) / (den + eps)).item())


# -------------------------
# Schedules (warmup + cosine + loss schedules)
# -------------------------
def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def linear_schedule(epoch: int, e0: int, e1: int, v0: float, v1: float) -> float:
    if epoch <= e0:
        return float(v0)
    if epoch >= e1:
        return float(v1)
    t = (epoch - e0) / max(1, (e1 - e0))
    return float(v0 + t * (v1 - v0))


# -------------------------
# Train config
# -------------------------
@dataclass
class TrainConfig:
    cache_dir: Path
    out_dir: Path

    num_classes: int = 2
    epochs: int = 300

    patch_size: Tuple[int, int, int] = (160, 160, 160)
    batch_size: int = 1
    accum_steps: int = 2
    num_workers: int = 8

    base_lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 10

    seed: int = 42
    gpu: int = 0

    amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"

    # model
    base_ch: int = 32
    gn_groups: int = 16
    mamba_layers: int = 4
    mamba_axes: List[str] = None
    snake_stages: Set[str] = None
    snake_k: int = 3
    checkpoint_bottleneck: bool = True
    deep_supervision: bool = True

    # sampling
    pos_ratio: float = 0.97
    min_fg_voxels: int = 256
    ram_cache_items: int = 0

    # loss weights
    w_tversky: float = 0.55
    w_dice: float = 0.20
    w_ce: float = 0.15
    w_bnd: float = 0.10

    # clDice scheduled
    w_cldice_start: float = 0.00
    w_cldice_end: float = 0.20
    cldice_iters_start: int = 6
    cldice_iters_end: int = 18
    cldice_ramp_e0: int = 30
    cldice_ramp_e1: int = 140

    # CE fg weight scheduled
    ce_w_fg_start: float = 12.0
    ce_w_fg_end: float = 5.0
    ce_ramp_e0: int = 10
    ce_ramp_e1: int = 80

    alpha: float = 0.3
    beta: float = 0.7
    include_background_losses: bool = False

    grad_clip: float = 0.8
    ema: float = 0.0  # keep 0 by default (EMA costs memory/time)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].data)


def _grads_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _autocast_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("amp dtype must be bf16 or fp16")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    loss_obj: TubularLoss,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    total_steps: int,
) -> Tuple[float, int]:
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    amp_dtype = _autocast_dtype(cfg.amp_dtype)

    for step, batch in enumerate(loader, start=1):
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        # update LR per step (warmup+cosine)
        lr = cosine_with_warmup(global_step, total_steps, cfg.warmup_epochs * len(loader), cfg.base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast(device_type="cuda", enabled=cfg.amp, dtype=amp_dtype):
            outputs = model(img)
            logits = outputs["logits"]
            loss_main = loss_obj(logits, lab)

            # deep supervision
            if ("aux2" in outputs) and ("aux3" in outputs) and ("aux4" in outputs):
                lab2 = F.interpolate(lab.unsqueeze(1).float(), scale_factor=0.5, mode="nearest").squeeze(1).long()
                lab3 = F.interpolate(lab.unsqueeze(1).float(), scale_factor=0.25, mode="nearest").squeeze(1).long()
                lab4 = F.interpolate(lab.unsqueeze(1).float(), scale_factor=0.125, mode="nearest").squeeze(1).long()
                loss2 = loss_obj(outputs["aux2"], lab2)
                loss3 = loss_obj(outputs["aux3"], lab3)
                loss4 = loss_obj(outputs["aux4"], lab4)
                loss = loss_main + 0.5 * loss2 + 0.25 * loss3 + 0.125 * loss4
            else:
                loss = loss_main

            loss = loss / max(1, cfg.accum_steps)

        if not torch.isfinite(loss):
            print("[WARN] non-finite loss; skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            global_step += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step % cfg.accum_steps) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            if not _grads_finite(model):
                print("[WARN] non-finite grads; skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                global_step += 1
                continue

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total += float(loss.item()) * max(1, cfg.accum_steps)
        n += 1
        global_step += 1

    return total / max(1, n), global_step


@torch.no_grad()
def validate_patchwise(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_obj: TubularLoss,
    cfg: TrainConfig,
) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    n = 0
    dfg = 0.0
    amp_dtype = _autocast_dtype(cfg.amp_dtype)

    for batch in loader:
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        with torch.amp.autocast(device_type="cuda", enabled=cfg.amp, dtype=amp_dtype):
            outputs = model(img)
            loss = loss_obj(outputs["logits"], lab)

        total += float(loss.item())
        dfg += dice_fg(outputs["logits"], lab)
        n += 1

    return (total / max(1, n)), (dfg / max(1, n))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-epochs", type=int, default=10)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--amp-dtype", type=str, choices=["bf16", "fp16"], default="bf16")

    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--gn-groups", type=int, default=16)
    ap.add_argument("--mamba-layers", type=int, default=4)
    ap.add_argument("--mamba-axes", type=str, default="dhw,hwd,wdh")

    ap.add_argument("--snake-stages", type=str, default="u3,u2", help="Use snake at these decoder stages (u4,u3,u2,u1).")
    ap.add_argument("--snake-k", type=int, default=3)
    ap.add_argument("--no-checkpoint-bottleneck", action="store_true")
    ap.add_argument("--no-deep-supervision", action="store_true")

    ap.add_argument("--pos-ratio", type=float, default=0.97)
    ap.add_argument("--min-fg-voxels", type=int, default=256)
    ap.add_argument("--ram-cache-items", type=int, default=0)

    ap.add_argument("--w-tversky", type=float, default=0.55)
    ap.add_argument("--w-dice", type=float, default=0.20)
    ap.add_argument("--w-ce", type=float, default=0.15)
    ap.add_argument("--w-bnd", type=float, default=0.10)

    ap.add_argument("--w-cldice-start", type=float, default=0.00)
    ap.add_argument("--w-cldice-end", type=float, default=0.20)
    ap.add_argument("--cldice-iters-start", type=int, default=6)
    ap.add_argument("--cldice-iters-end", type=int, default=18)
    ap.add_argument("--cldice-ramp-e0", type=int, default=30)
    ap.add_argument("--cldice-ramp-e1", type=int, default=140)

    ap.add_argument("--ce-w-fg-start", type=float, default=12.0)
    ap.add_argument("--ce-w-fg-end", type=float, default=5.0)
    ap.add_argument("--ce-ramp-e0", type=int, default=10)
    ap.add_argument("--ce-ramp-e1", type=int, default=80)

    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)

    ap.add_argument("--grad-clip", type=float, default=0.8)

    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--no-cudnn-benchmark", action="store_true")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    enable_perf_flags(tf32=bool(args.tf32), cudnn_benchmark=(not args.no_cudnn_benchmark))
    seed_everything(int(args.seed))

    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{int(args.gpu)}")

    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        epochs=int(args.epochs),
        patch_size=tuple(int(x) for x in args.patch_size),
        batch_size=int(args.batch_size),
        accum_steps=max(1, int(args.accum_steps)),
        num_workers=int(args.num_workers),
        base_lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_epochs=int(args.warmup_epochs),
        seed=int(args.seed),
        gpu=int(args.gpu),
        amp=(not args.no_amp),
        amp_dtype=str(args.amp_dtype),
        base_ch=int(args.base_ch),
        gn_groups=int(args.gn_groups),
        mamba_layers=int(args.mamba_layers),
        mamba_axes=_parse_mamba_axes(args.mamba_axes),
        snake_stages=_parse_stages(args.snake_stages) if args.snake_stages.strip().lower() != "none" else set(),
        snake_k=int(args.snake_k),
        checkpoint_bottleneck=(not args.no_checkpoint_bottleneck),
        deep_supervision=(not args.no_deep_supervision),
        pos_ratio=float(args.pos_ratio),
        min_fg_voxels=int(args.min_fg_voxels),
        ram_cache_items=int(args.ram_cache_items),
        w_tversky=float(args.w_tversky),
        w_dice=float(args.w_dice),
        w_ce=float(args.w_ce),
        w_bnd=float(args.w_bnd),
        w_cldice_start=float(args.w_cldice_start),
        w_cldice_end=float(args.w_cldice_end),
        cldice_iters_start=int(args.cldice_iters_start),
        cldice_iters_end=int(args.cldice_iters_end),
        cldice_ramp_e0=int(args.cldice_ramp_e0),
        cldice_ramp_e1=int(args.cldice_ramp_e1),
        ce_w_fg_start=float(args.ce_w_fg_start),
        ce_w_fg_end=float(args.ce_w_fg_end),
        ce_ramp_e0=int(args.ce_ramp_e0),
        ce_ramp_e1=int(args.ce_ramp_e1),
        alpha=float(args.alpha),
        beta=float(args.beta),
        grad_clip=float(args.grad_clip),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    items = list_cases_from_cache(cfg.cache_dir)
    train_items, val_items = split_train_val(items, val_ratio=0.1, seed=cfg.seed)

    train_ds = CachedPatchDataset(
        train_items, cache_dir=cfg.cache_dir,
        patch_size=cfg.patch_size, training=True,
        pos_ratio=cfg.pos_ratio, seed=cfg.seed,
        ram_cache_items=cfg.ram_cache_items,
        min_fg_voxels=cfg.min_fg_voxels,
    )
    val_ds = CachedPatchDataset(
        val_items, cache_dir=cfg.cache_dir,
        patch_size=cfg.patch_size, training=False,
        pos_ratio=0.0, seed=cfg.seed,
        ram_cache_items=cfg.ram_cache_items,
        min_fg_voxels=0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = MambaSnakeUNet3D(
        in_ch=1,
        num_classes=cfg.num_classes,
        base=cfg.base_ch,
        gn_groups=cfg.gn_groups,
        mamba_layers=cfg.mamba_layers,
        mamba_axes=cfg.mamba_axes,
        snake_stages=cfg.snake_stages,
        snake_k=cfg.snake_k,
        checkpoint_bottleneck=cfg.checkpoint_bottleneck,
        deep_supervision=cfg.deep_supervision,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    # scaler only for fp16
    scaler: Optional[torch.amp.GradScaler]
    if cfg.amp and cfg.amp_dtype == "fp16":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None

    # init loss with start schedule values
    loss_obj = TubularLoss(
        num_classes=cfg.num_classes,
        w_tversky=cfg.w_tversky,
        w_dice=cfg.w_dice,
        w_cldice=cfg.w_cldice_start,
        w_ce=cfg.w_ce,
        w_bnd=cfg.w_bnd,
        alpha=cfg.alpha,
        beta=cfg.beta,
        include_background=cfg.include_background_losses,
        cldice_iters=cfg.cldice_iters_start,
        ce_weight_fg=cfg.ce_w_fg_start,
    ).to(device)

    total_steps = cfg.epochs * len(train_loader)
    global_step = 0

    best_fg = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        # update scheduled loss params
        wcl = linear_schedule(epoch, cfg.cldice_ramp_e0, cfg.cldice_ramp_e1, cfg.w_cldice_start, cfg.w_cldice_end)
        iters = int(round(linear_schedule(epoch, cfg.cldice_ramp_e0, cfg.cldice_ramp_e1, cfg.cldice_iters_start, cfg.cldice_iters_end)))
        ce_wfg = linear_schedule(epoch, cfg.ce_ramp_e0, cfg.ce_ramp_e1, cfg.ce_w_fg_start, cfg.ce_w_fg_end)

        loss_obj.set_w_cldice(wcl)
        loss_obj.set_cldice_iters(iters)
        loss_obj.set_ce_weight_fg(ce_wfg)

        t0 = time.time()
        tr_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_obj, cfg,
            epoch=epoch, global_step=global_step, total_steps=total_steps
        )
        val_loss, val_fg = validate_patchwise(model, val_loader, loss_obj, cfg)
        dt = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_fg_dice": float(val_fg),
            "time_sec": float(dt),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "w_cldice": float(wcl),
            "cldice_iters": int(iters),
            "ce_weight_fg": float(ce_wfg),
            "patch_size": list(cfg.patch_size),
            "snake_stages": sorted(list(cfg.snake_stages)),
            "mamba_axes": cfg.mamba_axes,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)

        (cfg.out_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"epoch": epoch, "model": model.state_dict()}, cfg.out_dir / "last.pt")

        if val_fg > best_fg:
            best_fg = val_fg
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_fg": best_fg}, cfg.out_dir / "best.pt")

    print(f"Done. Best val_fg_dice={best_fg:.4f} -> {cfg.out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
