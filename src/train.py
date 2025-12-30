#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Mamba-Snake 3D tubular segmentation (cache-friendly, stable AMP, NaN-guarded)

Key fixes vs your current file:
- Works with EITHER --cache-dir (recommended) OR --dataset-dir (nnUNet_raw-like)
- Robust to moved cache paths: if cases.json contains stale absolute paths, it auto-falls back to cache_dir/images|labels/{uid}.pt
- Fixes NaNs:
  * clDice computed in fp32 (autocast disabled only for clDice)
  * fewer skeleton iterations by default
  * skip non-finite loss / non-finite grads (prevents poisoning the run)
  * optional grad clip + safer scaler usage
- Memory knobs:
  * --snake-stages (choose which Up blocks use snake)
  * --snake-k (K for snake; smaller is faster + less memory)
  * --checkpoint-bottleneck (activation checkpointing for bottleneck)
  * --amp-dtype fp16|bf16 (bf16 often more stable on A100)
- Speed knobs:
  * TF32 enabled, cudnn benchmark on
  * reduced clDice iterations, option to lower w_cldice early

Expected cached .pt format:
  img: [1, D, H, W] float16/float32 in [0,1]
  lab: [D, H, W] uint8 (0..num_classes-1)
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
import nibabel as nib

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


def enable_perf_flags() -> None:
    # TF32 speeds up conv/matmul on A100; usually safe for segmentation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# -------------------------
# Volume utils (NIfTI path)
# -------------------------
def _maybe_squeeze_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    return arr


def _normalize01(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)
    img = img - img.min()
    den = img.max()
    if den < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img / den).astype(np.float32)


def _nib_load_canonical(path: Path) -> nib.Nifti1Image:
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)  # RAS
    return nii


def _to_dhw(arr_xyz: np.ndarray) -> np.ndarray:
    # (X,Y,Z) -> (Z,Y,X) == (D,H,W)
    return np.transpose(arr_xyz, (2, 1, 0)).copy()


# -------------------------
# Patch ops
# -------------------------
def _pad_to_min_shape_torch(vol: torch.Tensor, target_dhw: Tuple[int, int, int], pad_value: float = 0.0) -> torch.Tensor:
    """
    vol: [D,H,W] or [C,D,H,W]
    Pads symmetrically to at least target_dhw.
    """
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

    # F.pad expects last dims first: (W_left, W_right, H_left, H_right, D_left, D_right)
    pad = (pw_before, pw_after, ph_before, ph_after, pd_before, pd_after)
    if has_c:
        return F.pad(vol, pad, mode="constant", value=pad_value)
    else:
        return F.pad(vol.unsqueeze(0), pad, mode="constant", value=pad_value).squeeze(0)


def _crop_around_center_torch(vol: torch.Tensor, center_dhw: Tuple[int, int, int], patch_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """
    vol: [D,H,W] or [C,D,H,W]
    """
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
    """
    label: [D,H,W] int
    """
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
    """
    img: [1,D,H,W] float in [0,1]
    lab: [D,H,W] long
    """
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
# Listing cases
# -------------------------
def list_cases_from_dataset(dataset_dir: Path) -> List[Dict[str, str]]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise RuntimeError(f"Missing imagesTr/labelsTr in {dataset_dir}")
    items: List[Dict[str, str]] = []
    for img_path in sorted(images_dir.glob("*_0000.nii.gz")):
        uid = img_path.name.replace("_0000.nii.gz", "")
        lab_path = labels_dir / f"{uid}.nii.gz"
        if lab_path.exists():
            items.append({"uid": uid, "image": str(img_path), "label": str(lab_path)})
    if not items:
        raise RuntimeError(f"No training pairs found under {images_dir} and {labels_dir}")
    return items


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


# -------------------------
# In-RAM LRU cache (per worker)
# -------------------------
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
    """
    If cases.json has absolute paths from another container (/workspace/...), they may not exist here.
    We try:
      1) original path
      2) cache_dir / relative(path)
      3) cache_dir / (images|labels) / f"{uid}.pt"
    """
    pp = Path(p)
    if pp.exists():
        return str(pp)

    # try cache_dir + relative
    if pp.is_absolute():
        try_rel = cache_dir / pp.relative_to(pp.anchor)
        if try_rel.exists():
            return str(try_rel)
    else:
        try_rel = cache_dir / pp
        if try_rel.exists():
            return str(try_rel)

    # canonical fallback
    if kind == "image":
        alt = cache_dir / "images" / f"{uid}.pt"
    else:
        alt = cache_dir / "labels" / f"{uid}.pt"
    return str(alt)


# -------------------------
# Datasets
# -------------------------
class CachedPatchDataset(torch.utils.data.Dataset):
    """
    Uses cached .pt volumes:
      img: [1,D,H,W] float16/float32 in [0,1]
      lab: [D,H,W] uint8
    """
    def __init__(
        self,
        items: List[Dict[str, str]],
        cache_dir: Path,
        patch_size: Tuple[int, int, int],
        training: bool,
        pos_ratio: float = 0.90,
        seed: int = 42,
        ram_cache_items: int = 0,
        min_fg_voxels: int = 0,
        max_center_tries: int = 12,
    ):
        self.items = items
        self.cache_dir = Path(cache_dir)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.training = bool(training)
        self.pos_ratio = float(pos_ratio)
        self.seed = int(seed)
        self.min_fg_voxels = int(min_fg_voxels)
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

            if img.ndim != 4 or lab.ndim != 3:
                raise RuntimeError(f"Bad cached shapes for uid={uid}: img={tuple(img.shape)} lab={tuple(lab.shape)}")
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

        # enforce min fg voxels in the resulting patch (helps ultra-sparse fg)
        D, H, W = lab.shape
        for _ in range(self.max_center_tries):
            c = _random_center_from_label(lab, want_fg=want_fg, rng=rng)
            lab_p = _crop_around_center_torch(lab, c, self.patch_size)
            if int((lab_p > 0).sum().item()) >= self.min_fg_voxels:
                return c

        # fallback: best-effort center
        return _random_center_from_label(lab, want_fg=want_fg, rng=rng)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        uid = it["uid"]

        # per-sample rng (time-mixed for training variety)
        base = self.seed + idx * 10007
        if self.training:
            base += int(time.time() * 1000) % 100000
        rng = np.random.default_rng(base)

        img, lab = self._load_pair(uid, it["image_pt"], it["label_pt"])  # img [1,D,H,W], lab [D,H,W]

        img = _pad_to_min_shape_torch(img, self.patch_size, pad_value=0.0)
        lab = _pad_to_min_shape_torch(lab, self.patch_size, pad_value=0)

        center = self._choose_center(lab, rng)

        img_p = _crop_around_center_torch(img, center, self.patch_size)  # [1,D,H,W]
        lab_p = _crop_around_center_torch(lab, center, self.patch_size)  # [D,H,W]

        if self.training:
            img_p, lab_p = _augment_patch(img_p, lab_p.long(), rng=rng)
        else:
            lab_p = lab_p.long()

        return {"image": img_p.contiguous(), "label": lab_p.contiguous(), "uid": uid}


class NiftiPatchDataset(torch.utils.data.Dataset):
    """
    For --dataset-dir usage (nnUNet_raw-like).
    Loads NIfTI on the fly. Slower than cache, but kept for completeness.
    """
    def __init__(
        self,
        items: List[Dict[str, str]],
        patch_size: Tuple[int, int, int],
        training: bool,
        pos_ratio: float = 0.90,
        seed: int = 42,
        min_fg_voxels: int = 0,
        max_center_tries: int = 12,
    ):
        self.items = items
        self.patch_size = tuple(int(x) for x in patch_size)
        self.training = bool(training)
        self.pos_ratio = float(pos_ratio)
        self.seed = int(seed)
        self.min_fg_voxels = int(min_fg_voxels)
        self.max_center_tries = int(max_center_tries)

    def __len__(self) -> int:
        return len(self.items)

    def _load_pair(self, uid: str, img_path: str, lab_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        img_nii = _nib_load_canonical(Path(img_path))
        lab_nii = _nib_load_canonical(Path(lab_path))

        img = np.asarray(img_nii.dataobj).astype(np.float32)
        lab = np.asarray(lab_nii.dataobj).astype(np.int64)
        img = _maybe_squeeze_4d(img)
        lab = _maybe_squeeze_4d(lab)
        if img.ndim != 3 or lab.ndim != 3:
            raise RuntimeError(f"Expected 3D volumes, got img={img.shape}, lab={lab.shape} for uid={uid}")

        img = _to_dhw(img)
        lab = _to_dhw(lab).astype(np.uint8)
        img = _normalize01(img)

        img_t = torch.from_numpy(img)[None, ...]      # [1,D,H,W]
        lab_t = torch.from_numpy(lab).to(torch.uint8) # [D,H,W]
        return img_t, lab_t

    def _choose_center(self, lab: torch.Tensor, rng: np.random.Generator) -> Tuple[int, int, int]:
        if not self.training:
            D, H, W = lab.shape
            return (D // 2, H // 2, W // 2)

        want_fg = (rng.random() < self.pos_ratio)
        if self.min_fg_voxels <= 0:
            return _random_center_from_label(lab, want_fg=want_fg, rng=rng)

        for _ in range(self.max_center_tries):
            c = _random_center_from_label(lab, want_fg=want_fg, rng=rng)
            lab_p = _crop_around_center_torch(lab, c, self.patch_size)
            if int((lab_p > 0).sum().item()) >= self.min_fg_voxels:
                return c
        return _random_center_from_label(lab, want_fg=want_fg, rng=rng)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        uid = it["uid"]

        base = self.seed + idx * 10007
        if self.training:
            base += int(time.time() * 1000) % 100000
        rng = np.random.default_rng(base)

        img, lab = self._load_pair(uid, it["image"], it["label"])

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
# Losses
# -------------------------
def _safe_softmax(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # compute in fp32 for safety, then cast back
    orig_dtype = logits.dtype
    p = torch.softmax(logits.float(), dim=dim)
    return p.to(orig_dtype)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = float(smooth)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)

        num_classes = logits.shape[1]
        probs = torch.softmax(logits.float(), dim=1)  # fp32
        onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        if (not self.include_background) and num_classes > 1:
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

        num_classes = logits.shape[1]
        probs = torch.softmax(logits.float(), dim=1)  # fp32
        onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        if (not self.include_background) and num_classes > 1:
            probs = probs[:, 1:]
            onehot = onehot[:, 1:]

        dims = tuple(range(2, probs.ndim))
        tp = torch.sum(probs * onehot, dim=dims)
        fp = torch.sum(probs * (1 - onehot), dim=dims)
        fn = torch.sum((1 - probs) * onehot, dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


# Soft morphological ops (fp32 recommended)
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img: torch.Tensor, iter_: int = 6) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    skel = torch.nan_to_num(skel, nan=0.0, posinf=0.0, neginf=0.0)
    return skel


class SoftclDiceLoss(nn.Module):
    def __init__(self, iterations: int = 6, smooth: float = 1e-5):
        super().__init__()
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(self, probs_fg: torch.Tensor, targets_fg: torch.Tensor) -> torch.Tensor:
        # expects fp32 inputs
        probs_fg = probs_fg.float()
        targets_fg = targets_fg.float()

        skel_p = _soft_skel(probs_fg, self.iterations)
        skel_t = _soft_skel(targets_fg, self.iterations)

        tprec = (torch.sum(skel_p * targets_fg) + self.smooth) / (torch.sum(skel_p) + self.smooth)
        tsens = (torch.sum(skel_t * probs_fg) + self.smooth) / (torch.sum(skel_t * probs_fg) + self.smooth)
        cldice = (2 * tprec * tsens) / (tprec + tsens + self.smooth)
        cldice = torch.nan_to_num(cldice, nan=0.0, posinf=0.0, neginf=0.0)
        return 1.0 - cldice


class TubularSegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_tversky=0.6,
        w_dice=0.2,
        w_cldice=0.2,
        w_ce=0.0,
        alpha=0.3,
        beta=0.7,
        include_background_losses: bool = False,
        cldice_iters: int = 6,
        ce_weight_fg: Optional[float] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        self.w_tv = float(w_tversky)
        self.w_dice = float(w_dice)
        self.w_cl = float(w_cldice)
        self.w_ce = float(w_ce)

        self.tv = TverskyLoss(alpha=alpha, beta=beta, include_background=include_background_losses)
        self.dice = SoftDiceLoss(include_background=include_background_losses)
        self.cldice = SoftclDiceLoss(iterations=cldice_iters)
        if ce_weight_fg is not None and self.num_classes == 2:
            w = torch.tensor([1.0, float(ce_weight_fg)], dtype=torch.float32)
            self.ce = nn.CrossEntropyLoss(weight=w)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)

        loss = logits.new_tensor(0.0)

        if self.w_tv > 0:
            loss = loss + self.w_tv * self.tv(logits, targets)
        if self.w_dice > 0:
            loss = loss + self.w_dice * self.dice(logits, targets)
        if self.w_ce > 0:
            loss = loss + self.w_ce * self.ce(logits.float(), targets.long())

        # IMPORTANT: clDice in fp32, with autocast disabled at call site
        if self.w_cl > 0:
            probs = torch.softmax(logits.float(), dim=1)  # fp32
            probs_fg = 1.0 - probs[:, 0:1]               # [B,1,D,H,W]
            tgt_fg = (targets > 0).float().unsqueeze(1)  # [B,1,D,H,W]
            loss = loss + self.w_cl * self.cldice(probs_fg, tgt_fg)

        loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        return loss


# -------------------------
# Model blocks
# -------------------------
class MambaSSMBlock(nn.Module):
    """
    Requires mamba_ssm installed and importable (with working CUDA extensions).
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        try:
            from mamba_ssm import Mamba  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mamba_ssm is required. Your environment must import it without errors.\n"
                "If import fails, your run will silently fall apart (or crash). Fix mamba first."
            ) from e

        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,C,D,H,W] -> seq [B, L, C]
        b, c, d, h, w = x.shape
        seq = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        y = self.norm(seq)
        y = self.mamba(y)
        y = self.dropout(y)
        out = (seq + y).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return out


def _make_base_grid_3d(B: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    zz = torch.linspace(-1, 1, D, device=device)
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)  # (D,H,W,3) ordering for grid_sample
    return grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B,D,H,W,3)


class SnakeRefine3D(nn.Module):
    """
    Snake refinement along z (depth) by default (cheap + helpful for vessels).
    Smaller K is faster and uses less VRAM.
    """
    def __init__(self, channels: int, K: int = 3, offset_scale: float = 0.25):
        super().__init__()
        assert K >= 3 and K % 2 == 1
        self.K = K
        self.half = K // 2
        self.offset_scale = float(offset_scale)

        # predict (K-1) offsets, applied cumulatively
        self.offset_pred = nn.Conv3d(channels, (K - 1), kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * K, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        delta = torch.tanh(self.offset_pred(x)) * self.offset_scale  # [B, K-1, D,H,W]
        base = _make_base_grid_3d(B, D, H, W, x.device)
        grids = [base]

        # positive direction
        cum = 0.0
        for s in range(1, self.half + 1):
            cum = cum + delta[:, s - 1:s, ...]
            g = base.clone()
            g[..., 2] = g[..., 2] + (cum.squeeze(1) * (2.0 / max(1, D - 1)))
            grids.append(g)

        # negative direction
        cum = 0.0
        for s in range(1, self.half + 1):
            cum = cum + delta[:, s - 1:s, ...]
            g = base.clone()
            g[..., 2] = g[..., 2] - (cum.squeeze(1) * (2.0 / max(1, D - 1)))
            grids.append(g)

        sampled = [F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True) for g in grids]
        y = torch.cat(sampled, dim=1)
        y = self.fuse(y)
        return x + y


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
        )
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_snake: bool, snake_k: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)
        self.snake = SnakeRefine3D(out_ch, K=snake_k) if use_snake else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        x = self.snake(x)
        return x


def _parse_stages(s: str) -> Set[str]:
    # e.g. "u1,u2" -> {"u1","u2"}
    out: Set[str] = set()
    for t in s.split(","):
        t = t.strip()
        if t:
            out.add(t)
    return out


class MambaSnakeUNet3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 2,
        base: int = 32,
        mamba_layers: int = 4,
        snake_stages: Iterable[str] = ("u0", "u1", "u2"),
        snake_k: int = 3,
        checkpoint_bottleneck: bool = False,
    ):
        super().__init__()
        self.checkpoint_bottleneck = bool(checkpoint_bottleneck)

        self.stem = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)

        bott_dim = base * 8
        self.bott_pre = ConvBlock(bott_dim, bott_dim)
        self.mamba_blocks = nn.ModuleList([MambaSSMBlock(bott_dim) for _ in range(int(mamba_layers))])
        self.bott_post = ConvBlock(bott_dim, bott_dim)

        stages = set(snake_stages)
        self.u3 = Up(bott_dim, base * 8, base * 4, use_snake=("u3" in stages), snake_k=snake_k)
        self.u2 = Up(base * 4, base * 4, base * 2, use_snake=("u2" in stages), snake_k=snake_k)
        self.u1 = Up(base * 2, base * 2, base,     use_snake=("u1" in stages), snake_k=snake_k)
        self.u0 = Up(base,     base,     base,     use_snake=("u0" in stages), snake_k=snake_k)

        self.head = nn.Conv3d(base, num_classes, kernel_size=1)

    def _run_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bott_pre(x)
        if self.checkpoint_bottleneck and self.training:
            # checkpoint each block to save memory (slower)
            for blk in self.mamba_blocks:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.mamba_blocks:
                x = blk(x)
        x = self.bott_post(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)

        b = self._run_bottleneck(x3)

        y2 = self.u3(b, x3)
        y1 = self.u2(y2, x2)
        y0 = self.u1(y1, x1)
        y  = self.u0(y0, x0)
        return self.head(y)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def dice_per_class(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-5) -> List[float]:
    if logits.shape[2:] != targets.shape[-3:]:
        logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)

    probs = torch.softmax(logits.float(), dim=1)
    pred = torch.argmax(probs, dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        den = p.sum() + t.sum()
        d = (2 * inter + eps) / (den + eps)
        dices.append(float(d.item()))
    return dices


# -------------------------
# EMA
# -------------------------
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


# -------------------------
# AMP helpers (torch.amp API, no deprecation warnings)
# -------------------------
def _amp_dtype(s: str) -> torch.dtype:
    s = str(s).lower()
    if s in ("fp16", "float16", "16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown amp dtype: {s}")


class AmpContext:
    def __init__(self, enabled: bool, dtype: torch.dtype):
        self.enabled = bool(enabled)
        self.dtype = dtype

    def __enter__(self):
        if not self.enabled:
            return None
        return torch.amp.autocast(device_type="cuda", dtype=self.dtype).__enter__()

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        return torch.amp.autocast(device_type="cuda", dtype=self.dtype).__exit__(exc_type, exc, tb)


# -------------------------
# Training config
# -------------------------
@dataclass
class TrainConfig:
    dataset_dir: Optional[Path]
    cache_dir: Optional[Path]
    out_dir: Path

    num_classes: int = 2
    epochs: int = 250
    batch_size: int = 1
    accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    val_ratio: float = 0.1
    num_workers: int = 8

    patch_size: Tuple[int, int, int] = (160, 160, 160)
    seed: int = 42
    gpu: int = 0

    amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16

    base_ch: int = 32
    mamba_layers: int = 4

    snake_stages: Set[str] = None  # set in main
    snake_k: int = 3
    checkpoint_bottleneck: bool = False

    w_tversky: float = 0.65
    w_dice: float = 0.20
    w_cldice: float = 0.05
    w_ce: float = 0.10
    alpha: float = 0.3
    beta: float = 0.7

    cldice_iters: int = 6
    ce_weight_fg: Optional[float] = None

    pos_ratio: float = 0.97
    min_fg_voxels: int = 0
    grad_clip: float = 0.5
    ema: float = 0.999

    include_background_losses: bool = False
    compile: bool = False
    ram_cache_items: int = 0


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _grads_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    loss_fn: TubularSegLoss,
    cfg: TrainConfig,
    ema: Optional[EMA] = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        # forward under AMP
        with torch.amp.autocast(device_type="cuda", enabled=cfg.amp, dtype=cfg.amp_dtype):
            logits = model(img)

            # IMPORTANT: disable autocast for clDice to avoid NaNs (loss_fn expects this)
            if loss_fn.w_cl > 0:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss = loss_fn(logits, lab) / max(1, cfg.accum_steps)
            else:
                loss = loss_fn(logits, lab) / max(1, cfg.accum_steps)

        # loss finite guard
        if not torch.isfinite(loss):
            print("[WARN] non-finite loss detected; skipping this batch.")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # step?
        if (step % cfg.accum_steps) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            # grad finite guard
            if not _grads_finite(model):
                print("[WARN] non-finite grads detected; skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                continue

            # grad clip
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        total += float(loss.item()) * max(1, cfg.accum_steps)
        n += 1

    return total / max(1, n)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: TubularSegLoss,
    cfg: TrainConfig,
) -> Tuple[float, List[float]]:
    model.eval()
    total = 0.0
    n = 0
    dices_sum = np.zeros((cfg.num_classes,), dtype=np.float64)

    for batch in loader:
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        with torch.amp.autocast(device_type="cuda", enabled=cfg.amp, dtype=cfg.amp_dtype):
            logits = model(img)
            # clDice in fp32
            if loss_fn.w_cl > 0:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss = loss_fn(logits, lab)
            else:
                loss = loss_fn(logits, lab)

        loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        total += float(loss.item())
        n += 1
        dices_sum += np.array(dice_per_class(logits, lab, cfg.num_classes), dtype=np.float64)

    return (total / max(1, n)), (dices_sum / max(1, n)).tolist()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="nnUNet_raw-like dataset dir (ONLY if --cache-dir is not provided)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="cache directory containing cases.json (recommended)")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[160, 160, 160])

    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                    help="bf16 is usually more stable on A100; fp16 is faster but can NaN easier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--mamba-layers", type=int, default=4)

    ap.add_argument("--snake-stages", type=str, default="u0,u1,u2",
                    help="Comma-separated: which decoder stages use snake. e.g. u1,u2 or none")
    ap.add_argument("--snake-k", type=int, default=3, help="Snake K (odd >=3). Smaller is faster/less VRAM.")
    ap.add_argument("--checkpoint-bottleneck", action="store_true", help="Activation checkpoint bottleneck (saves VRAM, slower)")

    ap.add_argument("--w-tversky", type=float, default=0.65)
    ap.add_argument("--w-dice", type=float, default=0.20)
    ap.add_argument("--w-cldice", type=float, default=0.05)
    ap.add_argument("--w-ce", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--cldice-iters", type=int, default=6, help="Soft skeleton iterations (lower is faster + more stable)")

    ap.add_argument("--ce-weight-fg", type=float, default=None,
                    help="If set (binary only), foreground class weight for CE (helps extreme imbalance).")

    ap.add_argument("--pos-ratio", type=float, default=0.97, help="Prob sample foreground-centered patch")
    ap.add_argument("--min-fg-voxels", type=int, default=0,
                    help="If >0, tries multiple centers until patch has at least this many fg voxels (helps ultra-sparse fg).")

    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--include-background-losses", action="store_true",
                    help="Include background in Dice/Tversky (usually worse for imbalance)")

    ap.add_argument("--compile", action="store_true", help="torch.compile model (PyTorch 2.x)")
    ap.add_argument("--ram-cache-items", type=int, default=0,
                    help="per-worker LRU cache of N full volumes (0 disables). Try 4..16 if RAM allows.")

    args = ap.parse_args()

    if args.cache_dir is None and args.dataset_dir is None:
        raise SystemExit("You must provide either --cache-dir or --dataset-dir")
    if args.cache_dir is not None:
        cj = Path(args.cache_dir) / "cases.json"
        if not cj.exists():
            raise SystemExit(f"--cache-dir provided but cases.json not found: {cj}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    enable_perf_flags()
    seed_everything(int(args.seed))

    # IMPORTANT: with CUDA_VISIBLE_DEVICES, visible GPUs are re-indexed from 0
    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{int(args.gpu)}")

    amp = (not args.no_amp)
    amp_dtype = _amp_dtype(args.amp_dtype)

    # GradScaler: use only for fp16 AMP (bf16 typically does not need scaling)
    scaler: Optional[torch.amp.GradScaler]
    if amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None

    snake_stages = _parse_stages(args.snake_stages) if args.snake_stages.strip().lower() != "none" else set()

    cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        num_classes=int(args.num_classes),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        accum_steps=max(1, int(args.accum_steps)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        num_workers=int(args.num_workers),
        patch_size=tuple(int(x) for x in args.patch_size),
        seed=int(args.seed),
        gpu=int(args.gpu),
        amp=amp,
        amp_dtype=amp_dtype,
        base_ch=int(args.base_ch),
        mamba_layers=int(args.mamba_layers),
        snake_stages=snake_stages,
        snake_k=int(args.snake_k),
        checkpoint_bottleneck=bool(args.checkpoint_bottleneck),
        w_tversky=float(args.w_tversky),
        w_dice=float(args.w_dice),
        w_cldice=float(args.w_cldice),
        w_ce=float(args.w_ce),
        alpha=float(args.alpha),
        beta=float(args.beta),
        cldice_iters=int(args.cldice_iters),
        ce_weight_fg=args.ce_weight_fg,
        pos_ratio=float(args.pos_ratio),
        min_fg_voxels=int(args.min_fg_voxels),
        grad_clip=float(args.grad_clip),
        ema=float(args.ema),
        include_background_losses=bool(args.include_background_losses),
        compile=bool(args.compile),
        ram_cache_items=int(args.ram_cache_items),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # --- load items ---
    if cfg.cache_dir is not None:
        items = list_cases_from_cache(cfg.cache_dir)
    else:
        assert cfg.dataset_dir is not None
        items = list_cases_from_dataset(cfg.dataset_dir)

    train_items, val_items = split_train_val(items, val_ratio=cfg.val_ratio, seed=cfg.seed)

    # --- datasets ---
    if cfg.cache_dir is not None:
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
    else:
        train_ds = NiftiPatchDataset(
            train_items, patch_size=cfg.patch_size, training=True,
            pos_ratio=cfg.pos_ratio, seed=cfg.seed,
            min_fg_voxels=cfg.min_fg_voxels,
        )
        val_ds = NiftiPatchDataset(
            val_items, patch_size=cfg.patch_size, training=False,
            pos_ratio=0.0, seed=cfg.seed,
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
        mamba_layers=cfg.mamba_layers,
        snake_stages=cfg.snake_stages,
        snake_k=cfg.snake_k,
        checkpoint_bottleneck=cfg.checkpoint_bottleneck,
    ).to(device)

    if cfg.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed, continuing without compile: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_factor(epoch_idx: int) -> float:
        t = epoch_idx / max(1, cfg.epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    loss_fn = TubularSegLoss(
        num_classes=cfg.num_classes,
        w_tversky=cfg.w_tversky,
        w_dice=cfg.w_dice,
        w_cldice=cfg.w_cldice,
        w_ce=cfg.w_ce,
        alpha=cfg.alpha,
        beta=cfg.beta,
        include_background_losses=cfg.include_background_losses,
        cldice_iters=cfg.cldice_iters,
        ce_weight_fg=cfg.ce_weight_fg,
    ).to(device)

    ema_obj: Optional[EMA] = EMA(model, decay=cfg.ema) if (cfg.ema and cfg.ema > 0) else None

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    # Save config
    (cfg.out_dir / "config.json").write_text(json.dumps({
        "dataset_dir": str(cfg.dataset_dir) if cfg.dataset_dir is not None else None,
        "cache_dir": str(cfg.cache_dir) if cfg.cache_dir is not None else None,
        "out_dir": str(cfg.out_dir),
        "num_classes": cfg.num_classes,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "accum_steps": cfg.accum_steps,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "val_ratio": cfg.val_ratio,
        "num_workers": cfg.num_workers,
        "patch_size": list(cfg.patch_size),
        "seed": cfg.seed,
        "gpu": cfg.gpu,
        "amp": cfg.amp,
        "amp_dtype": str(cfg.amp_dtype),
        "base_ch": cfg.base_ch,
        "mamba_layers": cfg.mamba_layers,
        "snake_stages": sorted(list(cfg.snake_stages)),
        "snake_k": cfg.snake_k,
        "checkpoint_bottleneck": cfg.checkpoint_bottleneck,
        "loss": {
            "w_tversky": cfg.w_tversky,
            "w_dice": cfg.w_dice,
            "w_cldice": cfg.w_cldice,
            "w_ce": cfg.w_ce,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "cldice_iters": cfg.cldice_iters,
            "ce_weight_fg": cfg.ce_weight_fg,
            "include_background_losses": cfg.include_background_losses,
        },
        "sampling": {
            "pos_ratio": cfg.pos_ratio,
            "min_fg_voxels": cfg.min_fg_voxels,
        },
        "stability": {
            "grad_clip": cfg.grad_clip,
            "ema": cfg.ema,
        },
        "ram_cache_items": cfg.ram_cache_items,
    }, indent=2))

    for epoch in range(1, cfg.epochs + 1):
        # cosine lr
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * lr_factor(epoch - 1)

        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, cfg, ema=ema_obj)

        # validate with EMA weights if enabled
        if ema_obj is not None:
            backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            ema_obj.copy_to(model)
            val_loss, val_dice = validate(model, val_loader, loss_fn, cfg)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.data.copy_(backup[n].data)
        else:
            val_loss, val_dice = validate(model, val_loader, loss_fn, cfg)

        dt = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_dice": [float(x) for x in val_dice],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(dt),
            "gpu": cfg.gpu,
            "patch_size": list(cfg.patch_size),
            "batch": cfg.batch_size,
            "accum_steps": cfg.accum_steps,
            "base_ch": cfg.base_ch,
            "mamba_layers": cfg.mamba_layers,
            "snake_stages": sorted(list(cfg.snake_stages)),
            "snake_k": cfg.snake_k,
            "checkpoint_bottleneck": cfg.checkpoint_bottleneck,
            "cache_dir": str(cfg.cache_dir) if cfg.cache_dir is not None else None,
            "amp": cfg.amp,
            "amp_dtype": str(cfg.amp_dtype),
            "ema": cfg.ema,
            "ram_cache_items": cfg.ram_cache_items,
            "min_fg_voxels": cfg.min_fg_voxels,
        }

        history.append(rec)
        print(json.dumps(rec), flush=True)

        # save last
        torch.save({"epoch": epoch, "model": model.state_dict()}, cfg.out_dir / "last.pt")
        (cfg.out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = float(val_loss)
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_val": best_val}, cfg.out_dir / "best.pt")

    print(f"Done. Best val_loss={best_val:.6f} -> {cfg.out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
