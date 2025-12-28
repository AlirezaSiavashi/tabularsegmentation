"""
Train a 3D vessel/tubular segmentation model:

- Mamba blocks act as a global "navigator" at the bottleneck (low-res tokens).
- Dynamic Snake Conv-inspired refinement acts as a local "precise brush" in the decoder.
- Loss: Tversky + SoftDice + soft-clDice (centerline/topology)

Expected data layout (nnUNet-like):
  DATASET_DIR/
    imagesTr/
      <uid>_0000.nii.gz
    labelsTr/
      <uid>.nii.gz

Fixes:
- No MONAI Orientationd / MetaTensor issues.
- Pad to >= patch_size so crop never fails.
- Ensure logits/labels shapes match; interpolate logits if needed.
- Adds --gpu, --accum-steps, --base-ch, --mamba-layers.
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
from typing import Tuple, List, Dict

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_cases(dataset_dir: Path) -> List[Dict[str, str]]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise RuntimeError(f"Missing imagesTr/labelsTr in {dataset_dir}")
    items = []
    for img_path in sorted(images_dir.glob("*_0000.nii.gz")):
        uid = img_path.name.replace("_0000.nii.gz", "")
        lab_path = labels_dir / f"{uid}.nii.gz"
        if lab_path.exists():
            items.append({"image": str(img_path), "label": str(lab_path), "uid": uid})
    if not items:
        raise RuntimeError(f"No training pairs found under {images_dir} and {labels_dir}")
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


def _maybe_squeeze_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    return arr


def _normalize_img(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)
    img = img - img.min()
    den = img.max() - img.min()
    if den < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img / den).astype(np.float32)


def _pad_to_min_shape(vol: np.ndarray, target_dhw: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    D, H, W = vol.shape
    td, th, tw = target_dhw
    pd0 = max(0, td - D)
    ph0 = max(0, th - H)
    pw0 = max(0, tw - W)
    if pd0 == 0 and ph0 == 0 and pw0 == 0:
        return vol
    pad_before = (pd0 // 2, ph0 // 2, pw0 // 2)
    pad_after = (pd0 - pad_before[0], ph0 - pad_before[1], pw0 - pad_before[2])
    pads = ((pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]))
    return np.pad(vol, pads, mode="constant", constant_values=pad_value)


def _crop_around_center(vol: np.ndarray, center_dhw: Tuple[int, int, int], patch_dhw: Tuple[int, int, int]) -> np.ndarray:
    D, H, W = vol.shape
    pd, ph, pw = patch_dhw
    cd, ch, cw = center_dhw

    sd = int(np.clip(cd - pd // 2, 0, max(0, D - pd)))
    sh = int(np.clip(ch - ph // 2, 0, max(0, H - ph)))
    sw = int(np.clip(cw - pw // 2, 0, max(0, W - pw)))

    return vol[sd:sd + pd, sh:sh + ph, sw:sw + pw]


def _random_center_from_label(label: np.ndarray, fg: bool, rng: np.random.Generator) -> Tuple[int, int, int]:
    idx = np.argwhere(label > 0) if fg else np.argwhere(label == 0)
    if idx.size == 0:
        D, H, W = label.shape
        return (int(rng.integers(0, D)), int(rng.integers(0, H)), int(rng.integers(0, W)))
    pick = idx[int(rng.integers(0, len(idx)))]
    return (int(pick[0]), int(pick[1]), int(pick[2]))


def _augment_patch(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    for axis in (0, 1, 2):
        if rng.random() < 0.5:
            img = np.flip(img, axis=axis).copy()
            lab = np.flip(lab, axis=axis).copy()

    if rng.random() < 0.25:
        scale = float(rng.uniform(0.9, 1.1))
        shift = float(rng.uniform(-0.05, 0.05))
        img = np.clip(img * scale + shift, 0.0, 1.0)

    if rng.random() < 0.20:
        noise = rng.normal(0.0, 0.01, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return img.astype(np.float32), lab.astype(np.int64)


class NiftiPatchDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, str]], patch_size: Tuple[int, int, int], training: bool,
                 pos_ratio: float = 0.67, seed: int = 42):
        self.items = items
        self.patch_size = tuple(int(x) for x in patch_size)
        self.training = training
        self.pos_ratio = float(pos_ratio)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        rng = np.random.default_rng(self.seed + idx * 10007 + (0 if not self.training else int(time.time() * 1e3) % 100000))

        img = nib.load(it["image"]).get_fdata().astype(np.float32)
        lab = nib.load(it["label"]).get_fdata().astype(np.int64)

        img = _maybe_squeeze_4d(img)
        lab = _maybe_squeeze_4d(lab)

        if img.ndim != 3 or lab.ndim != 3:
            raise RuntimeError(f"Expected 3D volumes. Got img={img.shape} lab={lab.shape} for uid={it.get('uid')}")

        # nib gives (X,Y,Z); convert to (D,H,W) = (Z,Y,X)
        img = np.transpose(img, (2, 1, 0))
        lab = np.transpose(lab, (2, 1, 0))

        img = _normalize_img(img)

        img = _pad_to_min_shape(img, self.patch_size, pad_value=0.0)
        lab = _pad_to_min_shape(lab, self.patch_size, pad_value=0)

        if self.training:
            want_fg = (rng.random() < self.pos_ratio)
            center = _random_center_from_label(lab, fg=want_fg, rng=rng)
        else:
            D, H, W = lab.shape
            center = (D // 2, H // 2, W // 2)

        img_p = _crop_around_center(img, center, self.patch_size)
        lab_p = _crop_around_center(lab, center, self.patch_size)

        if self.training:
            img_p, lab_p = _augment_patch(img_p, lab_p, rng=rng)

        img_t = torch.from_numpy(img_p[None, ...])  # (1,D,H,W)
        lab_t = torch.from_numpy(lab_p)             # (D,H,W)
        return {"image": img_t, "label": lab_t, "uid": it.get("uid", "")}


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, include_background=True):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        if (not self.include_background) and num_classes > 1:
            probs = probs[:, 1:]
            onehot = onehot[:, 1:]
        dims = tuple(range(2, probs.ndim))
        inter = torch.sum(probs * onehot, dim=dims)
        den = torch.sum(probs + onehot, dim=dims)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5, include_background=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
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


def _soft_erode(img):
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)

def _soft_dilate(img):
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)

def _soft_open(img):
    return _soft_dilate(_soft_erode(img))

def _soft_skel(img, iter_=10):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftclDiceLoss(nn.Module):
    def __init__(self, iterations=12, smooth=1e-5):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, probs_fg, targets_fg):
        skel_p = _soft_skel(probs_fg, self.iterations)
        skel_t = _soft_skel(targets_fg, self.iterations)
        tprec = (torch.sum(skel_p * targets_fg) + self.smooth) / (torch.sum(skel_p) + self.smooth)
        tsens = (torch.sum(skel_t * probs_fg) + self.smooth) / (torch.sum(skel_t) + self.smooth)
        cldice = (2 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cldice


class TubularSegLoss(nn.Module):
    def __init__(self, num_classes: int, w_tversky=0.6, w_dice=0.2, w_cldice=0.2, w_ce=0.0, alpha=0.3, beta=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.w_tv = float(w_tversky)
        self.w_dice = float(w_dice)
        self.w_cl = float(w_cldice)
        self.w_ce = float(w_ce)
        self.tv = TverskyLoss(alpha=alpha, beta=beta, include_background=True)
        self.dice = SoftDiceLoss(include_background=True)
        self.cldice = SoftclDiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        if logits.shape[2:] != targets.shape[-3:]:
            logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
        loss = 0.0
        if self.w_tv > 0:
            loss = loss + self.w_tv * self.tv(logits, targets)
        if self.w_dice > 0:
            loss = loss + self.w_dice * self.dice(logits, targets)
        if self.w_ce > 0:
            loss = loss + self.w_ce * self.ce(logits, targets.long())
        if self.w_cl > 0:
            probs = torch.softmax(logits, dim=1)
            probs_fg = 1.0 - probs[:, 0:1]
            tgt_fg = (targets > 0).float().unsqueeze(1)
            loss = loss + self.w_cl * self.cldice(probs_fg, tgt_fg)
        return loss


class MambaSSMBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.use_mamba = False
        try:
            from mamba_ssm import Mamba  # type: ignore
            self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_mamba = True
        except Exception:
            self.mixer = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=9, padding=4, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        seq = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        y = self.norm(seq)
        if self.use_mamba:
            y = self.mamba(y)
        else:
            y = self.mixer(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(y)
        out = (seq + y).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return out


def _make_base_grid_3d(B, D, H, W, device):
    zz = torch.linspace(-1, 1, D, device=device)
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)
    return grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

class SnakeRefine3D(nn.Module):
    def __init__(self, channels: int, K: int = 5, offset_scale: float = 0.25):
        super().__init__()
        assert K >= 3 and K % 2 == 1
        self.K = K
        self.half = K // 2
        self.offset_scale = offset_scale
        self.offset_pred = nn.Conv3d(channels, (K - 1), kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * K, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        delta = torch.tanh(self.offset_pred(x)) * self.offset_scale
        base = _make_base_grid_3d(B, D, H, W, x.device)
        grids = [base]

        cum = 0.0
        for s in range(1, self.half + 1):
            cum = cum + delta[:, s - 1:s, ...]
            g = base.clone()
            g[..., 2] = g[..., 2] + (cum.squeeze(1) * (2.0 / max(1, D - 1)))
            grids.append(g)

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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
        )
        self.block = ConvBlock(out_ch, out_ch)
    def forward(self, x):
        x = self.down(x)
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_snake: bool):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)
        self.snake = SnakeRefine3D(out_ch, K=5) if use_snake else nn.Identity()

    def forward(self, x, skip):
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

class MambaSnakeUNet3D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base=32, mamba_layers=4):
        super().__init__()
        self.stem = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)

        bott_dim = base * 8
        self.bottleneck = nn.Sequential(
            ConvBlock(bott_dim, bott_dim),
            *[MambaSSMBlock(bott_dim) for _ in range(mamba_layers)],
            ConvBlock(bott_dim, bott_dim),
        )

        self.u3 = Up(bott_dim, base * 8, base * 4, use_snake=False)
        self.u2 = Up(base * 4, base * 4, base * 2, use_snake=True)
        self.u1 = Up(base * 2, base * 2, base,     use_snake=True)
        self.u0 = Up(base,     base,     base,     use_snake=True)

        self.head = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)

        b = self.bottleneck(x3)

        y2 = self.u3(b, x3)
        y1 = self.u2(y2, x2)
        y0 = self.u1(y1, x1)
        y  = self.u0(y0, x0)

        return self.head(y)


@torch.no_grad()
def dice_per_class(logits, targets, num_classes: int, eps=1e-5):
    if logits.shape[2:] != targets.shape[-3:]:
        logits = F.interpolate(logits, size=targets.shape[-3:], mode="trilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        den = p.sum() + t.sum()
        dices.append(((2 * inter + eps) / (den + eps)).item())
    return dices


@dataclass
class TrainConfig:
    dataset_dir: Path
    out_dir: Path
    num_classes: int = 2
    epochs: int = 200
    batch_size: int = 1
    accum_steps: int = 1
    lr: float = 2e-4
    weight_decay: float = 1e-2
    val_ratio: float = 0.1
    num_workers: int = 4
    patch_size: Tuple[int, int, int] = (160, 160, 160)
    amp: bool = True
    seed: int = 42
    gpu: int = 0
    base_ch: int = 32
    mamba_layers: int = 4
    w_tversky: float = 0.6
    w_dice: float = 0.2
    w_cldice: float = 0.2
    w_ce: float = 0.0
    alpha: float = 0.3
    beta: float = 0.7


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, cfg: TrainConfig):
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        with autocast(enabled=cfg.amp):
            logits = model(img)
            loss = loss_fn(logits, lab) / max(1, cfg.accum_steps)

        if cfg.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step % cfg.accum_steps) == 0:
            if cfg.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.item()) * max(1, cfg.accum_steps)
        n += 1

    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, loss_fn, cfg: TrainConfig):
    model.eval()
    total = 0.0
    n = 0
    dices_sum = np.zeros((cfg.num_classes,), dtype=np.float64)

    for batch in loader:
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        with autocast(enabled=cfg.amp):
            logits = model(img)
            loss = loss_fn(logits, lab)

        total += float(loss.item())
        n += 1
        dices_sum += np.array(dice_per_class(logits, lab, cfg.num_classes), dtype=np.float64)

    return (total / max(1, n)), (dices_sum / max(1, n)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--mamba-layers", type=int, default=4)

    ap.add_argument("--w-tversky", type=float, default=0.6)
    ap.add_argument("--w-dice", type=float, default=0.2)
    ap.add_argument("--w-cldice", type=float, default=0.2)
    ap.add_argument("--w-ce", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)

    args = ap.parse_args()

    cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=max(1, args.accum_steps),
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        amp=(not args.no_amp),
        seed=args.seed,
        gpu=args.gpu,
        base_ch=args.base_ch,
        mamba_layers=args.mamba_layers,
        w_tversky=args.w_tversky,
        w_dice=args.w_dice,
        w_cldice=args.w_cldice,
        w_ce=args.w_ce,
        alpha=args.alpha,
        beta=args.beta,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # IMPORTANT: if CUDA_VISIBLE_DEVICES=0, then visible GPUs are re-indexed from 0.
    torch.cuda.set_device(cfg.gpu)
    device = torch.device(f"cuda:{cfg.gpu}")
    torch.backends.cudnn.benchmark = True

    items = list_cases(cfg.dataset_dir)
    train_items, val_items = split_train_val(items, val_ratio=cfg.val_ratio, seed=cfg.seed)

    train_ds = NiftiPatchDataset(train_items, patch_size=cfg.patch_size, training=True, seed=cfg.seed)
    val_ds   = NiftiPatchDataset(val_items,   patch_size=cfg.patch_size, training=False, seed=cfg.seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    model = MambaSnakeUNet3D(
        in_ch=1,
        num_classes=cfg.num_classes,
        base=cfg.base_ch,
        mamba_layers=cfg.mamba_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.amp)

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
    ).to(device)

    best_val = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * lr_factor(epoch - 1)

        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, cfg)
        val_loss, val_dice = validate(model, val_loader, loss_fn, cfg)
        dt = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt,
            "gpu": cfg.gpu,
            "patch_size": list(cfg.patch_size),
            "batch": cfg.batch_size,
            "accum_steps": cfg.accum_steps,
            "base_ch": cfg.base_ch,
            "mamba_layers": cfg.mamba_layers,
        }
        history.append(rec)
        print(json.dumps(rec))

        torch.save({"epoch": epoch, "model": model.state_dict()}, cfg.out_dir / "last.pt")
        (cfg.out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_val": best_val}, cfg.out_dir / "best.pt")

    print(f"Done. Best val_loss={best_val:.6f} -> {cfg.out_dir/'best.pt'}")


if __name__ == "__main__":
    main()

