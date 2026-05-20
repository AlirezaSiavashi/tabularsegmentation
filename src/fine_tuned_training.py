#!/usr/bin/env python3
# fine_tuned_training_v2.py
#
# Vessel segmentation training from cached .pt volumes (cases.json), with:
# - NO online augmentation (cache_augmented already contains augmentation)
# - JSON+text logging in out-dir: config.json, history.json, metrics.jsonl, train.log
# - best.pt + last.pt + periodic ckpt_epXXX.pt
# - tqdm progress in terminal
# - vessel-friendly losses: BCE + Tversky + soft-clDice + deep supervision
# - FG / hardneg / BG sampling with optional curriculum
# - Direction/flow head at bottleneck + guided warp offsets
# - Robust resume from previous checkpoints (PyTorch 2.6 safe load)
# - Validation improvements:
#     * optional threshold sweep (logs best threshold + dice@0.5)
#     * optional small-component removal (if scipy available)
#
# Example:
# python fine_tuned_training_v2.py \
#   --cache-dir /workspace/mamba_snake/data/cache_augmented \
#   --out-dir   /workspace/mamba_snake/logs/vessel_run_dir \
#   --resume    /workspace/mamba_snake/logs/vessel_run_dir/best.pt \
#   --device cuda:1 \
#   --patch-size 64 192 192 \
#   --batch-size 1 --accum 4 \
#   --amp bf16 \
#   --epochs 200 --iters-per-epoch 800 \
#   --val-every 5 \
#   --val-thresholds 0.25 0.30 0.35 0.40 0.45 0.50 0.55

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------
# Atomic IO + logging
# -----------------------------
def _to_jsonable(x):
    """Make numpy/torch/path/device/dtype JSON-safe."""
    try:
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    try:
        if isinstance(x, torch.device):
            return str(x)
        if isinstance(x, torch.dtype):
            return str(x)
        if torch.is_tensor(x):
            return {"tensor": True, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    except Exception:
        pass

    if isinstance(x, Path):
        return str(x)
    if isinstance(x, set):
        return list(x)
    if isinstance(x, tuple):
        return list(x)

    return str(x)


def save_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def torch_save_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class TeeLogger:
    """Write lines to stdout + train.log."""
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "train.log"

    def print(self, msg: str, flush: bool = True) -> None:
        print(msg, flush=flush)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_to_jsonable) + "\n")
        f.flush()


# -----------------------------
# Repro / utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_device_from_string(s: str, logger: Optional[TeeLogger] = None) -> torch.device:
    """
    Parse and validate device string like 'cuda', 'cuda:0', 'cuda:1', 'cpu'.
    If invalid CUDA ordinal, raise a helpful error.
    """
    dev = torch.device(s)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        n = torch.cuda.device_count()
        idx = dev.index if dev.index is not None else 0
        if idx is not None and (idx < 0 or idx >= n):
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            msg = (
                f"Invalid CUDA device ordinal: {s} (resolved index={idx}), but device_count={n}.\n"
                f"CUDA_VISIBLE_DEVICES={vis}\n"
                f"Fix: use --device cuda:0 (or another valid index), or set CUDA_VISIBLE_DEVICES to expose the GPU you want."
            )
            if logger:
                logger.print("[device] " + msg)
            raise RuntimeError(msg)
    return dev


def load_pt_tensor(path: Path) -> torch.Tensor:
    """
    Loads a .pt file that may be:
      - a tensor
      - a dict containing a tensor under common keys
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("image", "label", "vol", "mask", "data"):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        # fallback: first tensor value
        for v in obj.values():
            if torch.is_tensor(v):
                return v
        raise RuntimeError(f"Unsupported dict keys in {path}: {list(obj.keys())}")
    raise RuntimeError(f"Unsupported content type in {path}: {type(obj)}")


def read_cases_json(cache_dir: Path) -> Dict[str, Any]:
    p = Path(cache_dir) / "cases.json"
    if not p.exists():
        raise RuntimeError(f"cases.json not found: {p}")
    js = json.loads(p.read_text())
    for k in ("uids", "images", "labels"):
        if k not in js:
            raise RuntimeError(f"cases.json must contain key '{k}'")
    return js


def resolve_case_paths(cache_dir: Path, js: Dict[str, Any], uid: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    ip = Path(js["images"].get(uid, "")) if isinstance(js.get("images", {}), dict) else Path("")
    lp = Path(js["labels"].get(uid, "")) if isinstance(js.get("labels", {}), dict) else Path("")
    if not ip.exists():
        ip = cache_dir / "images" / f"{uid}.pt"
    if not lp.exists():
        lp = cache_dir / "labels" / f"{uid}.pt"
    if not ip.exists() or not lp.exists():
        raise RuntimeError(f"Missing image/label for uid={uid}: image={ip} label={lp}")
    return ip, lp


def standardize_pair(img: torch.Tensor, lab: torch.Tensor, clamp01: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    img: [C,D,H,W] or [D,H,W] or [1,D,H,W]
    lab: [D,H,W] or [1,D,H,W]
    returns:
      img: float32 [C,D,H,W]
      lab: uint8   [D,H,W] with {0,1}
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise RuntimeError(f"Expected img [C,D,H,W] or [D,H,W], got {tuple(img.shape)}")
    img = img.float()
    if clamp01:
        img = img.clamp(0.0, 1.0)

    if lab.ndim == 4 and lab.shape[0] == 1:
        lab = lab[0]
    if lab.ndim != 3:
        raise RuntimeError(f"Expected lab [D,H,W] or [1,D,H,W], got {tuple(lab.shape)}")
    lab = (lab > 0).to(torch.uint8)
    return img.contiguous(), lab.contiguous()


def pad_to_min_3d(vol: torch.Tensor, target_dhw: Tuple[int, int, int], pad_value: float = 0.0) -> torch.Tensor:
    """
    vol: [C,D,H,W] or [D,H,W]
    pads with constant to ensure >= target_dhw
    """
    if vol.ndim == 4:
        _, D, H, W = vol.shape
        has_c = True
    elif vol.ndim == 3:
        D, H, W = vol.shape
        has_c = False
    else:
        raise ValueError(vol.shape)

    td, th, tw = target_dhw
    pd = max(0, td - D)
    ph = max(0, th - H)
    pw = max(0, tw - W)
    if pd == 0 and ph == 0 and pw == 0:
        return vol

    pd0, pd1 = pd // 2, pd - (pd // 2)
    ph0, ph1 = ph // 2, ph - (ph // 2)
    pw0, pw1 = pw // 2, pw - (pw // 2)
    pad = (pw0, pw1, ph0, ph1, pd0, pd1)  # W,H,D

    if has_c:
        return F.pad(vol, pad, mode="constant", value=float(pad_value))
    else:
        return F.pad(vol.unsqueeze(0), pad, mode="constant", value=float(pad_value)).squeeze(0)


def crop_center_3d(vol: torch.Tensor, center_zyx: Tuple[int, int, int], patch_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """
    vol: [C,D,H,W] or [D,H,W]
    """
    if vol.ndim == 4:
        _, D, H, W = vol.shape
        has_c = True
    else:
        D, H, W = vol.shape
        has_c = False

    pd, ph, pw = patch_dhw
    cz, cy, cx = center_zyx

    sd = int(np.clip(cz - pd // 2, 0, max(0, D - pd)))
    sh = int(np.clip(cy - ph // 2, 0, max(0, H - ph)))
    sw = int(np.clip(cx - pw // 2, 0, max(0, W - pw)))

    if has_c:
        return vol[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
    else:
        return vol[sd:sd + pd, sh:sh + ph, sw:sw + pw]


# -----------------------------
# Hard-neg mining (near-vessel band)
# -----------------------------
@torch.no_grad()
def dilate_mask_3d(mask: torch.Tensor, iters: int = 1) -> torch.Tensor:
    x = (mask > 0).float()[None, None]  # [1,1,D,H,W]
    for _ in range(int(iters)):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return (x[0, 0] > 0.5)


@torch.no_grad()
def build_coords(lab: torch.Tensor, hard_dilate: int = 3) -> Dict[str, np.ndarray]:
    fg = (lab > 0)
    fg_idx = torch.nonzero(fg, as_tuple=False)

    if fg_idx.numel() == 0:
        return {"fg": np.zeros((0, 3), np.int32), "hardneg": np.zeros((0, 3), np.int32)}

    band = dilate_mask_3d(lab, iters=int(hard_dilate)) & (~fg)
    hn_idx = torch.nonzero(band, as_tuple=False)

    return {
        "fg": fg_idx.cpu().numpy().astype(np.int32, copy=False),
        "hardneg": hn_idx.cpu().numpy().astype(np.int32, copy=False),
    }


# -----------------------------
# Patch dataset from cache (NO online aug)
# -----------------------------
class CachePatchDataset(torch.utils.data.Dataset):
    """
    Produces random patches from cached full volumes.
    Uses FG + hardneg + bg sampling to fight extreme imbalance.
    """

    def __init__(
        self,
        cache_dir: Path,
        uids: List[str],
        patch_size: Tuple[int, int, int],
        iters_per_epoch: int,
        p_fg: float = 0.60,
        p_hardneg: float = 0.25,
        hardneg_dilate: int = 3,
        seed: int = 123,
        clamp01: bool = True,
        cache_items: int = 4,
    ):
        self.cache_dir = Path(cache_dir)
        self.js = read_cases_json(self.cache_dir)
        self.uids = list(uids)
        self.patch_size = tuple(map(int, patch_size))
        self.iters_per_epoch = int(iters_per_epoch)

        self.p_fg = float(p_fg)
        self.p_hardneg = float(p_hardneg)
        self.hardneg_dilate = int(hardneg_dilate)

        self.seed = int(seed)
        self.clamp01 = bool(clamp01)

        self._cache_items = int(max(1, cache_items))
        self._vol_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]] = {}
        self._lru: List[str] = []

    def __len__(self) -> int:
        return self.iters_per_epoch

    def set_sampling(self, p_fg: float, p_hardneg: float) -> None:
        self.p_fg = float(np.clip(p_fg, 0.05, 0.95))
        self.p_hardneg = float(np.clip(p_hardneg, 0.0, 0.90))
        if self.p_fg + self.p_hardneg > 0.98:
            self.p_hardneg = 0.98 - self.p_fg

    def _get_rng(self, idx: int) -> np.random.Generator:
        pid = os.getpid() % 100000
        return np.random.default_rng(self.seed + 10007 * idx + 97 * pid)

    def _cache_put(self, uid: str, img: torch.Tensor, lab: torch.Tensor, coords: Dict[str, np.ndarray]) -> None:
        if uid in self._vol_cache:
            return
        self._vol_cache[uid] = (img, lab, coords)
        self._lru.append(uid)
        if len(self._lru) > self._cache_items:
            old = self._lru.pop(0)
            self._vol_cache.pop(old, None)

    def _load_case(self, uid: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
        if uid in self._vol_cache:
            return self._vol_cache[uid]

        ip, lp = resolve_case_paths(self.cache_dir, self.js, uid)
        img = load_pt_tensor(ip)
        lab = load_pt_tensor(lp)
        img, lab = standardize_pair(img, lab, clamp01=self.clamp01)
        coords = build_coords(lab, hard_dilate=self.hardneg_dilate)
        self._cache_put(uid, img, lab, coords)
        return img, lab, coords

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._get_rng(idx)
        uid = self.uids[int(rng.integers(0, len(self.uids)))]
        img, lab, coords = self._load_case(uid)

        img = pad_to_min_3d(img, self.patch_size, pad_value=0.0)
        lab = pad_to_min_3d(lab, self.patch_size, pad_value=0)

        D, H, W = lab.shape
        kind = 0  # bg
        r = float(rng.random())

        if coords["fg"].shape[0] > 0 and r < self.p_fg:
            kind = 1
            z, y, x = coords["fg"][int(rng.integers(0, coords["fg"].shape[0]))]
            center = (int(z), int(y), int(x))
        elif coords["hardneg"].shape[0] > 0 and r < (self.p_fg + self.p_hardneg):
            kind = 2
            z, y, x = coords["hardneg"][int(rng.integers(0, coords["hardneg"].shape[0]))]
            center = (int(z), int(y), int(x))
        else:
            center = (int(rng.integers(0, D)), int(rng.integers(0, H)), int(rng.integers(0, W)))

        x = crop_center_3d(img, center, self.patch_size)  # [C,pd,ph,pw]
        y = crop_center_3d(lab, center, self.patch_size)  # [pd,ph,pw]

        return {"x": x.float(), "y": y.to(torch.long), "kind": torch.tensor(kind, dtype=torch.int64)}


# -----------------------------
# Soft skeleton + clDice (3D)
# -----------------------------
def _soft_erode3d(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate3d(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def _soft_open3d(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate3d(_soft_erode3d(x))


def soft_skel3d(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    x = x.clamp(0, 1)
    skel = F.relu(x - _soft_open3d(x))
    for _ in range(int(iters)):
        x = _soft_erode3d(x)
        opened = _soft_open3d(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def tversky_loss(prob: torch.Tensor, tgt: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    tp = (prob * tgt).sum(dim=(2, 3, 4))
    fp = (prob * (1 - tgt)).sum(dim=(2, 3, 4))
    fn = ((1 - prob) * tgt).sum(dim=(2, 3, 4))
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tv.mean()


def soft_cldice_loss(prob: torch.Tensor, tgt: torch.Tensor, skel_iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    s_pred = soft_skel3d(prob, iters=skel_iters)
    s_tgt = soft_skel3d(tgt, iters=skel_iters)
    tprec = (s_pred * tgt).sum(dim=(2, 3, 4)) / (s_pred.sum(dim=(2, 3, 4)) + eps)
    tsens = (s_tgt * prob).sum(dim=(2, 3, 4)) / (s_tgt.sum(dim=(2, 3, 4)) + eps)
    cl = (2 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1.0 - cl.mean()


# -----------------------------
# Direction target from GT mask (structure tensor, low-res)
# -----------------------------
@torch.no_grad()
def direction_target_from_mask_lowres(
    y: torch.Tensor,     # [B,1,D,H,W] float {0,1}
    out_dhw: Tuple[int, int, int],
    smooth_ks: int = 5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a low-res vessel tangent direction field from GT mask using a structure tensor.
    Returns:
      dir_tgt: [B,3,Dl,Hl,Wl] unit vectors (tangent direction, sign-invariant)
      conf   : [B,1,Dl,Hl,Wl] confidence (0..1)
      mask_w : [B,1,Dl,Hl,Wl] foreground weight mask (inside vessel)
    """

    Dl, Hl, Wl = out_dhw

    # Downsample mask to bottleneck res
    y_low = F.interpolate(y.float(), size=(Dl, Hl, Wl), mode="trilinear", align_corners=False).float()

    # SAFE smoothing kernel (never larger than smallest dimension)
    k = int(smooth_ks) if smooth_ks is not None else 1
    if k > 1:
        k = min(k, int(min(y_low.shape[-3], y_low.shape[-2], y_low.shape[-1])))
        if k % 2 == 0:
            k -= 1
    k = max(1, k)

    # Foreground mask at low-res
    fg = (y_low > 0.5).float()

    # Smooth (optional)
    if k > 1:
        pad = k // 2
        y_low = F.avg_pool3d(y_low, kernel_size=k, stride=1, padding=pad)

    # Gradients via conv3d (float32 kernels)
    device = y_low.device
    kz = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1) / 2.0
    ky = torch.tensor([[[[-1.0], [0.0], [1.0]]]], device=device, dtype=torch.float32).view(1, 1, 1, 3, 1) / 2.0
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=torch.float32).view(1, 1, 1, 1, 3) / 2.0

    gz = F.conv3d(y_low, kz, padding=(1, 0, 0))
    gy = F.conv3d(y_low, ky, padding=(0, 1, 0))
    gx = F.conv3d(y_low, kx, padding=(0, 0, 1))

    # Structure tensor components
    Jxx = gx * gx
    Jyy = gy * gy
    Jzz = gz * gz
    Jxy = gx * gy
    Jxz = gx * gz
    Jyz = gy * gz

    # Local averaging
    if k > 1:
        pad = k // 2
        pool = lambda t: F.avg_pool3d(t, kernel_size=k, stride=1, padding=pad)
        Jxx, Jyy, Jzz = pool(Jxx), pool(Jyy), pool(Jzz)
        Jxy, Jxz, Jyz = pool(Jxy), pool(Jxz), pool(Jyz)

    # Tensor matrix per voxel, then eigendecomposition
    J = torch.stack(
        [
            torch.stack([Jxx, Jxy, Jxz], dim=-1),
            torch.stack([Jxy, Jyy, Jyz], dim=-1),
            torch.stack([Jxz, Jyz, Jzz], dim=-1),
        ],
        dim=-2,
    )  # [B,1,Dl,Hl,Wl,3,3]
    J = J.squeeze(1).float()  # [B,Dl,Hl,Wl,3,3]

    evals, evecs = torch.linalg.eigh(J)  # (...,3), (...,3,3)
    v = evecs[..., :, 0]  # smallest-eigenvalue vector = tangent direction
    v = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

    # Confidence: anisotropy
    e0 = evals[..., 0].clamp_min(0.0)
    e2 = evals[..., 2].clamp_min(eps)
    conf = ((e2 - e0) / e2).clamp(0.0, 1.0)  # [B,Dl,Hl,Wl]

    dir_tgt = v.permute(0, 4, 1, 2, 3).contiguous()          # [B,3,Dl,Hl,Wl]
    conf = conf.unsqueeze(1).contiguous()                      # [B,1,Dl,Hl,Wl]
    mask_w = fg.contiguous()                                   # [B,1,Dl,Hl,Wl]
    return dir_tgt, conf, mask_w


def cosine_loss_sign_invariant(pred: torch.Tensor, tgt: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    pred,tgt: [B,3,D,H,W] unit-ish vectors
    sign-invariant cosine: loss = 1 - |cos|
    """
    pred = pred / (pred.norm(dim=1, keepdim=True).clamp_min(eps))
    tgt = tgt / (tgt.norm(dim=1, keepdim=True).clamp_min(eps))
    cos = (pred * tgt).sum(dim=1, keepdim=True)  # [B,1,D,H,W]
    loss = 1.0 - cos.abs()
    if w is not None:
        loss = loss * w
        return loss.sum() / (w.sum().clamp_min(eps))
    return loss.mean()


# -----------------------------
# Model blocks (vessel-focused + guided warp)
# -----------------------------
def gn(ch: int, groups: int = 16) -> nn.GroupNorm:
    g = min(groups, ch)
    while g > 1 and (ch % g != 0):
        g -= 1
    return nn.GroupNorm(g, ch)


class CoordInject(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x.shape
        device = x.device
        zz = torch.linspace(-1, 1, D, device=device, dtype=x.dtype)
        yy = torch.linspace(-1, 1, H, device=device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, W, device=device, dtype=x.dtype)
        z, y, xx_ = torch.meshgrid(zz, yy, xx, indexing="ij")
        coords = torch.stack([z, y, xx_], dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B,3,D,H,W]
        return torch.cat([x, coords], dim=1)


class AxialSnakeRefine3D(nn.Module):
    def __init__(self, ch: int, k: int = 7):
        super().__init__()
        k = int(k)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        self.dwz = nn.Conv3d(ch, ch, kernel_size=(k, 1, 1), padding=(pad, 0, 0), groups=ch, bias=False)
        self.dwy = nn.Conv3d(ch, ch, kernel_size=(1, k, 1), padding=(0, pad, 0), groups=ch, bias=False)
        self.dwx = nn.Conv3d(ch, ch, kernel_size=(1, 1, k), padding=(0, 0, pad), groups=ch, bias=False)
        self.pw = nn.Conv3d(ch, ch, 1, bias=False)
        self.n = gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwz(x) + self.dwy(x) + self.dwx(x)
        y = self.pw(y)
        y = F.silu(self.n(y))
        return x + y


class WarpConv3D(nn.Module):
    """
    Deformable-like warp: predicts (dz,dy,dx) then warps via grid_sample.
    Guided by direction field with learnable scale gamma.
    """
    def __init__(self, ch: int, max_disp_vox: float = 1.5, use_guidance: bool = True):
        super().__init__()
        self.max_disp = float(max_disp_vox)
        self.use_guidance = bool(use_guidance)

        self.off = nn.Conv3d(ch, 3, 3, padding=1)
        self.conv = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.n = gn(ch)

        # Start at 0 -> guidance initially off, learns to turn on if helpful
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, dir_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, D, H, W = x.shape
        device = x.device
        dtype = x.dtype

        disp = torch.tanh(self.off(x)) * self.max_disp  # [B,3,D,H,W] = [dz,dy,dx]

        if self.use_guidance and (dir_field is not None):
            # tanh keeps it bounded and stable
            g = torch.tanh(self.gamma)
            disp = disp + g * dir_field

        zz = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        z, y, xx_ = torch.meshgrid(zz, yy, xx, indexing="ij")
        base = torch.stack([xx_, y, z], dim=-1)[None].repeat(B, 1, 1, 1, 1)  # [B,D,H,W,3]

        dz = disp[:, 0]
        dy = disp[:, 1]
        dx = disp[:, 2]

        dxn = dx * (2.0 / max(1, W - 1))
        dyn = dy * (2.0 / max(1, H - 1))
        dzn = dz * (2.0 / max(1, D - 1))

        grid = base.clone()
        grid[..., 0] = grid[..., 0] + dxn
        grid[..., 1] = grid[..., 1] + dyn
        grid[..., 2] = grid[..., 2] + dzn
        grid = grid.clamp(-1.2, 1.2)

        xw = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
        y = self.conv(xw)
        y = F.silu(self.n(y))
        return x + y


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_warp: bool = False, warp_guided: bool = True):
        super().__init__()
        self.use_proj = (in_ch != out_ch)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if self.use_proj else nn.Identity()
        self.c1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = gn(out_ch)
        self.c2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = gn(out_ch)
        self.warp = WarpConv3D(out_ch, use_guidance=warp_guided) if use_warp else None

    def forward(self, x: torch.Tensor, dir_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = self.proj(x)
        x = F.silu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        if self.warp is not None:
            x = self.warp(F.silu(x), dir_field=dir_field)
        x = F.silu(x + r)
        return x


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_warp: bool = False, warp_guided: bool = True):
        super().__init__()
        self.d = nn.Conv3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.n = gn(out_ch)
        self.b = ResBlock3D(out_ch, out_ch, use_warp=use_warp, warp_guided=warp_guided)

    def forward(self, x: torch.Tensor, dir_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.silu(self.n(self.d(x)))
        return self.b(x, dir_field=dir_field)


class Up3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_warp: bool = False, warp_guided: bool = True):
        super().__init__()
        self.u = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.b = ResBlock3D(out_ch + skip_ch, out_ch, use_warp=use_warp, warp_guided=warp_guided)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, dir_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.u(x)
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        return self.b(x, dir_field=dir_field)


class BiMambaBottleneck(nn.Module):
    """
    Bidirectional Mamba over flattened 3D tokens at bottleneck resolution.
    Fallback to conv if mamba_ssm isn't installed.
    """
    def __init__(self, ch: int, depth: int = 2):
        super().__init__()
        self.depth = int(depth)
        self.use_mamba = False
        try:
            from mamba_ssm import Mamba  # type: ignore
            self.use_mamba = True
            self.norms = nn.ModuleList([nn.LayerNorm(ch) for _ in range(self.depth)])
            self.mf = nn.ModuleList([Mamba(d_model=ch, d_state=16, d_conv=4, expand=2) for _ in range(self.depth)])
            self.mb = nn.ModuleList([Mamba(d_model=ch, d_state=16, d_conv=4, expand=2) for _ in range(self.depth)])
        except Exception:
            self.blocks = nn.ModuleList([ResBlock3D(ch, ch, use_warp=False) for _ in range(self.depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_mamba:
            for b in self.blocks:
                x = b(x)
            return x

        B, C, D, H, W = x.shape
        L = D * H * W
        xt = x.view(B, C, L).transpose(1, 2).contiguous()  # [B,L,C]
        for i in range(self.depth):
            y = self.norms[i](xt)
            yf = self.mf[i](y)
            yb = self.mb[i](torch.flip(y, dims=(1,)))
            yb = torch.flip(yb, dims=(1,))
            xt = xt + 0.5 * (yf + yb)
        x = xt.transpose(1, 2).contiguous().view(B, C, D, H, W)
        return x


class MambaSnakeUNet3D(nn.Module):
    """
    UNet with direction head at bottleneck + guided warp.
    """
    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        deep_supervision: bool = True,
        coord_inject: bool = True,
        warp_levels: Tuple[bool, bool, bool, bool] = (False, True, True, False),
        warp_guided: bool = True,
        mamba_depth: int = 2,
        snake_k: int = 7,
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.coord_inject = bool(coord_inject)
        self.coord = CoordInject() if self.coord_inject else None
        self.warp_guided = bool(warp_guided)

        stem_in = in_ch + (3 if self.coord_inject else 0)

        self.stem = ResBlock3D(stem_in, base, use_warp=warp_levels[0], warp_guided=self.warp_guided)
        self.d1 = Down3D(base, base * 2, use_warp=warp_levels[0], warp_guided=self.warp_guided)
        self.d2 = Down3D(base * 2, base * 4, use_warp=warp_levels[1], warp_guided=self.warp_guided)
        self.d3 = Down3D(base * 4, base * 8, use_warp=warp_levels[2], warp_guided=self.warp_guided)
        self.d4 = Down3D(base * 8, base * 16, use_warp=warp_levels[3], warp_guided=self.warp_guided)

        self.bot = BiMambaBottleneck(base * 16, depth=mamba_depth)

        # Direction head at bottleneck
        self.dir_head = nn.Conv3d(base * 16, 3, 1)

        self.u4 = Up3D(base * 16, base * 8, base * 8, use_warp=warp_levels[2], warp_guided=self.warp_guided)
        self.u3 = Up3D(base * 8, base * 4, base * 4, use_warp=warp_levels[1], warp_guided=self.warp_guided)
        self.u2 = Up3D(base * 4, base * 2, base * 2, use_warp=warp_levels[1], warp_guided=self.warp_guided)
        self.u1 = Up3D(base * 2, base, base, use_warp=warp_levels[0], warp_guided=self.warp_guided)

        self.sn3 = AxialSnakeRefine3D(base * 4, k=snake_k)
        self.sn2 = AxialSnakeRefine3D(base * 2, k=snake_k)

        self.head = nn.Conv3d(base, 1, 1)

        if self.deep_supervision:
            self.ds3 = nn.Conv3d(base * 4, 1, 1)
            self.ds2 = nn.Conv3d(base * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        if self.coord is not None:
            x = self.coord(x)

        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        b = self.bot(x4)
        dir_pred = self.dir_head(b)

        def up_dir(to_like: torch.Tensor) -> Optional[torch.Tensor]:
            if not self.warp_guided:
                return None
            tgt_sp = to_like.shape[-3:]
            d = F.interpolate(dir_pred, size=tgt_sp, mode="trilinear", align_corners=False)
            d = d / (d.norm(dim=1, keepdim=True).clamp_min(1e-6))
            return d

        y4 = self.u4(b, x3, dir_field=up_dir(x3))
        y3 = self.u3(y4, x2, dir_field=up_dir(x2))
        y3 = self.sn3(y3)
        y2 = self.u2(y3, x1, dir_field=up_dir(x1))
        y2 = self.sn2(y2)
        y1 = self.u1(y2, x0, dir_field=up_dir(x0))

        out = self.head(y1)

        aux: List[torch.Tensor] = []
        if self.deep_supervision:
            aux.append(self.ds3(y3))
            aux.append(self.ds2(y2))

        return out, aux, dir_pred


# -----------------------------
# Losses
# -----------------------------
def seg_loss_fn(
    logits: torch.Tensor,
    aux: List[torch.Tensor],
    y: torch.Tensor,  # [B,1,D,H,W] float {0,1}
    bce_pos_weight: float,
    w_bce: float,
    w_tversky: float,
    w_cldice: float,
    tversky_alpha: float,
    tversky_beta: float,
    cldice_iters: int,
    deep_supervision: bool,
) -> torch.Tensor:
    pos_w = torch.tensor([bce_pos_weight], device=logits.device, dtype=torch.float32)

    bce = F.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=pos_w)
    prob = torch.sigmoid(logits.float())
    lt = tversky_loss(prob, y, alpha=tversky_alpha, beta=tversky_beta)
    lc = soft_cldice_loss(prob, y, skel_iters=cldice_iters)

    main = w_bce * bce + w_tversky * lt + w_cldice * lc

    if deep_supervision and len(aux) > 0:
        # IMPORTANT FIX: downsample GT to the *actual aux size*, not assumed factors
        ds_loss = 0.0
        weights = [0.30, 0.50]  # ds3 lower-res, ds2 mid-res
        for i, a in enumerate(aux):
            w = weights[i] if i < len(weights) else 0.3
            ya = F.interpolate(y.float(), size=a.shape[-3:], mode="nearest")
            bce_a = F.binary_cross_entropy_with_logits(a.float(), ya.float(), pos_weight=pos_w)
            pa = torch.sigmoid(a.float())
            lt_a = tversky_loss(pa, ya, alpha=tversky_alpha, beta=tversky_beta)
            lc_a = soft_cldice_loss(pa, ya, skel_iters=max(5, cldice_iters // 2))
            ds_loss = ds_loss + w * (w_bce * bce_a + w_tversky * lt_a + w_cldice * lc_a)
        return main + ds_loss

    return main


# -----------------------------
# Sliding-window inference (validation)
# -----------------------------
@torch.no_grad()
def gaussian_weight_3d(patch_dhw: Tuple[int, int, int], sigma_scale: float = 0.125, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    pd, ph, pw = patch_dhw
    zz = torch.linspace(-1, 1, pd, device=device)
    yy = torch.linspace(-1, 1, ph, device=device)
    xx = torch.linspace(-1, 1, pw, device=device)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    dist2 = z * z + y * y + x * x
    sigma2 = (sigma_scale ** 2)
    w = torch.exp(-0.5 * dist2 / max(1e-6, sigma2))
    w = w / w.max().clamp_min(1e-6)
    return w  # [pd,ph,pw]


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module,
    vol: torch.Tensor,                 # [1,C,D,H,W]
    patch_dhw: Tuple[int, int, int],
    overlap: float = 0.5,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    device = vol.device
    _, _, D, H, W = vol.shape
    pd, ph, pw = patch_dhw

    v = vol[0]
    v = pad_to_min_3d(v, patch_dhw, pad_value=0.0)
    D2, H2, W2 = v.shape[1], v.shape[2], v.shape[3]
    v = v.unsqueeze(0)

    sd = max(1, int(pd * (1.0 - overlap)))
    sh = max(1, int(ph * (1.0 - overlap)))
    sw = max(1, int(pw * (1.0 - overlap)))

    w_patch = gaussian_weight_3d(patch_dhw, device=device)[None, None]
    out = torch.zeros((1, 1, D2, H2, W2), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, D2, H2, W2), device=device, dtype=torch.float32)

    dev_type = device.type
    for z0 in range(0, max(1, D2 - pd + 1), sd):
        for y0 in range(0, max(1, H2 - ph + 1), sh):
            for x0 in range(0, max(1, W2 - pw + 1), sw):
                z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw
                patch = v[:, :, z0:z1, y0:y1, x0:x1]
                with torch.autocast(device_type=dev_type, enabled=(use_amp and dev_type == "cuda"), dtype=amp_dtype or torch.bfloat16):
                    logits, _, _ = model(patch)
                logits = logits.float()
                out[:, :, z0:z1, y0:y1, x0:x1] += logits * w_patch
                wsum[:, :, z0:z1, y0:y1, x0:x1] += w_patch

    out = out / wsum.clamp_min(1e-6)
    return out[:, :, :D, :H, :W]


# -----------------------------
# Metrics + optional postproc
# -----------------------------
@torch.no_grad()
def dice_score(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> float:
    p = (pred > 0).float()
    t = (tgt > 0).float()
    inter = (p * t).sum()
    den = p.sum() + t.sum()
    return float(((2 * inter + eps) / (den + eps)).item())


@torch.no_grad()
def cldice_score_hard(pred: torch.Tensor, tgt: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> float:
    p = (pred > 0).float()[None, None]
    t = (tgt > 0).float()[None, None]
    sp = soft_skel3d(p, iters=iters)
    st = soft_skel3d(t, iters=iters)
    tprec = (sp * t).sum() / (sp.sum() + eps)
    tsens = (st * p).sum() / (st.sum() + eps)
    cl = (2 * tprec * tsens + eps) / (tprec + tsens + eps)
    return float(cl.item())


def remove_small_components_np(mask_u8: np.ndarray, min_vox: int) -> np.ndarray:
    """
    Remove connected components smaller than min_vox.
    Uses scipy if available; otherwise returns original mask.
    """
    if min_vox <= 0:
        return mask_u8
    try:
        import scipy.ndimage as ndi  # type: ignore
        lab, n = ndi.label(mask_u8 > 0)
        if n <= 0:
            return mask_u8
        counts = np.bincount(lab.reshape(-1))
        keep = counts >= int(min_vox)
        keep[0] = False
        out = keep[lab]
        return (out.astype(np.uint8) * 1)
    except Exception:
        return mask_u8


@torch.no_grad()
def validate_fullvol(
    model: nn.Module,
    cache_dir: Path,
    js: Dict[str, Any],
    val_uids: List[str],
    patch_dhw: Tuple[int, int, int],
    device: torch.device,
    overlap: float,
    max_cases: int,
    clamp01: bool,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    thresholds: List[float],
    pp_min_vox: int,
    logger: TeeLogger,
) -> Dict[str, float]:
    model.eval()
    dices_05: List[float] = []
    dices_best: List[float] = []
    clds_best: List[float] = []
    best_thrs: List[float] = []

    for uid in val_uids[: max_cases]:
        ip, lp = resolve_case_paths(cache_dir, js, uid)
        img = load_pt_tensor(ip)
        lab = load_pt_tensor(lp)
        img, lab = standardize_pair(img, lab, clamp01=clamp01)

        vol = img.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
        logits = sliding_window_logits(
            model, vol, patch_dhw=patch_dhw, overlap=overlap,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        prob = torch.sigmoid(logits)[0, 0].detach().float().cpu()  # [D,H,W] on CPU
        lab_u8 = lab.cpu()

        # dice@0.5 (for continuity with older logs)
        pred05 = (prob > 0.5).to(torch.uint8).numpy()
        if pp_min_vox > 0:
            pred05 = remove_small_components_np(pred05, pp_min_vox)
        dices_05.append(dice_score(torch.from_numpy(pred05), lab_u8))

        # best threshold sweep
        best_d = -1.0
        best_t = float(thresholds[0]) if thresholds else 0.5
        best_pred = None

        if not thresholds:
            thresholds = [0.5]

        for t in thresholds:
            p = (prob > float(t)).to(torch.uint8).numpy()
            if pp_min_vox > 0:
                p = remove_small_components_np(p, pp_min_vox)
            d = dice_score(torch.from_numpy(p), lab_u8)
            if d > best_d:
                best_d = d
                best_t = float(t)
                best_pred = p

        if best_pred is None:
            best_pred = (prob > 0.5).to(torch.uint8).numpy()

        dices_best.append(float(best_d))
        best_thrs.append(float(best_t))
        clds_best.append(cldice_score_hard(torch.from_numpy(best_pred), lab_u8, iters=10))

    out = {
        "dice@0.5": float(np.mean(dices_05)) if dices_05 else 0.0,
        "dice_best": float(np.mean(dices_best)) if dices_best else 0.0,
        "cldice_best": float(np.mean(clds_best)) if clds_best else 0.0,
        "thr_best_mean": float(np.mean(best_thrs)) if best_thrs else 0.5,
    }
    logger.print(
        f"[val] dice@0.5={out['dice@0.5']:.6f}  dice_best={out['dice_best']:.6f}  "
        f"cldice_best={out['cldice_best']:.6f}  thr_best_mean={out['thr_best_mean']:.3f}"
    )
    return out


# -----------------------------
# Train helpers
# -----------------------------
def make_splits(uids: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    u = list(uids)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    n_val = max(1, int(round(len(u) * float(val_frac))))
    val = u[:n_val]
    tr = u[n_val:] or u[:-1]
    if len(tr) == 0:
        tr, val = u[:-1], u[-1:]
    return tr, val


def get_amp_dtype(name: str) -> Optional[torch.dtype]:
    name = name.lower()
    if name == "off":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("--amp must be off|bf16|fp16")


def lr_cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, (total - warmup))
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def is_bf16_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return True


def safe_torch_load_checkpoint(path: Path, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Train vessel segmentation from cache (cases.json + pt volumes)")

    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iters-per-epoch", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--patch-size", type=int, nargs=3, default=[64, 192, 192])

    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-overlap", type=float, default=0.5)
    ap.add_argument("--val-max-cases", type=int, default=25)

    # validation threshold sweep + optional postproc
    ap.add_argument("--val-thresholds", type=float, nargs="*", default=[0.5])
    ap.add_argument("--pp-min-vox", type=int, default=0, help="remove small components < N voxels in validation (requires scipy). 0=off")

    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--cache-items", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"])
    ap.add_argument("--clip-grad", type=float, default=1.0)

    # sampling
    ap.add_argument("--p-fg", type=float, default=0.60)
    ap.add_argument("--p-hardneg", type=float, default=0.25)
    ap.add_argument("--hardneg-dilate", type=int, default=3)
    ap.add_argument("--curriculum", action="store_true")

    # data assumptions
    ap.add_argument("--no-clamp01", action="store_true")

    # model
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--no-deep-supervision", action="store_true")
    ap.add_argument("--no-coord", action="store_true")
    ap.add_argument("--mamba-depth", type=int, default=2)
    ap.add_argument("--snake-k", type=int, default=7)

    # warp guidance
    ap.add_argument("--no-warp-guidance", action="store_true")

    # direction supervision
    ap.add_argument("--w-dir", type=float, default=0.15)
    ap.add_argument("--dir-smooth-ks", type=int, default=5)
    ap.add_argument("--dir-conf-power", type=float, default=1.0)
    ap.add_argument("--dir-min-fg", type=float, default=0.02)
    ap.add_argument("--dir-warmup-epochs", type=int, default=20, help="ramp w-dir from 0 to w-dir over N epochs")

    # losses
    ap.add_argument("--bce-pos-weight", type=float, default=12.0)
    ap.add_argument("--w-bce", type=float, default=0.20)
    ap.add_argument("--w-tversky", type=float, default=0.55)
    ap.add_argument("--w-cldice", type=float, default=0.25)
    ap.add_argument("--tversky-alpha", type=float, default=0.30)
    ap.add_argument("--tversky-beta", type=float, default=0.70)
    ap.add_argument("--cldice-iters", type=int, default=10)

    # saving
    ap.add_argument("--save-every", type=int, default=5)

    # resume
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--reset-optimizer", action="store_true", help="ignore optimizer state in checkpoint even if present")

    # optional compile (PyTorch 2.x)
    ap.add_argument("--compile", action="store_true")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(args.out_dir)

    seed_everything(args.seed)
    device = safe_device_from_string(args.device, logger=logger)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    amp_dtype = get_amp_dtype(args.amp)
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not is_bf16_ok():
        logger.print("[warn] BF16 not supported on this GPU; switching AMP to fp16")
        amp_dtype = torch.float16

    clamp01 = not args.no_clamp01

    cache_dir = Path(args.cache_dir)
    js = read_cases_json(cache_dir)
    all_uids = list(js["uids"])
    tr_uids, val_uids = make_splits(all_uids, args.val_frac, args.seed)

    # probe channels
    ip0, lp0 = resolve_case_paths(cache_dir, js, tr_uids[0])
    img0 = load_pt_tensor(ip0)
    lab0 = load_pt_tensor(lp0)
    img0, lab0 = standardize_pair(img0, lab0, clamp01=clamp01)
    in_ch = int(img0.shape[0])

    patch_dhw = tuple(map(int, args.patch_size))

    ds = CachePatchDataset(
        cache_dir=cache_dir,
        uids=tr_uids,
        patch_size=patch_dhw,
        iters_per_epoch=int(args.iters_per_epoch),
        p_fg=float(args.p_fg),
        p_hardneg=float(args.p_hardneg),
        hardneg_dilate=int(args.hardneg_dilate),
        seed=int(args.seed),
        clamp01=clamp01,
        cache_items=int(args.cache_items),
    )

    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch)

    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    model = MambaSnakeUNet3D(
        in_ch=in_ch,
        base=int(args.base),
        deep_supervision=(not args.no_deep_supervision),
        coord_inject=(not args.no_coord),
        mamba_depth=int(args.mamba_depth),
        snake_k=int(args.snake_k),
        warp_guided=(not args.no_warp_guidance),
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            logger.print("[compile] torch.compile enabled")
        except Exception as e:
            logger.print(f"[compile] failed, continuing without compile: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    total_steps = int(args.epochs) * int(args.iters_per_epoch)
    warmup_steps = max(200, int(0.03 * total_steps))

    scaler = None
    if device.type == "cuda" and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    config_path = args.out_dir / "config.json"
    history_path = args.out_dir / "history.json"
    metrics_path = args.out_dir / "metrics.jsonl"

    # Safer config dict (prevents any circular references)
    args_dict = {k: _to_jsonable(v) for k, v in vars(args).items()}
    save_json_atomic(config_path, args_dict)

    # history init
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            hist_obj = json.loads(history_path.read_text())
            if isinstance(hist_obj, list):
                history = hist_obj
        except Exception:
            history = []

    # resume state
    start_epoch = 1
    global_step = 0
    best_val = -1.0

    if args.resume is not None and Path(args.resume).exists():
        ckpt = safe_torch_load_checkpoint(args.resume, map_location="cpu")

        missing, unexpected = model.load_state_dict(ckpt.get("model", {}), strict=False)
        logger.print("[resume] model loaded with strict=False")
        logger.print(f"[resume] missing keys: {len(missing)} (up to 20) {missing[:20]}")
        logger.print(f"[resume] unexpected keys: {len(unexpected)} (up to 20) {unexpected[:20]}")

        if (not args.reset_optimizer) and ("opt" in ckpt):
            try:
                opt.load_state_dict(ckpt["opt"])
                logger.print("[resume] optimizer state loaded")
            except Exception as e:
                logger.print(f"[resume] optimizer state NOT loaded (will re-init optimizer): {e}")

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", -1.0))
        logger.print(f"[resume] epoch={start_epoch} global_step={global_step} best_val={best_val:.6f}")

    logger.print(f"[data] cases={len(all_uids)} train={len(tr_uids)} val={len(val_uids)}")
    logger.print(f"[data] in_ch={in_ch} patch={patch_dhw} iters/epoch={args.iters_per_epoch}")
    logger.print(f"[amp] {args.amp} (effective dtype={amp_dtype})")
    logger.print("[aug] online augmentation is DISABLED (cache is already augmented).")
    logger.print(f"[dir] enabled, warp_guided={not args.no_warp_guidance}, w_dir={args.w_dir}, warmup_epochs={args.dir_warmup_epochs}")
    logger.print(f"[val] thresholds={args.val_thresholds}  pp_min_vox={args.pp_min_vox}")

    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, int(args.epochs) + 1):
        # optional curriculum
        if args.curriculum:
            t = (epoch - 1) / max(1, (int(args.epochs) - 1))
            pfg = float(np.interp(t, [0.0, 1.0], [0.75, float(args.p_fg)]))
            phn = float(np.interp(t, [0.0, 1.0], [0.15, float(args.p_hardneg)]))
            ds.set_sampling(pfg, phn)

        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_seg = 0.0
        run_dir = 0.0
        run_kind = np.zeros((3,), dtype=np.int64)

        pbar = tqdm(loader, total=int(args.iters_per_epoch), ncols=120, desc=f"epoch {epoch:03d}", leave=False)
        opt.zero_grad(set_to_none=True)

        # warmup direction weight
        dir_w = float(args.w_dir)
        if int(args.dir_warmup_epochs) > 0:
            dir_w = dir_w * min(1.0, max(0.0, (epoch - 1) / float(args.dir_warmup_epochs)))

        for it, batch in enumerate(pbar, start=1):
            lr = lr_cosine_with_warmup(global_step, total_steps, warmup_steps, float(args.lr), float(args.min_lr))
            for pg in opt.param_groups:
                pg["lr"] = lr

            x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)    # [B,C,D,H,W]
            y = batch["y"].to(device=device, dtype=torch.float32, non_blocking=True)    # [B,D,H,W]
            kind = batch["kind"].cpu().numpy()
            for k in kind:
                run_kind[int(k)] += 1

            y = y.unsqueeze(1)  # [B,1,D,H,W] in {0,1}

            use_amp = (device.type == "cuda" and amp_dtype is not None)

            # forward + seg loss under autocast
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype or torch.bfloat16):
                logits, aux, dir_pred = model(x)
                l_seg = seg_loss_fn(
                    logits=logits,
                    aux=aux,
                    y=y,
                    bce_pos_weight=float(args.bce_pos_weight),
                    w_bce=float(args.w_bce),
                    w_tversky=float(args.w_tversky),
                    w_cldice=float(args.w_cldice),
                    tversky_alpha=float(args.tversky_alpha),
                    tversky_beta=float(args.tversky_beta),
                    cldice_iters=int(args.cldice_iters),
                    deep_supervision=(not args.no_deep_supervision),
                )

            # direction supervision in float32 (autocast OFF for numerical stability)
            with torch.autocast(device_type=device.type, enabled=False):
                Db, Hb, Wb = dir_pred.shape[-3:]
                # compute a cheap FG density check at bottleneck size
                y_lr = F.interpolate(y.float(), size=(Db, Hb, Wb), mode="trilinear", align_corners=False)
                fg_density = float((y_lr > 0.5).float().mean().item())

                l_dir = torch.tensor(0.0, device=device, dtype=torch.float32)
                if fg_density >= float(args.dir_min_fg):
                    dir_tgt, conf, fg_w = direction_target_from_mask_lowres(
                        y=y.float(), out_dhw=(Db, Hb, Wb), smooth_ks=int(args.dir_smooth_ks)
                    )
                    # weight: inside vessel * confidence^power
                    w = (fg_w * (conf.clamp(0, 1) ** float(args.dir_conf_power))).float()

                    dp = dir_pred.float()
                    dp = dp / (dp.norm(dim=1, keepdim=True).clamp_min(1e-6))
                    l_dir = cosine_loss_sign_invariant(dp, dir_tgt.float(), w=w)

                loss = l_seg.float() + dir_w * l_dir
                loss = loss / max(1, int(args.accum))

            # backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it % int(args.accum)) == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                if float(args.clip_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            loss_val = float(loss.item()) * max(1, int(args.accum))
            run_loss += loss_val
            run_seg += float(l_seg.detach().float().item())
            run_dir += float(l_dir.detach().float().item())
            global_step += 1

            pbar.set_postfix(
                loss=f"{run_loss / max(1, it):.4f}",
                seg=f"{run_seg / max(1, it):.4f}",
                dir=f"{run_dir / max(1, it):.4f}",
                lr=f"{lr:.2e}",
                fg=int(run_kind[1]),
                hn=int(run_kind[2]),
                bg=int(run_kind[0]),
            )

            if it >= int(args.iters_per_epoch):
                break

        dt = time.time() - t0
        train_rec = {
            "epoch": epoch,
            "loss": run_loss / max(1, int(args.iters_per_epoch)),
            "seg_loss": run_seg / max(1, int(args.iters_per_epoch)),
            "dir_loss": run_dir / max(1, int(args.iters_per_epoch)),
            "lr": float(opt.param_groups[0]["lr"]),
            "sec": dt,
            "bg": int(run_kind[0]),
            "fg": int(run_kind[1]),
            "hardneg": int(run_kind[2]),
            "p_fg_eff": float(ds.p_fg),
            "p_hardneg_eff": float(ds.p_hardneg),
            "warp_guided": (not args.no_warp_guidance),
            "dir_w_eff": float(dir_w),
        }
        logger.print(json.dumps(train_rec))
        append_jsonl(metrics_path, {"type": "train", **train_rec})

        # validation
        val_rec: Optional[Dict[str, Any]] = None
        if (epoch % int(args.val_every) == 0) or (epoch == int(args.epochs)):
            val = validate_fullvol(
                model=model,
                cache_dir=cache_dir,
                js=js,
                val_uids=val_uids,
                patch_dhw=patch_dhw,
                device=device,
                overlap=float(args.val_overlap),
                max_cases=int(args.val_max_cases),
                clamp01=clamp01,
                use_amp=(device.type == "cuda" and amp_dtype is not None),
                amp_dtype=amp_dtype,
                thresholds=list(map(float, args.val_thresholds)) if args.val_thresholds is not None else [0.5],
                pp_min_vox=int(args.pp_min_vox),
                logger=logger,
            )

            # Use best-threshold metrics for checkpoint scoring (stronger for vessels)
            score = 0.7 * float(val["dice_best"]) + 0.3 * float(val["cldice_best"])

            val_rec = {
                "epoch": epoch,
                "val_dice@0.5": float(val["dice@0.5"]),
                "val_dice_best": float(val["dice_best"]),
                "val_cldice_best": float(val["cldice_best"]),
                "val_thr_best_mean": float(val["thr_best_mean"]),
                "val_score": float(score),
            }
            logger.print(json.dumps(val_rec))
            append_jsonl(metrics_path, {"type": "val", **val_rec})

            if score > best_val:
                best_val = score
                torch_save_atomic(
                    args.out_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "best_val": best_val,
                        "args": {k: _to_jsonable(v) for k, v in vars(args).items()},
                    },
                )
                logger.print(f"[ckpt] best.pt updated (score={best_val:.6f})")

        # always save last.pt
        torch_save_atomic(
            args.out_dir / "last.pt",
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val": best_val,
                "args": {k: _to_jsonable(v) for k, v in vars(args).items()},
            },
        )

        # periodic checkpoint
        if (epoch % int(args.save_every) == 0) or (epoch == int(args.epochs)):
            torch_save_atomic(
                args.out_dir / f"ckpt_ep{epoch:03d}.pt",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_val": best_val,
                    "args": {k: _to_jsonable(v) for k, v in vars(args).items()},
                },
            )

        entry: Dict[str, Any] = {"train": train_rec}
        if val_rec is not None:
            entry["val"] = val_rec
        history.append(entry)
        save_json_atomic(history_path, history)

    logger.print("done.")


if __name__ == "__main__":
    main()
