from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Optional deps for post-processing/HD95
try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from monai.metrics import HausdorffDistanceMetric
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False


# -------------------------
# Utilities
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

def _autocast_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("amp dtype must be bf16 or fp16")

def json_dump_safe(obj: Any, indent: int = 2) -> str:
    def _conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, set):
            return sorted(list(x))
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    return json.dumps(obj, indent=indent, default=_conv)

def _resolve_cached_path(uid: str, p: str, cache_dir: Path, kind: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)
    if kind == "image":
        alt = cache_dir / "images" / f"{uid}.pt"
    else:
        alt = cache_dir / "labels" / f"{uid}.pt"
    return str(alt)

def _uid_base(uid: str) -> str:
    """
    Maps augmented uid -> base uid to reuse sampling_db npz.
    Examples:
      foo_aug003 -> foo
      foo__aug003 -> foo
      foo-aug003 -> foo
      foo_aug -> foo
    """
    m = re.match(r"^(.*?)(?:[_\-]{1,2}aug\d*|_aug\d*|-aug\d*|__aug\d*)$", uid)
    if m and m.group(1):
        return m.group(1)
    m = re.match(r"^(.*?)(?:[_\-]{1,2}aug\d+)$", uid)
    if m and m.group(1):
        return m.group(1)
    m = re.match(r"^(.*?)(?:[_\-]{1,2}aug\d+).*?$", uid)
    if m and m.group(1):
        return m.group(1)
    return uid


# -------------------------
# Cache listing + grouped split
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
        raise RuntimeError("No cached items found.")
    return items

def split_train_val_grouped(items: List[Dict[str, str]], val_ratio: float = 0.1, seed: int = 42):
    """
    Prevent leakage: augmented variants of same base UID must not cross train/val.
    """
    groups: Dict[str, List[Dict[str, str]]] = {}
    for it in items:
        b = _uid_base(it["uid"])
        groups.setdefault(b, []).append(it)

    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])

    train = [it for k in keys if k not in val_keys for it in groups[k]]
    val   = [it for k in keys if k in val_keys for it in groups[k]]
    return train, val


# -------------------------
# Small LRU RAM cache
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


# -------------------------
# Patch ops
# -------------------------
def _pad_to_min_shape_torch(vol: torch.Tensor, target_dhw: Tuple[int, int, int], pad_value: float = 0.0) -> torch.Tensor:
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
        return F.pad(vol, pad, mode="constant", value=float(pad_value))
    else:
        return F.pad(vol.unsqueeze(0), pad, mode="constant", value=float(pad_value)).squeeze(0)

def _crop_around_center_torch(vol: torch.Tensor, center_dhw: Tuple[int, int, int], patch_dhw: Tuple[int, int, int]) -> torch.Tensor:
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

def _augment_light_cpu(img: torch.Tensor, lab: torch.Tensor, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    # cheap flips + mild intensity jitter + small noise
    for axis in (1, 2, 3):  # img: C,D,H,W
        if rng.random() < 0.5:
            img = torch.flip(img, dims=(axis,))
            lab = torch.flip(lab, dims=(axis - 1,))

    if rng.random() < 0.20:
        scale = float(rng.uniform(0.95, 1.05))
        shift = float(rng.uniform(-0.03, 0.03))
        img = torch.clamp(img * scale + shift, 0.0, 1.0)

    if rng.random() < 0.15:
        noise = torch.from_numpy(rng.normal(0.0, 0.008, size=img.shape).astype(np.float32))
        img = torch.clamp(img + noise, 0.0, 1.0)

    return img, lab


# -------------------------
# Sampling DB loader
# -------------------------
def load_sampling_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int]]:
    z = np.load(path, allow_pickle=False)
    fg = z["fg"].astype(np.int32)
    hard = z["hardneg"].astype(np.int32)
    shape = tuple(z["shape"].astype(np.int32).tolist())
    return fg, hard, shape


# -------------------------
# Dataset
# -------------------------
class CachedPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[Dict[str, str]],
        cache_dir: Path,
        sampling_db_dir: Optional[Path],
        patch_size: Tuple[int, int, int],
        training: bool,
        pos_ratio: float,
        bg_hard_prob: float,
        seed: int,
        ram_cache_items: int,
        center_jitter_prob: float,
        center_jitter_frac: float,
        online_light_aug: bool = True,
        prefer_sampling_base_uid: bool = True,
        val_force_fg: bool = False,
    ):
        self.items = items
        self.cache_dir = Path(cache_dir)
        self.sampling_db_dir = Path(sampling_db_dir) if sampling_db_dir is not None else None
        self.patch_size = tuple(int(x) for x in patch_size)
        self.training = bool(training)

        self.pos_ratio = float(pos_ratio)
        self.bg_hard_prob = float(bg_hard_prob)

        self.seed = int(seed)
        self.ram_cache_items = int(ram_cache_items)

        self.center_jitter_prob = float(center_jitter_prob) if training else 0.0
        self.center_jitter_frac = float(center_jitter_frac) if training else 0.0

        self.online_light_aug = bool(online_light_aug) if training else False
        self.prefer_sampling_base_uid = bool(prefer_sampling_base_uid)

        self.val_force_fg = bool(val_force_fg)

        self.cache_img = _LRUVolCache(self.ram_cache_items)
        self.cache_lab = _LRUVolCache(self.ram_cache_items)
        self.cache_samp = _LRUVolCache(max_items=max(0, self.ram_cache_items))

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
            img = img.float()
            img = torch.clamp(img, 0.0, 1.0)
            lab = (lab > 0).to(torch.uint8)
            self.cache_img.put(uid, img)
            self.cache_lab.put(uid, lab)
        return img, lab

    def _sampling_uid(self, uid: str) -> str:
        return _uid_base(uid) if self.prefer_sampling_base_uid else uid

    def _load_sampling(self, uid: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.sampling_db_dir is None:
            return np.zeros((0,3), np.int32), np.zeros((0,3), np.int32)

        suid = self._sampling_uid(uid)
        cached = self.cache_samp.get(suid)
        if cached is not None:
            return cached

        p = self.sampling_db_dir / f"{suid}.npz"
        if not p.exists():
            return np.zeros((0,3), np.int32), np.zeros((0,3), np.int32)
        fg, hard, _ = load_sampling_npz(p)
        self.cache_samp.put(suid, (fg, hard))
        return fg, hard

    def _choose_center(self, uid: str, lab: torch.Tensor, rng: np.random.Generator) -> Tuple[int,int,int]:
        D, H, W = lab.shape

        # Validation: pick FG center if possible (stability + relevance)
        if not self.training:
            if self.val_force_fg:
                fg_coords, _ = self._load_sampling(uid)
                if fg_coords.shape[0] > 0:
                    z,y,x = fg_coords[int(rng.integers(0, fg_coords.shape[0]))]
                    return (int(z), int(y), int(x))
            return (D//2, H//2, W//2)

        want_fg = (rng.random() < self.pos_ratio)
        fg_coords, hard_coords = self._load_sampling(uid)

        if want_fg and fg_coords.shape[0] > 0:
            z,y,x = fg_coords[int(rng.integers(0, fg_coords.shape[0]))]
            return (int(z), int(y), int(x))

        # background: prefer hard negatives
        use_hard = (hard_coords.shape[0] > 0) and (rng.random() < self.bg_hard_prob)
        if use_hard:
            z,y,x = hard_coords[int(rng.integers(0, hard_coords.shape[0]))]
            return (int(z), int(y), int(x))

        return (int(rng.integers(0, D)), int(rng.integers(0, H)), int(rng.integers(0, W)))

    def _jitter_center(self, center: Tuple[int,int,int], vol_shape: Tuple[int,int,int], rng: np.random.Generator) -> Tuple[int,int,int]:
        if rng.random() > self.center_jitter_prob:
            return center
        D,H,W = vol_shape
        pd,ph,pw = self.patch_size
        jd = int(round(pd * self.center_jitter_frac))
        jh = int(round(ph * self.center_jitter_frac))
        jw = int(round(pw * self.center_jitter_frac))
        od = int(rng.integers(-jd, jd+1)) if jd>0 else 0
        oh = int(rng.integers(-jh, jh+1)) if jh>0 else 0
        ow = int(rng.integers(-jw, jw+1)) if jw>0 else 0
        cd,ch,cw = center
        cd = int(np.clip(cd+od, 0, max(0,D-1)))
        ch = int(np.clip(ch+oh, 0, max(0,H-1)))
        cw = int(np.clip(cw+ow, 0, max(0,W-1)))
        return (cd,ch,cw)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        uid = it["uid"]

        # Stable validation; diverse training.
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        base = (self.seed + 10007 * idx + 1000003 * wid) & 0x7FFFFFFF
        if self.training:
            base = (base + (time.time_ns() & 0x7FFFFFFF)) & 0x7FFFFFFF
        rng = np.random.default_rng(base)

        img, lab = self._load_pair(uid, it["image_pt"], it["label_pt"])
        img = _pad_to_min_shape_torch(img, self.patch_size, pad_value=0.0)
        lab = _pad_to_min_shape_torch(lab, self.patch_size, pad_value=0)

        center = self._choose_center(uid, lab, rng)
        center = self._jitter_center(center, tuple(lab.shape), rng) if self.training else center

        img_p = _crop_around_center_torch(img, center, self.patch_size)          # [1,pd,ph,pw]
        lab_p = _crop_around_center_torch(lab, center, self.patch_size).long()  # [pd,ph,pw]

        if self.online_light_aug:
            img_p, lab_p = _augment_light_cpu(img_p, lab_p, rng=rng)

        return {"image": img_p.contiguous(), "label": lab_p.contiguous(), "uid": uid}


# -------------------------
# Deep supervision label downsample (MaxPool)
# -------------------------
def downsample_label_maxpool(lab: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return lab
    x = (lab > 0).float().unsqueeze(1)  # [B,1,D,H,W]
    y = F.max_pool3d(x, kernel_size=factor, stride=factor, padding=0)
    return (y.squeeze(1) > 0.5).long()


# -------------------------
# Losses
# -------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)
        fg = probs[:, 1:2]
        tgt = (targets > 0).float().unsqueeze(1)
        dims = (2,3,4)
        inter = (fg * tgt).sum(dims)
        den = (fg + tgt).sum(dims)
        dice = (2*inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=0.75, smooth=1e-5):
        super().__init__()
        self.alpha=float(alpha); self.beta=float(beta); self.smooth=float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)
        p = probs[:,1:2]
        t = (targets>0).float().unsqueeze(1)
        dims=(2,3,4)
        tp = (p*t).sum(dims)
        fp = (p*(1-t)).sum(dims)
        fn = ((1-p)*t).sum(dims)
        tv = (tp + self.smooth)/(tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1.0 - tv.mean()

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-img, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
    p2 = -F.max_pool3d(-img, kernel_size=(1,3,1), stride=1, padding=(0,1,0))
    p3 = -F.max_pool3d(-img, kernel_size=(1,1,3), stride=1, padding=(0,0,1))
    return torch.min(torch.min(p1,p2),p3)

def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)

def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))

def _soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    img = torch.clamp(img, 0, 1)
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel*delta)
    return torch.nan_to_num(skel, nan=0.0, posinf=0.0, neginf=0.0)

class SoftclDiceLoss(nn.Module):
    def __init__(self, iters: int = 15, smooth: float = 1e-5):
        super().__init__()
        self.iters=int(iters); self.smooth=float(smooth)

    def set_iters(self, iters: int):
        self.iters=int(iters)

    def forward(self, probs_fg: torch.Tensor, targets_fg: torch.Tensor) -> torch.Tensor:
        p = probs_fg.float()
        t = targets_fg.float()
        sp = _soft_skel(p, self.iters)
        st = _soft_skel(t, self.iters)
        tprec = (sp*t).sum() + self.smooth
        tprec = tprec / (sp.sum() + self.smooth)
        tsens = (st*p).sum() + self.smooth
        tsens = tsens / (st.sum() + self.smooth)
        cl = (2*tprec*tsens)/(tprec+tsens+self.smooth)
        cl = torch.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)
        return 1.0 - cl

class BoundaryLoss(nn.Module):
    def forward(self, probs_fg: torch.Tensor, targets_fg: torch.Tensor) -> torch.Tensor:
        p = probs_fg.float()
        t = targets_fg.float()
        t_edge = torch.clamp(_soft_dilate(t) - _soft_erode(t), 0.0, 1.0)
        p_edge = torch.clamp(_soft_dilate(p) - _soft_erode(p), 0.0, 1.0)
        return F.l1_loss(p_edge, t_edge)

class FocalCE(nn.Module):
    def __init__(self, alpha: float=0.80, gamma: float=2.0):
        super().__init__()
        self.alpha=float(alpha); self.gamma=float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)
        p = torch.exp(logp)
        tgt = targets.long().unsqueeze(1)
        logpt = torch.gather(logp, dim=1, index=tgt).squeeze(1)
        pt = torch.gather(p, dim=1, index=tgt).squeeze(1)
        alpha_t = torch.where(
            targets > 0,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device),
        )
        loss = -alpha_t * (1.0 - pt).clamp_min(0)**self.gamma * logpt
        return loss.mean()

class TubularLoss(nn.Module):
    def __init__(self, ce_weight_fg: float):
        super().__init__()
        self.tv = TverskyLoss(0.25, 0.75)
        self.dice = SoftDiceLoss()
        self.cldice = SoftclDiceLoss(15)
        self.bnd = BoundaryLoss()
        self.focal = FocalCE(alpha=0.80, gamma=2.0)
        self.register_buffer("ce_w", torch.tensor([1.0, float(ce_weight_fg)], dtype=torch.float32))

        self.w_tv=0.55
        self.w_dice=0.15
        self.w_ce=0.20
        self.w_bnd=0.08
        self.w_focal=0.02
        self.w_cldice=0.15

    def set_weights(self, w_focal: float, w_cldice: float, w_bnd: float, ce_wfg: float, cl_iters: int,
                    w_tv: Optional[float]=None, w_dice: Optional[float]=None, w_ce: Optional[float]=None):
        self.w_focal=float(w_focal)
        self.w_cldice=float(w_cldice)
        self.w_bnd=float(w_bnd)
        if w_tv is not None: self.w_tv=float(w_tv)
        if w_dice is not None: self.w_dice=float(w_dice)
        if w_ce is not None: self.w_ce=float(w_ce)
        self.ce_w[0]=1.0; self.ce_w[1]=float(ce_wfg)
        self.cldice.set_iters(int(cl_iters))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss = loss + self.w_tv   * self.tv(logits, targets)
        loss = loss + self.w_dice * self.dice(logits, targets)

        if self.w_ce > 0:
            loss = loss + self.w_ce * F.cross_entropy(logits.float(), targets.long(), weight=self.ce_w)

        if self.w_focal > 0:
            loss = loss + self.w_focal * self.focal(logits, targets)

        if self.w_cldice > 0 or self.w_bnd > 0:
            probs = torch.softmax(logits.float(), dim=1)
            pfg = probs[:,1:2]
            tfg = (targets>0).float().unsqueeze(1)
            if self.w_cldice > 0:
                loss = loss + self.w_cldice * self.cldice(pfg, tfg)
            if self.w_bnd > 0:
                loss = loss + self.w_bnd * self.bnd(pfg, tfg)

        return torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)

def ft_loss_schedule(epoch: int, total_epochs: int) -> Tuple[float,float,float,int,float]:
    """
    A Dice-friendly schedule:
      - Keep CE high for tiny vessels.
      - clDice helps continuity but is capped so it doesn't suppress Dice.
    Returns: (w_focal, w_cldice, w_bnd, cl_iters, ce_wfg)
    """
    if epoch <= 10:
        return 0.05, 0.12, 0.10, 12, 30.0

    t = (epoch - 10) / max(1, total_epochs - 10)
    t = float(np.clip(t, 0.0, 1.0))

    w_focal = 0.05 * (1.0 - t)          # fade out
    w_bnd   = 0.10 + 0.05 * t           # mild increase
    w_cl    = 0.12 + 0.13 * t           # cap ~0.25
    iters   = int(round(12 + 8 * t))    # 12 -> 20
    ce_w    = 30.0 - 12.0 * t           # 30 -> 18 (don’t go too low)

    return w_focal, w_cl, w_bnd, iters, ce_w

def deep_sup_weights(epoch: int, start: int, end: int, w2: float, w3: float, w4: float) -> Tuple[float,float,float]:
    if epoch <= start:
        return w2,w3,w4
    if epoch >= end:
        return 0.0,0.0,0.0
    t = (epoch-start)/max(1,(end-start))
    s = 1.0 - t
    return w2*s, w3*s, w4*s


# -------------------------
# Model blocks
# -------------------------
def _gn(ch: int, groups: int = 16) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = _gn(out_ch, gn_groups)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = _gn(out_ch, gn_groups)
        self.drop = nn.Dropout3d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.silu(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = F.silu(self.gn2(self.conv2(x)))
        return x

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, stride=2, bias=False),
            _gn(out_ch, gn_groups),
            nn.SiLU(),
        )
        self.block = ConvBlock(out_ch, out_ch, gn_groups, dropout=dropout)

    def forward(self, x):
        return self.block(self.down(x))

def _make_base_grid_3d(B: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    zz = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    grid = torch.stack([x, y, z], dim=0)  # (3,D,H,W)
    return grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B,3,D,H,W)

def _make_grid_for_sample(B: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = _make_base_grid_3d(B, D, H, W, device, dtype)  # (B,3,D,H,W)
    return g.permute(0,2,3,4,1).contiguous()          # (B,D,H,W,3)

class CoordInject3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(3, channels, 1, bias=False),
            _gn(channels, 16),
            nn.SiLU(),
        )

    def forward(self, x):
        B,C,D,H,W = x.shape
        g = _make_base_grid_3d(B,D,H,W,x.device,x.dtype)
        return x + self.proj(g)

class SnakeRefine3DXYZ(nn.Module):
    """
    FIXED: positive and negative direction branches now use different offsets.
    """
    def __init__(self, channels: int, K: int = 3, offset_scale: float = 0.25):
        super().__init__()
        assert K >= 3 and K % 2 == 1
        self.K = int(K)
        self.half = self.K // 2
        self.offset_scale = float(offset_scale)
        self.offset_pred = nn.Conv3d(channels, 3*(self.K-1), 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels*self.K, channels, 1, bias=False),
            _gn(channels, 16),
            nn.SiLU(),
        )

    def forward(self, x):
        B,C,D,H,W = x.shape
        delta = torch.tanh(self.offset_pred(x)) * self.offset_scale
        delta = delta.view(B, 3, self.K-1, D,H,W)

        base = _make_grid_for_sample(B,D,H,W,x.device,delta.dtype)  # (B,D,H,W,3)
        grids = [base]

        # + direction: offsets 0..half-1
        cum = torch.zeros((B,3,D,H,W), device=x.device, dtype=delta.dtype)
        for s in range(self.half):
            cum = cum + delta[:,:,s]
            g = base.clone()
            g[...,0] = g[...,0] + (cum[:,0] * (2.0/max(1,W-1)))
            g[...,1] = g[...,1] + (cum[:,1] * (2.0/max(1,H-1)))
            g[...,2] = g[...,2] + (cum[:,2] * (2.0/max(1,D-1)))
            grids.append(g)

        # - direction: offsets half..(K-2)
        cum = torch.zeros((B,3,D,H,W), device=x.device, dtype=delta.dtype)
        for s in range(self.half):
            cum = cum + delta[:,:,self.half + s]
            g = base.clone()
            g[...,0] = g[...,0] - (cum[:,0] * (2.0/max(1,W-1)))
            g[...,1] = g[...,1] - (cum[:,1] * (2.0/max(1,H-1)))
            g[...,2] = g[...,2] - (cum[:,2] * (2.0/max(1,D-1)))
            grids.append(g)

        sampled = [F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True) for g in grids]
        y = self.fuse(torch.cat(sampled, dim=1))
        return x + y

class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_snake: bool, snake_k: int,
                 gn_groups: int = 16, checkpoint_block: bool = False, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, gn_groups, dropout=dropout)
        self.snake = SnakeRefine3DXYZ(out_ch, K=snake_k) if use_snake else nn.Identity()
        self.checkpoint_block = bool(checkpoint_block)

    def forward(self, x, skip):
        x = self.up(x)
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        if dz or dy or dx:
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2, dz//2, dz-dz//2])
        x = torch.cat([skip, x], dim=1)
        if self.training and self.checkpoint_block:
            x = checkpoint(self.block, x, use_reentrant=False)
        else:
            x = self.block(x)
        return self.snake(x)

def _parse_stages(s: str) -> Set[str]:
    out=set()
    for t in s.split(","):
        t=t.strip()
        if t: out.add(t)
    return out

def _parse_mamba_axes(s: str) -> List[str]:
    axes=[]
    for t in s.split(","):
        t=t.strip().lower()
        if not t:
            continue
        if set(t)!=set("dhw") or len(t)!=3:
            raise ValueError(f"Bad mamba axis order: {t}")
        axes.append(t)
    return axes

def _permute_for_axis(x: torch.Tensor, order: str) -> torch.Tensor:
    axes={'d':2,'h':3,'w':4}
    perm=[0,1,axes[order[0]],axes[order[1]],axes[order[2]]]
    return x.permute(*perm).contiguous()

class BiMambaSSM(nn.Module):
    def __init__(self, dim: int, axes: List[str], dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.axes = axes
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise RuntimeError("mamba_ssm not installed: pip install mamba-ssm") from e
        self.mf = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mb = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    def _run_seq(self, seq: torch.Tensor) -> torch.Tensor:
        y = self.norm(seq)
        y_f = self.mf(y)
        y_r = torch.flip(self.mb(torch.flip(y, dims=(1,))), dims=(1,))
        y = y_f + y_r
        y = self.dropout(y)
        return seq + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,D,H,W = x.shape
        outs=[]
        for ax in self.axes:
            xp = _permute_for_axis(x, ax)
            _,_,Dp,Hp,Wp = xp.shape
            seq = xp.permute(0,2,3,4,1).reshape(B, Dp*Hp*Wp, C)
            seq = self._run_seq(seq)
            y = seq.reshape(B,Dp,Hp,Wp,C).permute(0,4,1,2,3).contiguous()
            cur = {'d': 2 + ax.index('d'), 'h': 2 + ax.index('h'), 'w': 2 + ax.index('w')}
            y = y.permute(0,1,cur['d'],cur['h'],cur['w']).contiguous()
            outs.append(y)
        return torch.stack(outs, dim=0).mean(dim=0)

class MambaSnakeUNet3D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int, gn_groups: int, mamba_layers: int,
                 mamba_axes: List[str], snake_stages: Iterable[str], snake_k: int,
                 checkpoint_bottleneck: bool, checkpoint_decoder: bool,
                 coord_inject_levels: str = "bottleneck,enc2,enc3",
                 dropout: float = 0.0,
                 mamba_dropout: float = 0.0):
        super().__init__()
        self.checkpoint_bottleneck = bool(checkpoint_bottleneck)
        self.coord_levels = set([t.strip() for t in coord_inject_levels.split(",") if t.strip()])

        self.stem = ConvBlock(in_ch, base, gn_groups, dropout=dropout)
        self.c0 = CoordInject3D(base) if "enc0" in self.coord_levels else nn.Identity()
        self.d1 = Down(base, base*2, gn_groups, dropout=dropout)
        self.c1 = CoordInject3D(base*2) if "enc1" in self.coord_levels else nn.Identity()
        self.d2 = Down(base*2, base*4, gn_groups, dropout=dropout)
        self.c2 = CoordInject3D(base*4) if "enc2" in self.coord_levels else nn.Identity()
        self.d3 = Down(base*4, base*8, gn_groups, dropout=dropout)
        self.c3 = CoordInject3D(base*8) if "enc3" in self.coord_levels else nn.Identity()
        self.d4 = Down(base*8, base*16, gn_groups, dropout=dropout)

        bott = base*16
        self.bpre = ConvBlock(bott, bott, gn_groups, dropout=dropout)
        self.cb = CoordInject3D(bott) if "bottleneck" in self.coord_levels else nn.Identity()
        self.mamba = nn.ModuleList([BiMambaSSM(bott, axes=mamba_axes, dropout=mamba_dropout) for _ in range(int(mamba_layers))])
        self.bpost = ConvBlock(bott, bott, gn_groups, dropout=dropout)

        stages = set(snake_stages)
        self.u4 = Up(bott,    base*8, base*8, use_snake=("u4" in stages), snake_k=snake_k, gn_groups=gn_groups, checkpoint_block=checkpoint_decoder, dropout=dropout)
        self.u3 = Up(base*8,  base*4, base*4, use_snake=("u3" in stages), snake_k=snake_k, gn_groups=gn_groups, checkpoint_block=checkpoint_decoder, dropout=dropout)
        self.u2 = Up(base*4,  base*2, base*2, use_snake=("u2" in stages), snake_k=snake_k, gn_groups=gn_groups, checkpoint_block=checkpoint_decoder, dropout=dropout)
        self.u1 = Up(base*2,  base,   base,   use_snake=("u1" in stages), snake_k=snake_k, gn_groups=gn_groups, checkpoint_block=checkpoint_decoder, dropout=dropout)

        self.head = nn.Conv3d(base, num_classes, 1)
        self.aux2 = nn.Conv3d(base*2, num_classes, 1)
        self.aux3 = nn.Conv3d(base*4, num_classes, 1)
        self.aux4 = nn.Conv3d(base*8, num_classes, 1)

    def _run_bottleneck(self, x):
        x = self.bpre(x)
        x = self.cb(x)
        if self.checkpoint_bottleneck and self.training:
            for blk in self.mamba:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.mamba:
                x = blk(x)
        x = self.bpost(x)
        return x

    def forward(self, x):
        x0 = self.c0(self.stem(x))
        x1 = self.c1(self.d1(x0))
        x2 = self.c2(self.d2(x1))
        x3 = self.c3(self.d3(x2))
        x4 = self.d4(x3)
        b  = self._run_bottleneck(x4)

        y4 = self.u4(b, x3)
        y3 = self.u3(y4, x2)
        y2 = self.u2(y3, x1)
        y1 = self.u1(y2, x0)

        out = {"logits": self.head(y1)}
        if self.training:
            out["aux2"] = self.aux2(y2)
            out["aux3"] = self.aux3(y3)
            out["aux4"] = self.aux4(y4)
        return out


# -------------------------
# GPU geometric augmentation (light)
# -------------------------
def gpu_geom_aug(
    img: torch.Tensor,          # [B,1,D,H,W]
    lab: torch.Tensor,          # [B,D,H,W]
    rot_prob: float,
    rot_deg: float,
    elastic_prob: float,
    elastic_alpha: float,
    elastic_coarse: int,
    do_elastic: bool,
):
    B, C, D, H, W = img.shape
    device = img.device
    dtype_grid = torch.float32

    lab_in = lab.unsqueeze(1).float()
    grid = _make_grid_for_sample(B, D, H, W, device=device, dtype=dtype_grid)

    # Rotation
    if rot_prob > 0 and torch.rand((), device=device) < rot_prob:
        deg = float(rot_deg)
        ax = (torch.rand((), device=device) * 2 - 1) * deg * (math.pi / 180.0)
        ay = (torch.rand((), device=device) * 2 - 1) * deg * (math.pi / 180.0)
        az = (torch.rand((), device=device) * 2 - 1) * deg * (math.pi / 180.0)

        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)

        Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=device, dtype=dtype_grid)
        Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=device, dtype=dtype_grid)
        Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=device, dtype=dtype_grid)
        R = (Rz @ Ry @ Rx).unsqueeze(0).repeat(B, 1, 1)

        g = grid.view(B, -1, 3)
        g = torch.bmm(g, R.transpose(1, 2))
        grid = g.view(B, D, H, W, 3)

    # Elastic (throttled)
    if do_elastic and elastic_prob > 0 and torch.rand((), device=device) < elastic_prob:
        coarse = int(max(2, elastic_coarse))
        dc = max(2, D // coarse)
        hc = max(2, H // coarse)
        wc = max(2, W // coarse)

        disp_c = torch.randn((B, 3, dc, hc, wc), device=device, dtype=dtype_grid)
        disp = F.interpolate(disp_c, size=(D, H, W), mode="trilinear", align_corners=True)

        sx = (2.0 / max(1, W - 1)) * float(elastic_alpha)
        sy = (2.0 / max(1, H - 1)) * float(elastic_alpha)
        sz = (2.0 / max(1, D - 1)) * float(elastic_alpha)

        disp[:, 0] *= sx
        disp[:, 1] *= sy
        disp[:, 2] *= sz

        disp = disp.permute(0, 2, 3, 4, 1).contiguous()
        grid = grid + disp

    img_aug = F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True).to(dtype=img.dtype)
    lab_aug = F.grid_sample(lab_in, grid, mode="nearest", padding_mode="border", align_corners=True)[:, 0].long()
    return img_aug, lab_aug


# -------------------------
# Sliding-window inference (full-volume val)
# -------------------------
_WEIGHT_CACHE: Dict[Tuple[int, int, int, str, str], torch.Tensor] = {}

def _make_blend_weight_3d(patch_dhw: Tuple[int,int,int], device: torch.device, kind: str) -> torch.Tensor:
    key = (patch_dhw[0],patch_dhw[1],patch_dhw[2],kind,str(device))
    w = _WEIGHT_CACHE.get(key, None)
    if w is not None:
        return w
    pd,ph,pw = patch_dhw
    if kind == "hann":
        wd = torch.hann_window(pd, periodic=False, device=device, dtype=torch.float32).clamp_min(1e-3)
        wh = torch.hann_window(ph, periodic=False, device=device, dtype=torch.float32).clamp_min(1e-3)
        ww = torch.hann_window(pw, periodic=False, device=device, dtype=torch.float32).clamp_min(1e-3)
    else:
        def gauss(n, sigma_frac=0.125):
            x = torch.linspace(-1,1,n,device=device,dtype=torch.float32)
            g = torch.exp(-0.5*(x/(sigma_frac))**2)
            return g.clamp_min(1e-3)
        wd=gauss(pd); wh=gauss(ph); ww=gauss(pw)
    w3 = (wd[:,None,None]*wh[None,:,None]*ww[None,None,:])
    w3 = (w3 / w3.max().clamp_min(1e-6)).contiguous()
    w = w3[None,None]  # [1,1,D,H,W]
    _WEIGHT_CACHE[key]=w
    return w

def _safe_pad3d(img: torch.Tensor, halo: int) -> torch.Tensor:
    if halo <= 0:
        return img
    pad = [halo,halo,halo,halo,halo,halo]
    try:
        return F.pad(img, pad, mode="reflect")
    except Exception:
        return F.pad(img, pad, mode="replicate")

def _make_starts(L: int, P: int, stride: int) -> List[int]:
    max0 = max(0, L-P)
    if max0 == 0:
        return [0]
    s = list(range(0, max0+1, stride))
    if s[-1] != max0:
        s.append(max0)
    return s

@torch.inference_mode()
def sliding_window_logits(model: nn.Module, img: torch.Tensor, patch: Tuple[int,int,int],
                          overlap: float, halo: int, blend: str,
                          amp: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    _,_,D0,H0,W0 = img.shape
    pd,ph,pw = patch

    pad_d = max(0, pd - D0)
    pad_h = max(0, ph - H0)
    pad_w = max(0, pw - W0)
    if pad_d or pad_h or pad_w:
        img = F.pad(img, [pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2, pad_d//2, pad_d-pad_d//2])

    _,_,D,H,W = img.shape
    imgp = _safe_pad3d(img, halo)

    sd = max(1, int(pd*(1.0-overlap)))
    sh = max(1, int(ph*(1.0-overlap)))
    sw = max(1, int(pw*(1.0-overlap)))

    dzs = _make_starts(D, pd, sd)
    hzs = _make_starts(H, ph, sh)
    wzs = _make_starts(W, pw, sw)

    weight = _make_blend_weight_3d((pd,ph,pw), img.device, blend)
    out = None
    wsum = torch.zeros((1,1,D,H,W), device=img.device, dtype=torch.float32)

    for dz in dzs:
        for hy in hzs:
            for wx in wzs:
                patch_ext = imgp[:,:, dz:dz+pd+2*halo, hy:hy+ph+2*halo, wx:wx+pw+2*halo]
                with torch.amp.autocast(device_type="cuda", enabled=amp, dtype=amp_dtype):
                    logits_ext = model(patch_ext)["logits"].float()
                logits = logits_ext[:,:, halo:halo+pd, halo:halo+ph, halo:halo+pw] if halo>0 else logits_ext
                if out is None:
                    out = torch.zeros((1, logits.shape[1], D,H,W), device=img.device, dtype=torch.float32)
                out[:,:,dz:dz+pd, hy:hy+ph, wx:wx+pw] += logits * weight
                wsum[:,:,dz:dz+pd, hy:hy+ph, wx:wx+pw] += weight

    out = out / wsum.clamp_min(1e-6)

    if pad_d or pad_h or pad_w:
        dd0 = pad_d//2; hh0 = pad_h//2; ww0 = pad_w//2
        out = out[:,:, dd0:dd0+D0, hh0:hh0+H0, ww0:ww0+W0]
    return out


# -------------------------
# Post-processing + metrics
# -------------------------
def remove_small_components_3d(mask: np.ndarray, min_vox: int = 0, min_rel: float = 0.001) -> np.ndarray:
    if not _HAS_SCIPY:
        return mask
    lab, n = ndi.label(mask.astype(bool), structure=np.ones((3,3,3), dtype=np.uint8))
    if n <= 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = int(sizes.max()) if sizes.size else 0
    thr_rel = int(math.ceil(float(min_rel) * largest)) if (min_rel and largest > 0) else 0
    thr = max(int(min_vox), thr_rel)
    keep = sizes >= thr
    keep[0] = False
    return keep[lab].astype(np.uint8)

@torch.inference_mode()
def dice_fg_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float=1e-5) -> float:
    probs = torch.softmax(logits.float(), dim=1)
    pred = (torch.argmax(probs, dim=1) > 0).float()
    tgt  = (targets > 0).float()
    inter = (pred*tgt).sum()
    den = pred.sum() + tgt.sum()
    return float(((2*inter+eps)/(den+eps)).item())

@torch.inference_mode()
def cldice_from_logits(logits: torch.Tensor, targets: torch.Tensor, iters: int = 20) -> float:
    probs = torch.softmax(logits.float(), dim=1)[:,1:2]
    tgt = (targets>0).float().unsqueeze(1)
    cl = SoftclDiceLoss(iters=iters)(probs, tgt)
    return float((1.0 - cl).item())

def hd95_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(max(pred.shape))
    if _HAS_MONAI:
        p = torch.from_numpy(pred[None,None].astype(np.float32))
        g = torch.from_numpy(gt[None,None].astype(np.float32))
        metric = HausdorffDistanceMetric(include_background=True, percentile=95.0, directed=False)
        v = metric(p, g)
        return float(v.item())
    if not _HAS_SCIPY:
        return float("nan")
    er_p = ndi.binary_erosion(pred, structure=np.ones((3,3,3), bool), iterations=1)
    er_g = ndi.binary_erosion(gt, structure=np.ones((3,3,3), bool), iterations=1)
    sp = pred & (~er_p)
    sg = gt & (~er_g)
    if sp.sum()==0 or sg.sum()==0:
        return float(max(pred.shape))
    dt_g = ndi.distance_transform_edt(~sg)
    dt_p = ndi.distance_transform_edt(~sp)
    d = np.concatenate([dt_g[sp], dt_p[sg]], axis=0)
    return float(np.percentile(d, 95))


# -------------------------
# EMA
# -------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self._init(model)

    def _init(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach().float(), alpha=(1.0 - d))

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.dtype).to(p.device))
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name].to(p.device))


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    cache_dir: Path
    out_dir: Path
    sampling_db_dir: Optional[Path]

    gpu: int = 0
    val_gpu: int = 0

    epochs: int = 200
    epoch_size: int = 1024

    batch_size: int = 1
    accum_steps: int = 2
    num_workers: int = 8
    seed: int = 42
    val_ratio: float = 0.10

    # A100-friendly default
    patch_size: Tuple[int,int,int] = (192,192,192)
    val_patch_size: Tuple[int,int,int] = (192,192,192)

    lr: float = 2e-5
    weight_decay: float = 5e-3
    warmup_epochs: int = 5
    grad_clip: float = 0.8

    amp: bool = True
    amp_dtype: str = "bf16"

    # sampling
    pos_ratio: float = 0.70
    bg_hard_prob: float = 0.85
    ram_cache_items: int = 0
    center_jitter_prob: float = 0.75
    center_jitter_frac: float = 0.30
    online_light_aug: bool = True

    # GPU geom aug (light)
    rot_prob: float = 0.05
    rot_deg: float = 8.0
    elastic_prob: float = 0.02
    elastic_alpha: float = 1.5
    elastic_coarse: int = 6
    elastic_every: int = 4

    # model
    base_ch: int = 32
    gn_groups: int = 16
    mamba_layers: int = 4
    mamba_axes: Tuple[str,...] = ("dhw","hwd","wdh")
    snake_stages: Tuple[str,...] = ("u3","u2","u1")  # enable u1 by default
    snake_k: int = 3
    checkpoint_bottleneck: bool = True
    checkpoint_decoder: bool = False
    coord_inject_levels: str = "bottleneck,enc2,enc3"
    dropout: float = 0.0
    mamba_dropout: float = 0.0

    # deep supervision
    ds_w2: float = 0.50
    ds_w3: float = 0.25
    ds_w4: float = 0.125
    ds_decay_start: int = 80
    ds_decay_end: int = 200

    # validation
    val_full: bool = True
    val_full_max: int = 16
    val_interval: int = 5
    val_async: bool = False

    # overlap schedule
    val_overlap_early: float = 0.45
    val_overlap_late: float = 0.65
    val_overlap_switch: int = 60

    val_halo: int = 16
    blend_window: str = "hann"
    val_rm_small: bool = True
    val_min_cc_vox: int = 0
    val_min_cc_rel: float = 0.001

    # HD95
    hd95_start: int = 10

    # best selection weights
    select_w_dice: float = 0.60
    select_w_cldice: float = 0.40
    select_cl_iters: int = 25

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # compile
    compile: bool = False


# -------------------------
# LR schedule
# -------------------------
def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

def _grads_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(model, loader, optimizer, scaler, loss_obj: TubularLoss, cfg: TrainConfig,
                    epoch: int, global_step: int, total_steps: int, ema: Optional[ModelEMA]=None) -> Tuple[float,int]:
    model.train()
    total=0.0; n=0
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = _autocast_dtype(cfg.amp_dtype)
    warmup_steps = cfg.warmup_epochs * len(loader)

    w2,w3,w4 = deep_sup_weights(epoch, cfg.ds_decay_start, cfg.ds_decay_end, cfg.ds_w2, cfg.ds_w3, cfg.ds_w4)

    for step, batch in enumerate(loader, start=1):
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()

        do_elastic = (cfg.elastic_every <= 1) or ((global_step % cfg.elastic_every) == 0)
        if cfg.rot_prob > 0 or cfg.elastic_prob > 0:
            img, lab = gpu_geom_aug(
                img, lab,
                cfg.rot_prob, cfg.rot_deg,
                cfg.elastic_prob, cfg.elastic_alpha, cfg.elastic_coarse,
                do_elastic=do_elastic
            )

        lr = cosine_with_warmup(global_step, total_steps, warmup_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast(device_type="cuda", enabled=cfg.amp, dtype=amp_dtype):
            out = model(img)
            loss = loss_obj(out["logits"], lab)

            if w2>0 or w3>0 or w4>0:
                lab2 = downsample_label_maxpool(lab, factor=2)
                lab3 = downsample_label_maxpool(lab, factor=4)
                lab4 = downsample_label_maxpool(lab, factor=8)
                loss = loss + w2 * loss_obj(out["aux2"], lab2)
                loss = loss + w3 * loss_obj(out["aux3"], lab3)
                loss = loss + w4 * loss_obj(out["aux4"], lab4)

            loss = loss / max(1, cfg.accum_steps)

        if not torch.isfinite(loss):
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
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                global_step += 1
                continue

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        total += float(loss.item()) * max(1, cfg.accum_steps)
        n += 1
        global_step += 1

    return total/max(1,n), global_step

@torch.inference_mode()
def validate_patchwise(model, loader, loss_obj: TubularLoss, cfg: TrainConfig) -> Dict[str,float]:
    model.eval()
    amp_dtype = _autocast_dtype(cfg.amp_dtype)
    total=0.0; dfg=0.0; cd=0.0; n=0
    for batch in loader:
        img = batch["image"].cuda(non_blocking=True)
        lab = batch["label"].cuda(non_blocking=True).long()
        with torch.amp.autocast(device_type="cuda", enabled=cfg.amp, dtype=amp_dtype):
            out = model(img)
            loss = loss_obj(out["logits"], lab)
        total += float(loss.item())
        dfg += dice_fg_from_logits(out["logits"], lab)
        cd  += cldice_from_logits(out["logits"], lab, iters=cfg.select_cl_iters)
        n += 1
    return {
        "val_loss": total/max(1,n),
        "val_fg_dice": dfg/max(1,n),
        "val_fg_cldice": cd/max(1,n),
    }

@torch.inference_mode()
def validate_full_volume(model: nn.Module, val_items: List[Dict[str,str]], cfg: TrainConfig, epoch_id: int) -> Dict[str, Any]:
    """
    Full-volume sliding-window validation (sync).
    Uses a random subset each time for representativeness.
    """
    model.eval()
    amp_dtype = _autocast_dtype(cfg.amp_dtype)

    overlap = cfg.val_overlap_early if epoch_id < cfg.val_overlap_switch else cfg.val_overlap_late
    compute_hd95 = (epoch_id >= cfg.hd95_start)

    m = min(int(cfg.val_full_max), len(val_items))
    rng = np.random.default_rng(int(cfg.seed) + 1337 * int(epoch_id))
    idxs = rng.choice(len(val_items), size=m, replace=False)

    dices=[]; clds=[]; hd95s=[]
    for ii in idxs:
        it = val_items[int(ii)]
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], Path(cfg.cache_dir), "image")
        lp = _resolve_cached_path(uid, it["label_pt"], Path(cfg.cache_dir), "label")

        img = torch.load(ip, map_location="cpu").float()  # [1,D,H,W]
        lab = torch.load(lp, map_location="cpu")
        lab = (lab>0).to(torch.uint8)

        img = img.unsqueeze(0).cuda(non_blocking=True)         # [1,1,D,H,W]
        labt = lab.unsqueeze(0).cuda(non_blocking=True).long() # [1,D,H,W]

        logits = sliding_window_logits(
            model, img, cfg.val_patch_size, overlap, cfg.val_halo, cfg.blend_window,
            cfg.amp, amp_dtype
        )

        dices.append(dice_fg_from_logits(logits, labt))
        clds.append(cldice_from_logits(logits, labt, iters=cfg.select_cl_iters))

        if compute_hd95:
            probs = torch.softmax(logits.float(), dim=1)
            pred = (torch.argmax(probs, dim=1) > 0).detach().cpu().numpy().astype(np.uint8)[0]
            gt   = (lab.numpy() > 0).astype(np.uint8)
            if cfg.val_rm_small and _HAS_SCIPY and (cfg.val_min_cc_rel>0 or cfg.val_min_cc_vox>0):
                pred = remove_small_components_3d(pred, min_vox=cfg.val_min_cc_vox, min_rel=cfg.val_min_cc_rel)
            hd95s.append(hd95_binary(pred, gt))

        torch.cuda.empty_cache()

    return {
        "val_full_fg_dice": float(np.mean(dices)) if dices else float("nan"),
        "val_full_fg_cldice": float(np.mean(clds)) if clds else float("nan"),
        "val_full_hd95": float(np.mean(hd95s)) if (compute_hd95 and hd95s) else float("nan"),
        "val_full_overlap_used": float(overlap),
        "val_full_hd95_enabled": bool(compute_hd95),
    }


# -------------------------
# Async full-val worker (optional, if you have a separate val GPU)
# -------------------------
def validate_full_worker(payload: Dict[str,Any], q):
    cfgd = payload["cfg"]
    cfg = TrainConfig(**cfgd)
    ckpt_path = Path(payload["ckpt_path"])
    val_items = payload["val_items"]
    epoch_id = int(payload["epoch_id"])

    dev_count = torch.cuda.device_count()
    if int(cfg.val_gpu) < 0 or int(cfg.val_gpu) >= dev_count:
        q.put({"epoch_id": epoch_id, "error": f"val_gpu {cfg.val_gpu} invalid; visible device_count={dev_count}"})
        return

    torch.cuda.set_device(int(cfg.val_gpu))
    device = torch.device(f"cuda:{int(cfg.val_gpu)}")
    enable_perf_flags(tf32=True, cudnn_benchmark=True)
    amp_dtype = _autocast_dtype(cfg.amp_dtype)

    model = MambaSnakeUNet3D(
        in_ch=1, num_classes=2, base=cfg.base_ch, gn_groups=cfg.gn_groups,
        mamba_layers=cfg.mamba_layers, mamba_axes=list(cfg.mamba_axes),
        snake_stages=cfg.snake_stages, snake_k=cfg.snake_k,
        checkpoint_bottleneck=False, checkpoint_decoder=False,
        coord_inject_levels=cfg.coord_inject_levels,
        dropout=cfg.dropout, mamba_dropout=cfg.mamba_dropout,
    ).to(device)

    sd = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Full val on val_gpu
    overlap = cfg.val_overlap_early if epoch_id < cfg.val_overlap_switch else cfg.val_overlap_late
    compute_hd95 = (epoch_id >= cfg.hd95_start)

    m = min(int(cfg.val_full_max), len(val_items))
    rng = np.random.default_rng(int(cfg.seed) + 1337 * int(epoch_id))
    idxs = rng.choice(len(val_items), size=m, replace=False)

    dices=[]; clds=[]; hd95s=[]
    for ii in idxs:
        it = val_items[int(ii)]
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], Path(cfg.cache_dir), "image")
        lp = _resolve_cached_path(uid, it["label_pt"], Path(cfg.cache_dir), "label")

        img = torch.load(ip, map_location="cpu").float()
        lab = torch.load(lp, map_location="cpu")
        lab = (lab>0).to(torch.uint8)

        img = img.unsqueeze(0).to(device, non_blocking=True)
        labt = lab.unsqueeze(0).to(device, non_blocking=True).long()

        logits = sliding_window_logits(model, img, cfg.val_patch_size, overlap, cfg.val_halo, cfg.blend_window,
                                       cfg.amp, amp_dtype)

        dices.append(dice_fg_from_logits(logits, labt))
        clds.append(cldice_from_logits(logits, labt, iters=cfg.select_cl_iters))

        if compute_hd95:
            probs = torch.softmax(logits.float(), dim=1)
            pred = (torch.argmax(probs, dim=1) > 0).detach().cpu().numpy().astype(np.uint8)[0]
            gt   = (lab.numpy() > 0).astype(np.uint8)
            if cfg.val_rm_small and _HAS_SCIPY and (cfg.val_min_cc_rel>0 or cfg.val_min_cc_vox>0):
                pred = remove_small_components_3d(pred, min_vox=cfg.val_min_cc_vox, min_rel=cfg.val_min_cc_rel)
            hd95s.append(hd95_binary(pred, gt))

        torch.cuda.empty_cache()

    q.put({
        "epoch_id": epoch_id,
        "val_full_fg_dice": float(np.mean(dices)) if dices else float("nan"),
        "val_full_fg_cldice": float(np.mean(clds)) if clds else float("nan"),
        "val_full_hd95": float(np.mean(hd95s)) if (compute_hd95 and hd95s) else float("nan"),
        "val_full_overlap_used": float(overlap),
        "val_full_hd95_enabled": bool(compute_hd95),
    })


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sampling-db-dir", type=Path, default=None)

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--val-gpu", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--epoch-size", type=int, default=1024)

    ap.add_argument("--patch-size", type=int, nargs=3, default=[192,192,192])
    ap.add_argument("--val-patch-size", type=int, nargs=3, default=[192,192,192])

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.10)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-3)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=0.8)

    ap.add_argument("--amp-dtype", choices=["bf16","fp16"], default="bf16")
    ap.add_argument("--no-amp", action="store_true")

    ap.add_argument("--pos-ratio", type=float, default=0.70)
    ap.add_argument("--bg-hard-prob", type=float, default=0.85)

    ap.add_argument("--ram-cache-items", type=int, default=0)
    ap.add_argument("--center-jitter-prob", type=float, default=0.75)
    ap.add_argument("--center-jitter-frac", type=float, default=0.30)

    ap.add_argument("--online-light-aug", action="store_true")
    ap.add_argument("--no-online-light-aug", action="store_true")

    # GPU geom aug
    ap.add_argument("--rot-prob", type=float, default=0.05)
    ap.add_argument("--rot-deg", type=float, default=8.0)
    ap.add_argument("--elastic-prob", type=float, default=0.02)
    ap.add_argument("--elastic-alpha", type=float, default=1.5)
    ap.add_argument("--elastic-coarse", type=int, default=6)
    ap.add_argument("--elastic-every", type=int, default=4)

    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--gn-groups", type=int, default=16)
    ap.add_argument("--mamba-layers", type=int, default=4)
    ap.add_argument("--mamba-axes", type=str, default="dhw,hwd,wdh")
    ap.add_argument("--snake-stages", type=str, default="u3,u2,u1")
    ap.add_argument("--snake-k", type=int, default=3)
    ap.add_argument("--checkpoint-decoder", action="store_true")
    ap.add_argument("--no-checkpoint-bottleneck", action="store_true")
    ap.add_argument("--coord-inject-levels", type=str, default="bottleneck,enc2,enc3")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--mamba-dropout", type=float, default=0.0)

    ap.add_argument("--ds-decay-start", type=int, default=80)
    ap.add_argument("--ds-decay-end", type=int, default=200)

    # validation
    ap.add_argument("--val-full", action="store_true")
    ap.add_argument("--no-val-full", action="store_true")
    ap.add_argument("--val-full-max", type=int, default=16)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--val-async", action="store_true")

    ap.add_argument("--val-overlap-early", type=float, default=0.45)
    ap.add_argument("--val-overlap-late", type=float, default=0.65)
    ap.add_argument("--val-overlap-switch", type=int, default=60)

    ap.add_argument("--val-halo", type=int, default=16)
    ap.add_argument("--blend-window", choices=["hann","gaussian"], default="hann")
    ap.add_argument("--val-rm-small", action="store_true")
    ap.add_argument("--val-min-cc-vox", type=int, default=0)
    ap.add_argument("--val-min-cc-rel", type=float, default=0.001)

    ap.add_argument("--hd95-start", type=int, default=10)

    ap.add_argument("--select-w-dice", type=float, default=0.60)
    ap.add_argument("--select-w-cldice", type=float, default=0.40)
    ap.add_argument("--select-cl-iters", type=int, default=25)

    # EMA
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)

    # compile
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--init-ckpt", type=Path, required=True)

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    seed_everything(int(args.seed))
    enable_perf_flags(tf32=True, cudnn_benchmark=True)

    dev_count = torch.cuda.device_count()
    print(f"[info] torch.cuda.device_count()={dev_count}", flush=True)

    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{int(args.gpu)}")

    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        sampling_db_dir=args.sampling_db_dir,
        gpu=int(args.gpu),
        val_gpu=int(args.val_gpu),
        epochs=int(args.epochs),
        epoch_size=int(args.epoch_size),
        batch_size=int(args.batch_size),
        accum_steps=max(1,int(args.accum_steps)),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        patch_size=tuple(int(x) for x in args.patch_size),
        val_patch_size=tuple(int(x) for x in args.val_patch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_epochs=int(args.warmup_epochs),
        grad_clip=float(args.grad_clip),
        amp=(not args.no_amp),
        amp_dtype=str(args.amp_dtype),
        pos_ratio=float(args.pos_ratio),
        bg_hard_prob=float(args.bg_hard_prob),
        ram_cache_items=int(args.ram_cache_items),
        center_jitter_prob=float(args.center_jitter_prob),
        center_jitter_frac=float(args.center_jitter_frac),
        online_light_aug=(False if args.no_online_light_aug else True if args.online_light_aug else True),
        rot_prob=float(args.rot_prob),
        rot_deg=float(args.rot_deg),
        elastic_prob=float(args.elastic_prob),
        elastic_alpha=float(args.elastic_alpha),
        elastic_coarse=int(args.elastic_coarse),
        elastic_every=max(1,int(args.elastic_every)),
        base_ch=int(args.base_ch),
        gn_groups=int(args.gn_groups),
        mamba_layers=int(args.mamba_layers),
        mamba_axes=tuple(_parse_mamba_axes(args.mamba_axes)),
        snake_stages=tuple(sorted(list(_parse_stages(args.snake_stages)))),
        snake_k=int(args.snake_k),
        checkpoint_bottleneck=(not args.no_checkpoint_bottleneck),
        checkpoint_decoder=bool(args.checkpoint_decoder),
        coord_inject_levels=str(args.coord_inject_levels),
        dropout=float(args.dropout),
        mamba_dropout=float(args.mamba_dropout),
        ds_decay_start=int(args.ds_decay_start),
        ds_decay_end=int(args.ds_decay_end),
        val_full=(False if args.no_val_full else True if args.val_full else True),
        val_full_max=int(args.val_full_max),
        val_interval=max(1,int(args.val_interval)),
        val_async=bool(args.val_async),
        val_overlap_early=float(args.val_overlap_early),
        val_overlap_late=float(args.val_overlap_late),
        val_overlap_switch=int(args.val_overlap_switch),
        val_halo=int(args.val_halo),
        blend_window=str(args.blend_window),
        val_rm_small=bool(args.val_rm_small),
        val_min_cc_vox=int(args.val_min_cc_vox),
        val_min_cc_rel=float(args.val_min_cc_rel),
        hd95_start=int(args.hd95_start),
        select_w_dice=float(args.select_w_dice),
        select_w_cldice=float(args.select_w_cldice),
        select_cl_iters=int(args.select_cl_iters),
        use_ema=(False if args.no_ema else True if args.use_ema else True),
        ema_decay=float(args.ema_decay),
        compile=bool(args.compile),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json_dump_safe(asdict(cfg), indent=2))

    # Data
    items = list_cases_from_cache(cfg.cache_dir)
    train_items, val_items = split_train_val_grouped(items, val_ratio=cfg.val_ratio, seed=cfg.seed)

    train_ds = CachedPatchDataset(
        train_items, cfg.cache_dir, cfg.sampling_db_dir, cfg.patch_size, True,
        cfg.pos_ratio, cfg.bg_hard_prob,
        cfg.seed, cfg.ram_cache_items,
        cfg.center_jitter_prob, cfg.center_jitter_frac,
        online_light_aug=cfg.online_light_aug,
        prefer_sampling_base_uid=True,
        val_force_fg=False,
    )
    val_ds = CachedPatchDataset(
        val_items, cfg.cache_dir, cfg.sampling_db_dir, cfg.patch_size, False,
        0.0, 0.0,
        cfg.seed, cfg.ram_cache_items,
        0.0, 0.0,
        online_light_aug=False,
        prefer_sampling_base_uid=True,
        val_force_fg=True,   # important
    )

    sampler = torch.utils.data.RandomSampler(
        train_ds, replacement=True, num_samples=int(cfg.epoch_size * cfg.batch_size)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(cfg.num_workers>0),
        prefetch_factor=2 if cfg.num_workers>0 else None
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(0, cfg.num_workers//2), pin_memory=True, drop_last=False,
        persistent_workers=(cfg.num_workers>0),
        prefetch_factor=2 if cfg.num_workers>0 else None
    )

    # Model
    model = MambaSnakeUNet3D(
        in_ch=1, num_classes=2, base=cfg.base_ch, gn_groups=cfg.gn_groups,
        mamba_layers=cfg.mamba_layers, mamba_axes=list(cfg.mamba_axes),
        snake_stages=cfg.snake_stages, snake_k=cfg.snake_k,
        checkpoint_bottleneck=cfg.checkpoint_bottleneck,
        checkpoint_decoder=cfg.checkpoint_decoder,
        coord_inject_levels=cfg.coord_inject_levels,
        dropout=cfg.dropout, mamba_dropout=cfg.mamba_dropout,
    ).to(device)

    # Warm-start
    ckpt_obj = torch.load(args.init_ckpt, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        state = ckpt_obj["model"]
    elif isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        state = ckpt_obj
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[init-ckpt] loaded: {args.init_ckpt}", flush=True)
    if missing:
        print(f"[init-ckpt] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[init-ckpt] unexpected keys: {len(unexpected)}", flush=True)

    # Optional torch.compile (A100 usually benefits, but can be brittle)
    if cfg.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[info] torch.compile enabled", flush=True)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = None
    if cfg.amp and cfg.amp_dtype == "fp16":
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    loss_obj = TubularLoss(ce_weight_fg=30.0).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    total_steps = cfg.epochs * len(train_loader)
    global_step = 0
    best_score = -1e9
    best_epoch = -1
    history: List[Dict[str,Any]] = []

    # async val state
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    val_proc = None
    val_queue = None
    val_ckpt_path = None
    val_epoch_inflight = None

    def collect_async_if_ready():
        nonlocal val_proc, val_queue, val_ckpt_path, val_epoch_inflight, best_score, best_epoch
        if val_proc is None:
            return None
        if val_proc.is_alive():
            return None

        try:
            res = val_queue.get_nowait()
        except Exception:
            res = {"epoch_id": val_epoch_inflight, "error": "val worker finished but queue empty"}

        val_proc.join(timeout=1.0)

        # read snapshot BEFORE deleting
        snap_model = None
        if val_ckpt_path is not None and Path(val_ckpt_path).exists():
            snap_model = torch.load(val_ckpt_path, map_location="cpu")["model"]

        # cleanup snapshot file
        try:
            if val_ckpt_path is not None and Path(val_ckpt_path).exists():
                Path(val_ckpt_path).unlink()
        except Exception:
            pass

        val_proc = None
        val_queue = None
        val_ckpt_path = None
        val_epoch_inflight = None

        if "error" in res:
            print(f"[val-async-error] {res['error']}", flush=True)
            return res

        # Update best using full-val
        d = float(res.get("val_full_fg_dice", float("nan")))
        c = float(res.get("val_full_fg_cldice", float("nan")))
        if not (math.isnan(d) or math.isnan(c)):
            score = cfg.select_w_dice*d + cfg.select_w_cldice*c
            if score > best_score and snap_model is not None:
                best_score = score
                best_epoch = int(res.get("epoch_id", -1))
                torch.save({"epoch": best_epoch, "model": snap_model, "best_score": best_score}, cfg.out_dir / "best.pt")

        return res

    def launch_async_full_val(epoch: int):
        nonlocal val_proc, val_queue, val_ckpt_path, val_epoch_inflight
        if not cfg.val_full or not cfg.val_async:
            return
        if cfg.val_gpu == cfg.gpu:
            return  # async needs a different GPU
        if not (epoch == 1 or epoch == cfg.epochs or (epoch % cfg.val_interval == 0)):
            return
        if val_proc is not None and val_proc.is_alive():
            return

        ckpt = cfg.out_dir / f"val_ep{epoch:03d}.pt"
        # snapshot current model weights
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)

        payload = {
            "cfg": json.loads(json_dump_safe(asdict(cfg), indent=0)),
            "ckpt_path": str(ckpt),
            "val_items": val_items,
            "epoch_id": int(epoch),
        }
        q = ctx.Queue()
        p = ctx.Process(target=validate_full_worker, args=(payload, q))
        p.start()

        val_proc = p
        val_queue = q
        val_ckpt_path = str(ckpt)
        val_epoch_inflight = int(epoch)

    # Training loop
    for epoch in range(1, cfg.epochs+1):
        # collect async result if ready
        async_res = collect_async_if_ready()
        if async_res is not None:
            # attach to history
            ep = int(async_res.get("epoch_id", -1))
            for r in history:
                if int(r.get("epoch", -999)) == ep:
                    if "error" in async_res:
                        r["val_full_error"] = str(async_res["error"])
                    else:
                        r.update({k: async_res[k] for k in async_res.keys() if k != "epoch_id"})
                    break
            (cfg.out_dir / "history.json").write_text(json_dump_safe(history, indent=2))

        # loss schedule
        w_focal, w_cl, w_bnd, cl_iters, ce_w = ft_loss_schedule(epoch, cfg.epochs)
        loss_obj.set_weights(w_focal=w_focal, w_cldice=w_cl, w_bnd=w_bnd, ce_wfg=ce_w, cl_iters=cl_iters)

        t0 = time.time()
        tr_loss, global_step = train_one_epoch(model, train_loader, opt, scaler, loss_obj, cfg, epoch, global_step, total_steps, ema=ema)
        dt = time.time() - t0

        # patchwise val on EMA weights (usually better)
        if ema is not None:
            backup = ema.apply_to(model)
            val_metrics = validate_patchwise(model, val_loader, loss_obj, cfg)
            ema.restore(model, backup)
        else:
            val_metrics = validate_patchwise(model, val_loader, loss_obj, cfg)

        # save last
        torch.save({"epoch": epoch, "model": model.state_dict()}, cfg.out_dir / "last.pt")

        # full-volume val (sync) if not async, and only at interval
        full_metrics = {"val_full_fg_dice": float("nan"), "val_full_fg_cldice": float("nan"), "val_full_hd95": float("nan")}
        if cfg.val_full and (not cfg.val_async) and (epoch == 1 or epoch == cfg.epochs or (epoch % cfg.val_interval == 0)):
            # evaluate EMA weights if enabled
            if ema is not None:
                backup = ema.apply_to(model)
                full_metrics = validate_full_volume(model, val_items, cfg, epoch_id=epoch)
                ema.restore(model, backup)
            else:
                full_metrics = validate_full_volume(model, val_items, cfg, epoch_id=epoch)

        # scoring: prefer full metrics when available
        d_full = float(full_metrics.get("val_full_fg_dice", float("nan")))
        c_full = float(full_metrics.get("val_full_fg_cldice", float("nan")))
        if not (math.isnan(d_full) or math.isnan(c_full)):
            score = cfg.select_w_dice*d_full + cfg.select_w_cldice*c_full
            score_source = "full"
        else:
            score = cfg.select_w_dice*val_metrics["val_fg_dice"] + cfg.select_w_cldice*val_metrics["val_fg_cldice"]
            score_source = "patch"

        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            torch.save({"epoch": best_epoch, "model": model.state_dict(), "best_score": best_score}, cfg.out_dir / "best.pt")

        rec = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "time_sec": float(dt),
            "lr": float(opt.param_groups[0]["lr"]),
            "w_focal": float(w_focal),
            "w_cldice": float(w_cl),
            "w_bnd": float(w_bnd),
            "cldice_iters": int(cl_iters),
            "ce_weight_fg": float(ce_w),
            "val_loss": float(val_metrics["val_loss"]),
            "val_fg_dice": float(val_metrics["val_fg_dice"]),
            "val_fg_cldice": float(val_metrics["val_fg_cldice"]),
            "val_full_fg_dice": float(full_metrics.get("val_full_fg_dice", float("nan"))),
            "val_full_fg_cldice": float(full_metrics.get("val_full_fg_cldice", float("nan"))),
            "val_full_hd95": float(full_metrics.get("val_full_hd95", float("nan"))),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "best_source": score_source,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        (cfg.out_dir / "history.json").write_text(json_dump_safe(history, indent=2))

        # if async enabled and separate val GPU exists, launch it
        launch_async_full_val(epoch)

    # Final async drain
    for _ in range(60):
        res = collect_async_if_ready()
        if res is None:
            break
        time.sleep(0.2)

    print(f"Done. Best score={best_score:.4f} (epoch {best_epoch}) -> {cfg.out_dir/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
