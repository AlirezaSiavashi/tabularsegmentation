import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from tqdm import tqdm

from vmamba3d import (
    PretrainedVMamba3DNavigator,
    inflate_vmamba_tiny_to_3d,
    get_pretrained_nav_param_groups,
)

# Sparse encoder (optional, requires spconv-cu121)
try:
    from sparse_encoder import SparseResEncoder, SPCONV_AVAILABLE
except ImportError:
    SPCONV_AVAILABLE = False
    SparseResEncoder = None


# -----------------------------------------------------------------------------
# Repro / perf / safety
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda_for_a100(
    safe_cudnn: bool,
    allow_tf32: bool = True,
    enable_cudnn: bool = False
):
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    # Default SAFE: cuDNN OFF (prevents giant workspace allocations with 3D + checkpoint recompute)
    torch.backends.cudnn.enabled = bool(enable_cudnn)

    if not enable_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        if safe_cudnn:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _autocast_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
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


def collate_bs1(batch):
    return batch[0]


def log_gpu_mem(prefix: str = ""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024 ** 3)
        r = torch.cuda.memory_reserved() / (1024 ** 3)
        m = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[mem]{prefix} alloc={a:.2f}GB reserved={r:.2f}GB max_alloc={m:.2f}GB", flush=True)


# -----------------------------------------------------------------------------
# cases.json parsing
# -----------------------------------------------------------------------------
def read_cases_json(cache_dir: Path) -> Dict[str, Any]:
    p = Path(cache_dir) / "cases.json"
    if not p.exists():
        raise RuntimeError(f"cases.json not found: {p}")
    js = json.loads(p.read_text())
    for k in ("uids", "images", "labels"):
        if k not in js:
            raise RuntimeError(f"cases.json must contain key '{k}'")
    return js


def list_cases_from_cache(cache_dir: Path) -> List[Dict[str, str]]:
    js = read_cases_json(cache_dir)
    uids = list(js["uids"])
    images = js["images"]
    labels = js["labels"]
    if not isinstance(images, dict) or not isinstance(labels, dict):
        raise RuntimeError("cases.json keys 'images'/'labels' must be dict uid->path")

    items: List[Dict[str, str]] = []
    for uid in uids:
        if uid not in images or uid not in labels:
            raise RuntimeError(f"cases.json missing image/label for uid={uid}")
        items.append({"uid": uid, "image_pt": images[uid], "label_pt": labels[uid]})
    if not items:
        raise RuntimeError("No cached items found.")
    return items


def _resolve_cached_path(uid: str, p: str, cache_dir: Path, kind: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)
    cache_dir = Path(cache_dir)
    if kind == "image":
        alt = cache_dir / "images" / f"{uid}.pt"
    else:
        alt = cache_dir / "labels" / f"{uid}.pt"
    return str(alt)


def _uid_base(uid: str) -> str:
    # robust aug suffix stripping (supports _aug12, -aug12, __aug12, etc.)
    m = re.match(r"^(.*?)(?:[_\-]{1,2}aug\d+)$", uid)
    if m and m.group(1):
        return m.group(1)
    m = re.match(r"^(.*?)(?:[_\-]{1,2}aug\d*|_aug\d*|-aug\d*|__aug\d*)$", uid)
    if m and m.group(1):
        return m.group(1)
    return uid


def _is_aug(uid: str) -> bool:
    return _uid_base(uid) != uid


def split_train_val_baseuids(
    items: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
    val_original_only: bool = True,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for it in items:
        b = _uid_base(it["uid"])
        groups.setdefault(b, []).append(it)

    base_uids = list(groups.keys())
    rng = np.random.default_rng(int(seed))
    rng.shuffle(base_uids)

    if float(val_ratio) <= 0.0:
        # fold_all: use ALL data for training, no validation holdout
        return list(items), []
    n_val = max(1, int(len(base_uids) * float(val_ratio)))
    val_bases = set(base_uids[:n_val])

    train_items: List[Dict[str, str]] = []
    val_items: List[Dict[str, str]] = []

    for b, lst in groups.items():
        if b in val_bases:
            if val_original_only:
                orig = [x for x in lst if x["uid"] == b]
                if not orig:
                    orig = [x for x in lst if not _is_aug(x["uid"])]
                if not orig:
                    orig = [lst[0]]
                val_items.extend(orig)
            else:
                val_items.extend(lst)
        else:
            train_items.extend(lst)

    return train_items, val_items


# -----------------------------------------------------------------------------
# Tiny RAM LRU cache
# -----------------------------------------------------------------------------
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


def load_sampling_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {
        "fg": z.get("fg", np.zeros((0, 3), np.int32)).astype(np.int32),
        "hardneg": z.get("hardneg", np.zeros((0, 3), np.int32)).astype(np.int32),
        "bg_boundary": z.get("bg_boundary", np.zeros((0, 3), np.int32)).astype(np.int32),
        "bg_easy": z.get("bg_easy", np.zeros((0, 3), np.int32)).astype(np.int32),
        "vessel_boundary": z.get("vessel_boundary", np.zeros((0, 3), np.int32)).astype(np.int32),
    }


class CachedVolumeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[Dict[str, str]],
        cache_dir: Path,
        sampling_db_dir: Optional[Path],
        ram_cache_items: int,
        input_clamp_01: bool,
        prefer_sampling_base_uid: bool = True,
        pool_cache_items: int = 1,
        compute_skeleton: bool = False,
    ):
        self.items = items
        self.cache_dir = Path(cache_dir)
        self.sampling_db_dir = Path(sampling_db_dir) if sampling_db_dir is not None else None
        self.input_clamp_01 = bool(input_clamp_01)
        self.prefer_sampling_base_uid = bool(prefer_sampling_base_uid)
        self.compute_skeleton = bool(compute_skeleton)

        self.cache_img = _LRUVolCache(int(ram_cache_items))
        self.cache_lab = _LRUVolCache(int(ram_cache_items))
        self.cache_skel = _LRUVolCache(int(ram_cache_items)) if compute_skeleton else None
        # pools can be big; keep this very small unless you really want more
        self.cache_pool = _LRUVolCache(int(pool_cache_items))

    def __len__(self) -> int:
        return len(self.items)

    def _sampling_uid(self, uid: str) -> str:
        return _uid_base(uid) if self.prefer_sampling_base_uid else uid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[int(idx)]
        uid = it["uid"]

        img = self.cache_img.get(uid)
        lab = self.cache_lab.get(uid)
        skel = self.cache_skel.get(uid) if self.cache_skel is not None else None
        if img is None or lab is None:
            ip = _resolve_cached_path(uid, it["image_pt"], self.cache_dir, "image")
            lp = _resolve_cached_path(uid, it["label_pt"], self.cache_dir, "label")
            img = torch.load(ip, map_location="cpu")
            lab = torch.load(lp, map_location="cpu")

            if img.ndim == 3:
                img = img.unsqueeze(0)  # [1,D,H,W]
            img = img.float()
            if self.input_clamp_01:
                img = img.clamp(0.0, 1.0)

            if lab.ndim == 4 and lab.shape[0] == 1:
                lab = lab[0]
            lab = (lab > 0).to(torch.uint8)  # [D,H,W]

            self.cache_img.put(uid, img)
            self.cache_lab.put(uid, lab)

            # Precompute TRUE skeleton (SOTA's key innovation)
            if self.compute_skeleton:
                skel = precompute_skeleton_3d(lab, do_tube=True)
                self.cache_skel.put(uid, skel)

        if self.compute_skeleton and skel is None:
            skel = precompute_skeleton_3d(lab, do_tube=True)
            self.cache_skel.put(uid, skel)

        pools = self.cache_pool.get(uid)
        if pools is None:
            pools = {
                "fg": np.zeros((0, 3), np.int32),
                "hardneg": np.zeros((0, 3), np.int32),
                "bg_boundary": np.zeros((0, 3), np.int32),
                "bg_easy": np.zeros((0, 3), np.int32),
                "vessel_boundary": np.zeros((0, 3), np.int32),
            }
            if self.sampling_db_dir is not None:
                suid = self._sampling_uid(uid)
                p = self.sampling_db_dir / f"{suid}.npz"
                if p.exists():
                    pools = load_sampling_npz(p)
            self.cache_pool.put(uid, pools)

        out = {"uid": uid, "image": img, "label": lab, "pools": pools}
        if self.compute_skeleton and skel is not None:
            out["skeleton"] = skel
        return out


# -----------------------------------------------------------------------------
# Patch utils
# -----------------------------------------------------------------------------
def _pad_to_min_shape(vol: torch.Tensor, target_dhw: Tuple[int, int, int], pad_value: float = 0.0) -> torch.Tensor:
    td, th, tw = target_dhw
    if vol.ndim == 4:
        _, D, H, W = vol.shape
        has_c = True
    elif vol.ndim == 3:
        D, H, W = vol.shape
        has_c = False
    else:
        raise RuntimeError(f"Unexpected vol ndim={vol.ndim}")

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
    return F.pad(vol.unsqueeze(0), pad, mode="constant", value=float(pad_value)).squeeze(0)


def crop_patch_from_start(
    img: torch.Tensor,   # [1,D,H,W]
    lab: torch.Tensor,   # [D,H,W]
    start_zyx: Tuple[int, int, int],
    patch_dhw: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    pd, ph, pw = patch_dhw
    z0, y0, x0 = start_zyx
    img_p = img[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    lab_p = lab[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return img_p.contiguous(), lab_p.contiguous()


def centers_to_starts_aligned(
    centers_zyx: np.ndarray,
    vol_dhw: Tuple[int, int, int],
    patch_dhw: Tuple[int, int, int],
    align: int,
) -> np.ndarray:
    D, H, W = vol_dhw
    pd, ph, pw = patch_dhw
    a = max(1, int(align))

    def _max_start(dim, p):
        if dim <= p:
            return 0
        return ((dim - p) // a) * a

    mz = _max_start(D, pd)
    my = _max_start(H, ph)
    mx = _max_start(W, pw)

    starts = np.zeros_like(centers_zyx, dtype=np.int32)
    for i in range(centers_zyx.shape[0]):
        cz, cy, cx = [int(v) for v in centers_zyx[i]]
        z0 = max(0, cz - pd // 2)
        y0 = max(0, cy - ph // 2)
        x0 = max(0, cx - pw // 2)

        z0 = (z0 // a) * a
        y0 = (y0 // a) * a
        x0 = (x0 // a) * a

        z0 = min(z0, mz)
        y0 = min(y0, my)
        x0 = min(x0, mx)
        starts[i] = (z0, y0, x0)
    return starts


# -----------------------------------------------------------------------------
# Sampling strategy
# -----------------------------------------------------------------------------
def choose_centers(
    pools: Dict[str, np.ndarray],
    lab: torch.Tensor,  # [D,H,W] uint8
    k: int,
    rng: np.random.Generator,
    p_fg: float,
    p_bg_boundary: float,
    p_bg_hard: float,
    p_bg_easy: float,
    p_vessel_boundary: float = 0.0,
) -> np.ndarray:
    # vessel_boundary is carved out of p_fg budget to avoid changing total
    p_fg_adj = max(0.0, p_fg - p_vessel_boundary)
    probs = np.array([p_fg_adj, p_vessel_boundary, p_bg_boundary, p_bg_hard, p_bg_easy], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = float(probs.sum())
    probs = (probs / s) if s > 0 else np.array([1.0, 0, 0, 0, 0], dtype=np.float64)

    fg = pools.get("fg", np.zeros((0, 3), np.int32))
    vb = pools.get("vessel_boundary", np.zeros((0, 3), np.int32))
    bgb = pools.get("bg_boundary", np.zeros((0, 3), np.int32))
    hard = pools.get("hardneg", np.zeros((0, 3), np.int32))
    bge = pools.get("bg_easy", np.zeros((0, 3), np.int32))

    D, H, W = lab.shape

    fg_fallback = None
    if fg is None or fg.shape[0] == 0:
        nz = torch.nonzero(lab > 0, as_tuple=False)
        if nz.numel() > 0:
            fg_fallback = nz.cpu().numpy().astype(np.int32)

    def _pick(arr: Optional[np.ndarray]) -> Tuple[int, int, int]:
        if arr is not None and arr.shape[0] > 0:
            z, y, x = arr[int(rng.integers(0, arr.shape[0]))]
            return int(z), int(y), int(x)
        return int(rng.integers(0, D)), int(rng.integers(0, H)), int(rng.integers(0, W))

    out = np.zeros((int(k), 3), np.int32)
    cum = np.cumsum(probs)
    for i in range(int(k)):
        r = float(rng.random())
        if r < cum[0]:
            c = _pick(fg if fg is not None and fg.shape[0] > 0 else fg_fallback)
        elif r < cum[1]:
            # vessel_boundary: fg voxel near patch-edge of vessel bbox
            # fall back to regular fg if pool empty
            c = _pick(vb if vb is not None and vb.shape[0] > 0 else (fg if fg is not None and fg.shape[0] > 0 else fg_fallback))
        elif r < cum[2]:
            c = _pick(bgb)
        elif r < cum[3]:
            c = _pick(hard)
        else:
            c = _pick(bge)
        out[i] = c
    return out


# -----------------------------------------------------------------------------
# Cached grids for grid_sample
# -----------------------------------------------------------------------------
_GRID_CACHE: Dict[Tuple[int, int, int, str], torch.Tensor] = {}


def _get_grid_bdhw3(B: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    key = (int(D), int(H), int(W), str(device))
    g = _GRID_CACHE.get(key, None)
    if g is None or g.device != device:
        zz = torch.linspace(-1, 1, D, device=device, dtype=torch.float32)
        yy = torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
        xx = torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
        z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
        g = torch.stack([x, y, z], dim=-1).unsqueeze(0).contiguous()  # [1,D,H,W,3]
        _GRID_CACHE[key] = g
    return g if B == 1 else g.expand(B, -1, -1, -1, -1)


def _rot_mats_xyz(ax: torch.Tensor, ay: torch.Tensor, az: torch.Tensor) -> torch.Tensor:
    device = ax.device
    dtype = torch.float32
    one = torch.ones((ax.shape[0],), device=device, dtype=dtype)
    zero = torch.zeros((ax.shape[0],), device=device, dtype=dtype)

    cx, sx = torch.cos(ax).to(dtype), torch.sin(ax).to(dtype)
    cy, sy = torch.cos(ay).to(dtype), torch.sin(ay).to(dtype)
    cz, sz = torch.cos(az).to(dtype), torch.sin(az).to(dtype)

    Rx = torch.stack([
        torch.stack([one, zero, zero], dim=-1),
        torch.stack([zero, cx, -sx], dim=-1),
        torch.stack([zero, sx, cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, zero, sy], dim=-1),
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-sy, zero, cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, zero], dim=-1),
        torch.stack([sz, cz, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)

    return (Rz @ Ry @ Rx).to(dtype)


def random_flip_aligned(img, lab, ctx, p: float):
    if p <= 0:
        return img, lab, ctx
    device = img.device
    B = img.shape[0]
    for axis_img, axis_lab, axis_ctx in [(2, 1, 2), (3, 2, 3), (4, 3, 4)]:
        m = (torch.rand((B,), device=device) < float(p))
        if m.any():
            sel = torch.arange(B, device=device)[m]
            img[sel] = torch.flip(img[sel], dims=(axis_img,))
            lab[sel] = torch.flip(lab[sel], dims=(axis_lab,))
            ctx[sel] = torch.flip(ctx[sel], dims=(axis_ctx,))
    return img, lab, ctx


def aligned_geom_aug(img, lab, ctx, rot_prob, rot_deg, elastic_prob, elastic_alpha, elastic_coarse):
    device = img.device
    B, _, D, H, W = img.shape
    _, _, d, h, w = ctx.shape

    do_rot = (rot_prob > 0) & (torch.rand((B,), device=device) < float(rot_prob))
    do_ela = (elastic_prob > 0) & (torch.rand((B,), device=device) < float(elastic_prob))
    if not (do_rot.any() or do_ela.any()):
        return img, lab, ctx

    grid_img = _get_grid_bdhw3(B, D, H, W, device).clone()
    grid_ctx = _get_grid_bdhw3(B, d, h, w, device).clone()

    if do_rot.any():
        deg = float(rot_deg) * (math.pi / 180.0)
        ax = ((torch.rand((B,), device=device) * 2 - 1) * deg) * do_rot.float()
        ay = ((torch.rand((B,), device=device) * 2 - 1) * deg) * do_rot.float()
        az = ((torch.rand((B,), device=device) * 2 - 1) * deg) * do_rot.float()
        R = _rot_mats_xyz(ax, ay, az)

        g1 = grid_img.view(B, -1, 3)
        g1 = torch.bmm(g1, R.transpose(1, 2))
        grid_img = g1.view(B, D, H, W, 3)

        g2 = grid_ctx.view(B, -1, 3)
        g2 = torch.bmm(g2, R.transpose(1, 2))
        grid_ctx = g2.view(B, d, h, w, 3)

    if do_ela.any():
        coarse = int(max(2, elastic_coarse))
        disp_c = torch.randn((B, 3, coarse, coarse, coarse), device=device, dtype=torch.float32)
        disp_img_vox = F.interpolate(disp_c, size=(D, H, W), mode="trilinear", align_corners=False)
        disp_ctx_vox = F.interpolate(disp_c, size=(d, h, w), mode="trilinear", align_corners=False)

        m = do_ela.float().view(B, 1, 1, 1, 1)
        disp_img_vox = disp_img_vox * m
        disp_ctx_vox = disp_ctx_vox * m

        sx_img = (2.0 / max(1, W - 1)) * float(elastic_alpha)
        sy_img = (2.0 / max(1, H - 1)) * float(elastic_alpha)
        sz_img = (2.0 / max(1, D - 1)) * float(elastic_alpha)
        disp_img = disp_img_vox.permute(0, 2, 3, 4, 1).contiguous()
        disp_img[..., 0] *= sx_img
        disp_img[..., 1] *= sy_img
        disp_img[..., 2] *= sz_img

        sx_ctx = (2.0 / max(1, w - 1)) * float(elastic_alpha)
        sy_ctx = (2.0 / max(1, h - 1)) * float(elastic_alpha)
        sz_ctx = (2.0 / max(1, d - 1)) * float(elastic_alpha)
        disp_ctx = disp_ctx_vox.permute(0, 2, 3, 4, 1).contiguous()
        disp_ctx[..., 0] *= sx_ctx
        disp_ctx[..., 1] *= sy_ctx
        disp_ctx[..., 2] *= sz_ctx

        grid_img = grid_img + disp_img
        grid_ctx = grid_ctx + disp_ctx

    lab_in = lab.unsqueeze(1).float()
    img_aug = F.grid_sample(img.float(), grid_img, mode="bilinear", padding_mode="border", align_corners=False).to(img.dtype)
    lab_aug = F.grid_sample(lab_in, grid_img, mode="nearest", padding_mode="border", align_corners=False)[:, 0].long()
    ctx_aug = F.grid_sample(ctx.float(), grid_ctx, mode="bilinear", padding_mode="border", align_corners=False).to(ctx.dtype)
    return img_aug, lab_aug, ctx_aug


@torch.no_grad()
def gpu_intensity_aug(img: torch.Tensor, p: float):
    if p <= 0 or torch.rand((), device=img.device) > float(p):
        return img
    gamma = 2.0 ** (torch.rand((), device=img.device) * 0.4 - 0.2)
    img = torch.clamp(img, 0.0, 1.0) ** gamma
    if torch.rand((), device=img.device) < 0.5:
        img = torch.clamp(img + torch.randn_like(img) * 0.015, 0.0, 1.0)
    if torch.rand((), device=img.device) < 0.3:
        a = 1.0 + (torch.rand((), device=img.device) * 0.3 - 0.15)
        b = (torch.rand((), device=img.device) * 0.04 - 0.02)
        img = torch.clamp(img * a + b, 0.0, 1.0)
    return img


@torch.no_grad()
def gpu_intensity_aug_strong(img: torch.Tensor, p: float):
    """SOTA-level intensity augmentation matching 1st place nnUNet pipeline.

    Includes: Gamma, GaussianNoise, Contrast, Brightness, Sharpening,
    SimulateLowResolution (per-axis), BrightnessGradient, LocalGamma, Inversion.
    All on GPU for speed.
    """
    if p <= 0 or torch.rand((), device=img.device) > float(p):
        return img
    dev = img.device

    # 1. Gamma (p=0.3) — SOTA uses (0.7, 1.5) and also inverted gamma
    if torch.rand((), device=dev) < 0.3:
        gamma = 0.7 + torch.rand((), device=dev) * 0.8  # [0.7, 1.5]
        img = torch.clamp(img, 1e-6, 1.0) ** gamma
    if torch.rand((), device=dev) < 0.15:
        # inverted gamma (apply gamma to 1-img, then invert back)
        gamma = 0.7 + torch.rand((), device=dev) * 0.8
        img = 1.0 - (1.0 - img).clamp(1e-6, 1.0) ** gamma

    # 2. Gaussian noise (p=0.15) — SOTA uses variance (0, 0.1)
    if torch.rand((), device=dev) < 0.15:
        std = torch.rand((), device=dev) * 0.1
        img = torch.clamp(img + torch.randn_like(img) * std, 0.0, 1.0)

    # 3. Gaussian blur (p=0.2) — SOTA uses sigma (0.5, 1.0)
    if torch.rand((), device=dev) < 0.2:
        sigma = 0.5 + torch.rand((), device=dev).item() * 0.5
        ks = max(3, int(sigma * 4) | 1)  # ensure odd
        # Separable 3D blur via 1D convolutions along each axis
        x = torch.arange(ks, device=dev, dtype=torch.float32) - ks // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        B, C, D, H, W = img.shape
        pad = ks // 2
        # z-axis
        kz = kernel.view(1, 1, ks, 1, 1)
        img = F.conv3d(img, kz.expand(C, -1, -1, -1, -1), padding=(pad, 0, 0), groups=C)
        # y-axis
        ky = kernel.view(1, 1, 1, ks, 1)
        img = F.conv3d(img, ky.expand(C, -1, -1, -1, -1), padding=(0, pad, 0), groups=C)
        # x-axis
        kx = kernel.view(1, 1, 1, 1, ks)
        img = F.conv3d(img, kx.expand(C, -1, -1, -1, -1), padding=(0, 0, pad), groups=C)

    # 4. Contrast (multiplicative brightness) (p=0.15) — SOTA uses (0.75, 1.25)
    if torch.rand((), device=dev) < 0.15:
        factor = 0.75 + torch.rand((), device=dev) * 0.5
        mean = img.mean()
        img = torch.clamp((img - mean) * factor + mean, 0.0, 1.0)

    # 5. Brightness shift (p=0.15)
    if torch.rand((), device=dev) < 0.15:
        shift = (torch.rand((), device=dev) - 0.5) * 0.06
        img = torch.clamp(img + shift, 0.0, 1.0)

    # 6. SimulateLowResolution per axis (p=0.25 per axis) — SOTA's key augmentation
    # Downsample then upsample along one or more axes to simulate anisotropic resolution
    for axis in [2, 3, 4]:  # D, H, W
        if torch.rand((), device=dev) < 0.25:
            zoom_factor = 0.1 + torch.rand((), device=dev).item() * 0.8  # [0.1, 0.9]
            sz = list(img.shape)
            orig_size = sz[axis]
            new_size = max(1, int(orig_size * zoom_factor))
            # downsample
            target = list(img.shape[2:])
            target[axis - 2] = new_size
            down = F.interpolate(img, size=target, mode="trilinear", align_corners=False)
            # upsample back
            target[axis - 2] = orig_size
            img = F.interpolate(down, size=target, mode="trilinear", align_corners=False)

    # 7. Sharpening (p=0.2) — SOTA uses this
    if torch.rand((), device=dev) < 0.2:
        # Unsharp mask: sharp = img + alpha*(img - blurred)
        alpha = 0.1 + torch.rand((), device=dev).item() * 0.9  # [0.1, 1.0]
        blurred = F.avg_pool3d(img, kernel_size=3, stride=1, padding=1)
        img = torch.clamp(img + alpha * (img - blurred), 0.0, 1.0)

    # 8. Local Gamma (brightness gradient) (p=0.15) — SOTA uses BrightnessGradient
    if torch.rand((), device=dev) < 0.15:
        B, C, D, H, W = img.shape
        # Create a smooth gradient along a random axis
        grad_axis = int(torch.randint(0, 3, (1,), device=dev).item())
        if grad_axis == 0:
            grad = torch.linspace(0.8, 1.2, D, device=dev).view(1, 1, D, 1, 1)
        elif grad_axis == 1:
            grad = torch.linspace(0.8, 1.2, H, device=dev).view(1, 1, 1, H, 1)
        else:
            grad = torch.linspace(0.8, 1.2, W, device=dev).view(1, 1, 1, 1, W)
        img = torch.clamp(img * grad, 0.0, 1.0)

    return img


def mixup_3d(
    img1: torch.Tensor, lab1: torch.Tensor,
    img2: torch.Tensor, lab2: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp augmentation for 3D volumes.
    Linearly interpolates both image and label volumes.
    For segmentation, labels are also mixed (soft labels).
    """
    if alpha <= 0:
        return img1, lab1.float()
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Ensure lambda >= 0.5 for stability

    # Mix images
    mixed_img = lam * img1 + (1.0 - lam) * img2

    # Mix labels (convert to float for soft mixing)
    lab1_f = lab1.float()
    lab2_f = lab2.float()
    mixed_lab = lam * lab1_f + (1.0 - lam) * lab2_f

    return mixed_img, mixed_lab


def apply_mixup_batch(
    images: torch.Tensor, labels: torch.Tensor,
    prob: float = 0.3, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Apply MixUp to a batch by shuffling and mixing.
    Returns mixed data and whether mixup was applied.
    """
    if prob <= 0 or torch.rand(()).item() > prob:
        return images, labels, False

    B = images.shape[0]
    if B < 2:
        return images, labels, False

    # Shuffle indices
    indices = torch.randperm(B, device=images.device)

    # Sample lambda
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)

    # Mix
    mixed_images = lam * images + (1.0 - lam) * images[indices]
    mixed_labels = lam * labels.float() + (1.0 - lam) * labels[indices].float()

    return mixed_images, mixed_labels, True


# -----------------------------------------------------------------------------
# Losses / Metrics (with your patches)
# -----------------------------------------------------------------------------
# --- Skeleton Recall Loss (from SOTA 1st place solution) ---
# Uses TRUE precomputed skeletons, not soft iterative thinning.
# This is the single most impactful loss change.
def precompute_skeleton_3d(label: torch.Tensor, do_tube: bool = True) -> torch.Tensor:
    """Precompute TRUE 3D skeleton from binary label on CPU.

    Uses skimage.morphology.skeletonize (Lee94 algorithm) — topologically exact.
    Then dilates 2px to create a tube (matches SOTA).

    Args:
        label: [D,H,W] uint8 or bool tensor on CPU
        do_tube: if True, dilate skeleton by 2 voxels (SOTA default)
    Returns:
        skeleton: [D,H,W] uint8 tensor on CPU
    """
    from skimage.morphology import skeletonize, dilation
    seg_np = (label.numpy() > 0).astype(np.uint8)
    if seg_np.sum() == 0:
        return torch.zeros_like(label, dtype=torch.uint8)
    skel = skeletonize(seg_np).astype(np.uint8)
    if do_tube:
        skel = dilation(dilation(skel)).astype(np.uint8)
    # Mask skeleton to only be within the original segmentation
    skel = skel * seg_np
    return torch.from_numpy(skel).to(torch.uint8)


class SkeletonRecallLoss(nn.Module):
    """Skeleton Recall Loss — from RSNA 1st place solution.

    Measures how well the prediction covers the TRUE skeleton of GT.
    recall = (pred * skel_gt).sum() / skel_gt.sum()
    loss = -recall

    This is fundamentally different from soft clDice which uses iterative
    morphological thinning (noisy, approximate). True skeletons via
    skimage.morphology.skeletonize are topologically exact.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, skel_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 2, D, H, W] raw logits
            skel_gt: [B, D, H, W] binary skeleton (precomputed)
        Returns:
            -recall (scalar)
        """
        probs = torch.softmax(logits.float(), dim=1)[:, 1]  # [B, D, H, W]
        skel = skel_gt.float()
        axes = (1, 2, 3)  # spatial dims
        inter = (probs * skel).sum(axes)
        total = skel.sum(axes)
        recall = (inter + self.smooth) / (total.clamp(min=1e-8) + self.smooth)
        return -recall.mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probs = torch.softmax(logits.float(), dim=1)
        fg = probs[:, 1:2]
        tgt = (targets > 0).float().unsqueeze(1)
        dims = (2, 3, 4)
        inter = (fg * tgt).sum(dims)
        den = (fg + tgt).sum(dims)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - handles class imbalance for vessel segmentation."""
    def __init__(self, alpha=0.25, beta=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = float(alpha)  # FP weight
        self.beta = float(beta)    # FN weight (higher = more penalty for FN)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probs = torch.softmax(logits.float(), dim=1)
        p = probs[:, 1:2]
        t = (targets > 0).float().unsqueeze(1)
        dims = (2, 3, 4)
        tp = (p * t).sum(dims)
        fp = (p * (1 - t)).sum(dims)
        fn = ((1 - p) * t).sum(dims)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tv.mean()


# ---- PATCH 2: ConnectivityLoss (REQUIRED) ----
class ConnectivityLoss(nn.Module):
    """
    Penalizes disconnected components by measuring how much morphological closing changes the prediction.
    More gaps => more change after closing => higher penalty.
    """
    def __init__(self, iters: int = 2):
        super().__init__()
        self.iters = int(iters)

    def morphological_close(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iters):
            x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)  # dilate
        for _ in range(self.iters):
            x = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)  # erode
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_closed = self.morphological_close(pred)
        target_closed = self.morphological_close(target)

        pred_gap = F.l1_loss(pred_closed, pred)
        target_gap = F.l1_loss(target_closed, target)

        return F.relu(pred_gap - target_gap)


def _soft_erode(img):
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img):
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img, iters: int):
    img = torch.clamp(img, 0, 1)
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(int(iters)):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return torch.nan_to_num(skel, nan=0.0, posinf=0.0, neginf=0.0)


class SoftclDiceLoss(nn.Module):
    def __init__(self, iters: int = 10, smooth: float = 1e-5):
        super().__init__()
        self.iters = int(iters)
        self.smooth = float(smooth)

    def set_iters(self, iters: int):
        self.iters = int(iters)

    def forward(self, probs_fg, targets_fg):
        p = probs_fg.float()
        t = targets_fg.float()
        sp = _soft_skel(p, self.iters)
        st = _soft_skel(t, self.iters)
        tprec = (sp * t).sum() + self.smooth
        tprec = tprec / (sp.sum() + self.smooth)
        tsens = (st * p).sum() + self.smooth
        tsens = tsens / (st.sum() + self.smooth)
        cl = (2 * tprec * tsens) / (tprec + tsens + self.smooth)
        cl = torch.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)
        return 1.0 - cl


class AuxDiceCELoss(nn.Module):
    def __init__(self, ce_weight_fg: float = 20.0):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.register_buffer("ce_w", torch.tensor([1.0, float(ce_weight_fg)], dtype=torch.float32))

    def forward(self, logits, targets):
        return 0.6 * self.dice(logits, targets) + 0.4 * F.cross_entropy(logits.float(), targets.long(), weight=self.ce_w)


# ---- MainTubularLoss v6: SOTA-aligned (Dice + SkeletonRecall + CE) ----
class MainTubularLoss(nn.Module):
    """
    Loss combining SOTA's Dice+SkeletonRecall+CE with our Tversky+clDice+Connectivity.

    SOTA 1st place uses equal weight: Dice=1, SkeletonRecall=1, CE=1.
    We add skeleton recall ON TOP of existing losses.
    When skeleton GT is available: Dice + Tversky + CE + SkeletonRecall + clDice + Connectivity
    When not available (MixUp, no skeleton): falls back to Dice + Tversky + CE + clDice + Connectivity

    Weights (v6): Tversky=0.30, Dice=0.20, CE=0.20, SkelRecall=0.20, clDice=0.05, Conn=0.05
    """
    def __init__(
        self,
        ce_weight_fg: float = 30.0,
        cl_w: float = 0.15,
        cl_iters: int = 10,
        cl_down: int = 4,
        conn_w: float = 0.10,
        skel_w: float = 0.20,
    ):
        super().__init__()
        self.tv = TverskyLoss(0.25, 0.75)
        self.dice = SoftDiceLoss()
        self.cldice = SoftclDiceLoss(cl_iters)
        self.connectivity = ConnectivityLoss(iters=2)
        self.skel_recall = SkeletonRecallLoss()

        self.register_buffer("ce_w", torch.tensor([1.0, float(ce_weight_fg)], dtype=torch.float32))
        self.cl_w = float(cl_w)
        self.cl_down = int(cl_down)
        self.conn_w = float(conn_w)
        self.skel_w = float(skel_w)

    def set(self, ce_weight_fg: float, cl_w: float, cl_iters: int, cl_down: int):
        self.ce_w[1] = float(ce_weight_fg)
        self.cl_w = float(cl_w)
        self.cldice.set_iters(int(cl_iters))
        self.cl_down = int(cl_down)

    @staticmethod
    def _boundary_mask(targets: torch.Tensor, dilate: int = 1) -> torch.Tensor:
        """Returns a weight map: 1.5 at vessel boundary ring, 1.0 elsewhere."""
        fg = (targets > 0).float().unsqueeze(1)  # [B, 1, D, H, W]
        ks = 2 * dilate + 1
        dilated = F.max_pool3d(fg, kernel_size=ks, stride=1, padding=dilate)
        boundary = (dilated - fg).clamp(0.0, 1.0)
        return (1.0 + 0.5 * boundary).squeeze(1)  # [B, D, H, W]

    def forward(self, logits, targets, soft_labels: bool = False,
                skel_gt: Optional[torch.Tensor] = None):
        """
        Args:
            logits: [B, 2, D, H, W]
            targets: [B, D, H, W] (long for hard, float for soft)
            soft_labels: whether targets are soft (MixUp)
            skel_gt: [B, D, H, W] precomputed skeleton GT (None if unavailable)
        """
        if soft_labels:
            probs = torch.softmax(logits.float(), dim=1)
            tfg = targets.float().unsqueeze(1) if targets.ndim == 4 else targets.float()

            p = probs[:, 1:2]
            dims = (2, 3, 4)
            tp = (p * tfg).sum(dims)
            fp = (p * (1 - tfg)).sum(dims)
            fn = ((1 - p) * tfg).sum(dims)
            tv = (tp + 1e-5) / (tp + 0.25 * fp + 0.75 * fn + 1e-5)
            loss = 0.30 * (1.0 - tv.mean())

            inter = (p * tfg).sum(dims)
            den = (p + tfg).sum(dims)
            dice = (2 * inter + 1e-5) / (den + 1e-5)
            loss = loss + 0.20 * (1.0 - dice.mean())

            tbg = 1.0 - tfg.squeeze(1) if tfg.ndim == 5 else 1.0 - targets.float()
            tfg_sq = tfg.squeeze(1) if tfg.ndim == 5 else targets.float()
            log_probs = F.log_softmax(logits.float(), dim=1)
            soft_ce = -(tbg * log_probs[:, 0] + tfg_sq * log_probs[:, 1]).mean()
            loss = loss + 0.20 * soft_ce
        else:
            # Hard targets: Tversky + Dice + boundary-weighted CE
            loss = 0.30 * self.tv(logits, targets)
            loss = loss + 0.20 * self.dice(logits, targets)
            bw = self._boundary_mask(targets)
            loss = loss + 0.20 * (
                F.cross_entropy(logits.float(), targets.long(),
                                weight=self.ce_w, reduction="none") * bw
            ).mean()

        # *** NEW: Skeleton Recall Loss (SOTA's key innovation) ***
        if skel_gt is not None and self.skel_w > 0 and not soft_labels:
            loss = loss + self.skel_w * self.skel_recall(logits, skel_gt)

        # Existing topology losses (clDice + connectivity)
        if self.cl_w > 0:
            probs = torch.softmax(logits.float(), dim=1)[:, 1:2]
            if soft_labels:
                tfg = targets.float().unsqueeze(1) if targets.ndim == 4 else targets.float()
            else:
                tfg = (targets > 0).float().unsqueeze(1)

            d = max(1, int(self.cl_down))
            if d > 1:
                probs_cl = F.avg_pool3d(probs, kernel_size=d, stride=d)
                tfg_cl = F.avg_pool3d(tfg, kernel_size=d, stride=d) if soft_labels else F.max_pool3d(tfg, kernel_size=d, stride=d)
            else:
                probs_cl, tfg_cl = probs, tfg

            loss = loss + self.cl_w * self.cldice(probs_cl, tfg_cl)

            if self.conn_w > 0:
                loss = loss + self.conn_w * self.connectivity(probs, tfg)

        return torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)


# ---- PATCH 1: improved loss schedule (REQUIRED) ----
def loss_schedule(epoch: int, total_epochs: int):
    """
    Improved schedule:
      cl_w: 0.15 -> 0.30
      ce_w: 35 -> 25
      cl_iters: 10 -> 20
      earlier transitions to higher res
    """
    e = int(epoch)
    T = max(1, int(total_epochs))
    t = (e - 1) / max(1, T - 1)

    cl_w = 0.15 + 0.15 * t
    ce_w = 35.0 - 10.0 * t
    cl_iters = int(round(10 + 10 * t))

    if e < int(0.25 * T):
        cl_down = 4
    elif e < int(0.50 * T):
        cl_down = 2
    else:
        cl_down = 1

    return float(cl_w), float(ce_w), int(cl_iters), int(cl_down)


def deep_sup_weights_dynamic(epoch: int, total_epochs: int, w2: float, w3: float, w4: float):
    e = int(epoch)
    T = max(1, int(total_epochs))
    t0 = int(0.70 * T)
    if e <= t0:
        return float(w2), float(w3), float(w4)
    denom = max(1, (T - t0))
    t = (e - t0) / denom
    s = max(0.0, 1.0 - float(t))
    return float(w2 * s), float(w3 * s), float(w4 * s)


def downsample_label_maxpool(lab: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return lab
    x = (lab > 0).float().unsqueeze(1)
    y = F.max_pool3d(x, kernel_size=factor, stride=factor)
    return (y.squeeze(1) > 0.5).long()


# ---- Navigator supervision losses ----
def nav_connectivity_loss(nav_pred: torch.Tensor, token_label: torch.Tensor) -> torch.Tensor:
    """
    Graph-based vessel connectivity loss on navigator tokens.

    For each pair of 6-connected adjacent tokens that both contain vessels in GT,
    penalize if the predicted connectivity (min of adjacent logits) is low.

    CRITICAL FIX: Only compute BCE on MASKED vessel pairs. The old code computed
    BCE on ALL positions (including non-vessel pairs where both_vessel=0), which
    contributed BCE(sigmoid(0), 0) = 0.693 per non-vessel pair. With many non-vessel
    pairs divided by few vessel pairs, this dominated the loss and injected noise.

    Args:
        nav_pred: [K, 1, td, th, tw] - raw logits from nav_head
        token_label: [K, 1, td, th, tw] - binary GT at token resolution
    Returns:
        Scalar connectivity loss
    """
    loss_sum = torch.tensor(0.0, device=nav_pred.device)
    count = 0

    for dim in (2, 3, 4):  # D, H, W shifts
        sz = nav_pred.size(dim)
        if sz < 2:
            continue
        logit_a = nav_pred.narrow(dim, 0, sz - 1)
        logit_b = nav_pred.narrow(dim, 1, sz - 1)
        gt_a = token_label.narrow(dim, 0, sz - 1)
        gt_b = token_label.narrow(dim, 1, sz - 1)

        # Mask: both neighbors have vessels in GT
        both_vessel = (gt_a * gt_b).squeeze(1)  # [K, ..., ...]
        mask = both_vessel > 0.5
        n_pairs = mask.sum()
        if n_pairs < 1:
            continue

        # Min logit of adjacent pair (connectivity proxy)
        min_logit = torch.min(logit_a, logit_b).squeeze(1)  # [K, ..., ...]

        # ONLY compute BCE on vessel pairs (masked indexing)
        # Target = 1.0 for all vessel pairs (both should predict vessel)
        min_logit_masked = min_logit[mask]  # [n_pairs]
        target_ones = torch.ones_like(min_logit_masked)
        pair_loss = F.binary_cross_entropy_with_logits(
            min_logit_masked, target_ones,
        )  # mean over vessel pairs only
        loss_sum = loss_sum + pair_loss
        count += 1

    return loss_sum / max(count, 1)


class NavigatorLoss(nn.Module):
    """Combined loss for navigator vessel detection head.

    Supports both hard binary labels (from max_pool) and soft fraction labels
    (from avg_pool). Soft labels give the navigator a proper regression target
    that reflects how much vessel is in each token, preventing the 47% FP rate
    caused by binary max_pool exploding positives along vessel tubes.
    """
    def __init__(self, ce_weight_fg: float = 30.0, cldice_iters: int = 5):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.cldice = SoftclDiceLoss(iters=cldice_iters)
        self.register_buffer("ce_w", torch.tensor([1.0, float(ce_weight_fg)], dtype=torch.float32))

    def forward(
        self,
        nav_pred: torch.Tensor,      # [K,1,td,th,tw] raw logits
        token_label: torch.Tensor,    # [K,1,td,th,tw] soft [0,1] vessel fraction GT
        nav_cl_w: float = 0.10,
        nav_conn_w: float = 0.10,
    ) -> torch.Tensor:
        probs = nav_pred.sigmoid()   # [K,1,td,th,tw]
        t = token_label.float()      # soft [0,1]

        # Detect whether labels are soft (avg_pool) or hard (max_pool / binary).
        # With soft labels, pos_weight must NOT be applied — it amplifies border tokens
        # (t=0.06) with weight 30, pushing model to predict high logits for near-vessel
        # background tokens, which is exactly the FP problem we are trying to fix.
        # Soft Dice handles vessel recall; plain BCE calibrates the probability.
        is_soft = (t.max() < 0.999) and (t.max() > 0.0)
        if is_soft:
            # Plain BCE: target encodes true vessel fraction, no class reweighting needed
            loss = 0.50 * F.binary_cross_entropy_with_logits(nav_pred, t)
        else:
            # Hard binary labels: keep original pos_weight for recall bias
            loss = 0.50 * F.binary_cross_entropy_with_logits(
                nav_pred, t, pos_weight=self.ce_w[1:2])

        # Soft Dice against fraction target (drives vessel recall)
        dims = (0, 2, 3, 4)
        inter = (probs * t).sum(dims)
        den   = (probs + t).sum(dims)
        soft_dice = 1.0 - (2.0 * inter + 1e-5) / (den + 1e-5)
        loss = loss + 0.50 * soft_dice.mean()

        # clDice at token resolution (use soft label directly)
        if nav_cl_w > 0:
            loss = loss + nav_cl_w * self.cldice(probs, t)

        # Graph connectivity loss (binarize for adjacency pairs)
        if nav_conn_w > 0:
            token_label_bin = (t > 0.05).float()  # treat >5% vessel as positive for connectivity
            loss = loss + nav_conn_w * nav_connectivity_loss(nav_pred, token_label_bin)

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def gate_reg_loss(attn_gate: torch.Tensor, seg_label: torch.Tensor,
                  bg_target: float = 0.05) -> torch.Tensor:
    """Regularize the QueryGuidedFiLM attention gate to be selective.

    Uses a one-sided hinge: only penalizes bg attn when it exceeds bg_target.
    Once bg is already low, gradient → 0 so the vessel attn is not dragged down.

    Hinge: mean( max(0, attn_bg - bg_target)^2 )

    This solves both failure modes:
      - gate_reg_w too small (0.05): bg stays at 1.0 — gate saturates
      - gate_reg_w too large (0.30): vessel attn also collapses to 0

    Args:
        attn_gate: [B, 1, D, H, W] sigmoid gate values in [0, 1]
        seg_label: [B, D, H, W] integer label (0=bg, 1=vessel)
        bg_target: maximum allowed bg attention (default 0.05)
    Returns:
        scalar loss
    """
    bg_mask = (seg_label == 0)  # [B, D, H, W]
    attn_flat = attn_gate.squeeze(1)  # [B, D, H, W]
    if not bg_mask.any():
        return attn_gate.new_tensor(0.0)
    bg_attn = attn_flat[bg_mask]
    hinge = torch.clamp(bg_attn - bg_target, min=0.0)
    return (hinge ** 2).mean()


@torch.inference_mode()
def dice_fg_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5) -> float:
    pred = (logits[:, 1] > logits[:, 0]).float()
    tgt = (targets > 0).float()
    inter = (pred * tgt).sum()
    den = pred.sum() + tgt.sum()
    return float(((2 * inter + eps) / (den + eps)).item())


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * (step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(float(t), 0.0), 1.0)
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def cosine_with_warmup_restart(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    n_cycles: int = 2,
) -> float:
    """Cosine annealing with warm restarts. Warmup only on first cycle."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * (step / max(1, warmup_steps))
    effective_steps = total_steps - warmup_steps
    steps_per_cycle = max(1, effective_steps // n_cycles)
    step_in_cycle = (step - warmup_steps) % steps_per_cycle
    t = float(step_in_cycle) / float(steps_per_cycle)
    t = min(max(t, 0.0), 1.0)
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------
class EMA:
    """Enhanced EMA with warmup period for stable early training."""
    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup_steps: int = 100):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step_count += 1
        # Gradually increase decay during warmup (avoids copying bad early weights)
        if self.step_count <= self.warmup_steps:
            # Linear warmup from 0 to target decay
            d = self.decay * (self.step_count / self.warmup_steps)
        else:
            d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n].data)


# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------
def _gn(ch: int, groups: int = 16) -> nn.GroupNorm:
    g = min(int(groups), int(ch))
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


def _norm(ch: int, norm_type: str = "group", gn_groups: int = 16) -> nn.Module:
    """Create normalization layer. SOTA uses InstanceNorm3d."""
    if norm_type == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    return _gn(ch, gn_groups)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 16, dropout: float = 0.0,
                 norm_type: str = "group"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = _norm(out_ch, norm_type, gn_groups)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = _norm(out_ch, norm_type, gn_groups)
        self.drop = nn.Dropout3d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.silu(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = F.silu(self.gn2(self.conv2(x)))
        return x


class ResidualConvBlock(nn.Module):
    """Residual conv block matching SOTA's ResidualEncoder pattern.
    conv→norm→act→conv→norm + skip → act
    """
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 16, dropout: float = 0.0,
                 norm_type: str = "group"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = _norm(out_ch, norm_type, gn_groups)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = _norm(out_ch, norm_type, gn_groups)
        self.drop = nn.Dropout3d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        # 1x1 projection for skip if channels differ
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.01)
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.01)


class ResDown(nn.Module):
    """Residual downsampling block with N stacked residual blocks.
    SOTA uses n_blocks_per_stage = [1,3,4,6,6,6] — deeper stages = more blocks.
    """
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 2,
                 gn_groups: int = 16, dropout: float = 0.0, norm_type: str = "group"):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, stride=2, bias=False),
            _norm(out_ch, norm_type, gn_groups),
            nn.LeakyReLU(0.01),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualConvBlock(out_ch, out_ch, gn_groups, dropout, norm_type))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.down(x))


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


class SnakeRefine3DXYZ(nn.Module):
    def __init__(self, channels: int, K: int = 5, offset_scale: float = 3.0):
        super().__init__()
        assert K >= 3 and K % 2 == 1
        self.K = int(K)
        self.half = self.K // 2
        self.offset_scale = float(offset_scale)
        self.offset_pred = nn.Conv3d(channels, 3 * (self.K - 1), 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * self.K, channels, 1, bias=False),
            _gn(channels, 16),
            nn.SiLU(),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        delta = torch.tanh(self.offset_pred(x)) * self.offset_scale
        delta = delta.view(B, 3, self.K - 1, D, H, W)
        base = _get_grid_bdhw3(B, D, H, W, x.device)

        grids = [base]
        cum = torch.zeros((B, 3, D, H, W), device=x.device, dtype=delta.dtype)
        for s in range(self.half):
            cum = cum + delta[:, :, s]
            g = base.clone()
            g[..., 0] = g[..., 0] + (cum[:, 0] * (2.0 / max(1, W - 1)))
            g[..., 1] = g[..., 1] + (cum[:, 1] * (2.0 / max(1, H - 1)))
            g[..., 2] = g[..., 2] + (cum[:, 2] * (2.0 / max(1, D - 1)))
            grids.append(g)

        cum = torch.zeros((B, 3, D, H, W), device=x.device, dtype=delta.dtype)
        for s in range(self.half):
            cum = cum + delta[:, :, self.half + s]
            g = base.clone()
            g[..., 0] = g[..., 0] - (cum[:, 0] * (2.0 / max(1, W - 1)))
            g[..., 1] = g[..., 1] - (cum[:, 1] * (2.0 / max(1, H - 1)))
            g[..., 2] = g[..., 2] - (cum[:, 2] * (2.0 / max(1, D - 1)))
            grids.append(g)

        sampled = [F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False) for g in grids]
        y = self.fuse(torch.cat(sampled, dim=1))
        return x + y


class Up(nn.Module):
    """
    Decoder Up block with optional checkpointing.
    safe_cudnn_block disables cuDNN INSIDE this block to avoid workspace OOM.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 use_snake: bool, snake_k: int, gn_groups: int, dropout: float,
                 safe_cudnn_block: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, gn_groups, dropout=dropout)
        self.snake = SnakeRefine3DXYZ(out_ch, K=int(snake_k)) if use_snake else nn.Identity()
        self.safe_cudnn_block = bool(safe_cudnn_block)

    def _block_forward(self, x):
        if self.safe_cudnn_block:
            with torch.backends.cudnn.flags(enabled=False):
                return self.block(x)
        return self.block(x)

    def _snake_forward(self, x):
        if self.safe_cudnn_block and not isinstance(self.snake, nn.Identity):
            with torch.backends.cudnn.flags(enabled=False):
                return self.snake(x)
        return self.snake(x)

    def forward(self, x, skip, grad_ckpt: bool = False):
        x = self.up(x)
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)

        if grad_ckpt and self.training:
            x = cp.checkpoint(self._block_forward, x, use_reentrant=False, preserve_rng_state=False)
        else:
            x = self._block_forward(x)

        if not isinstance(self.snake, nn.Identity):
            if grad_ckpt and self.training:
                x = cp.checkpoint(self._snake_forward, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = self._snake_forward(x)

        return x


class FiLM3D(nn.Module):
    """Original FiLM: global vector modulation (DEPRECATED - use SpatialFiLM3D)."""
    def __init__(self, ctx_dim: int, feat_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ctx_dim, feat_ch * 2),
            nn.SiLU(),
            nn.Linear(feat_ch * 2, feat_ch * 2),
        )

    def forward(self, x: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        B, C, _, _, _ = x.shape
        gb = self.proj(ctx_vec).view(B, 2 * C, 1, 1, 1)
        gamma, beta = gb[:, :C], gb[:, C:]
        return x * (1.0 + gamma) + beta


class QueryGuidedFiLM3D(nn.Module):
    """Cross-Attention Guided FiLM for 3D vessel segmentation.

    Solves the trilinear upsampling information loss in SpatialFiLM3D.

    Problem with trilinear upsampling:
        Navigator ctx tokens live at coarse resolution (stride=4, ~1.8 mm/token).
        A 0.5 mm vessel boundary occupies <1 voxel but spans across multiple tokens.
        Blind trilinear interpolation between tokens creates a smooth gradient —
        the sharp boundary information is permanently destroyed before the decoder
        ever sees it. The decoder then receives the same blurred modulation signal
        regardless of whether a voxel is exactly ON a vessel or 3 voxels away.

    Solution — element-wise dot-product cross-attention:
        Instead of interpolating ctx tokens blindly and then projecting, we let
        the decoder's own feature map VOTE on how much of the ctx signal to trust
        at each voxel. This is implemented as:

          Q = q_proj(x)           [B, qk, D, H, W]   ← from decoder features
          K = k_proj(ctx_up)      [B, qk, D, H, W]   ← from upsampled ctx tokens
          attn = sigmoid(sum(Q*K, dim=1) / sqrt(qk))  [B, 1, D, H, W]  per-voxel scalar gate
          V = v_proj(ctx_up)      [B, 2C, D, H, W]   ← gamma+beta from ctx
          gamma, beta = split(attn * V)
          output = x * (1 + gamma) + beta

        The attention weight is computed WITHOUT an N×N matrix (no O(N²) memory).
        It is an element-wise dot product — O(N·qk) — so memory is linear in voxels.

    Memory budget (patch=128, base_ch=28, tok_dim=128, stride=4, ctx=32^3):
        u4 (16^3,  ch=224): ctx_up=1MB  Q+K=1MB  V=4MB   total ~6MB
        u3 (32^3,  ch=112): ctx_up=8MB  Q+K=4MB  V=15MB  total ~27MB
        u2 (64^3,  ch=56):  ctx_up=67MB Q+K=34MB V=59MB  total ~160MB
        u1 (128^3, ch=28):  ctx_up=537MB Q+K=235MB V=235MB total ~1007MB
        Grand total across all 4 levels: ~1.2 GB  (vs 8.6 GB for N×N attention)
        GPU headroom: 40GB - 12.7GB existing = 27GB → fits with margin.

    Why this works better than trilinear FiLM:
        The decoder feature map x already contains HIGH-RESOLUTION vessel boundary
        evidence from the encoder skip connections (sub-voxel precision from raw MRI).
        By using x to compute Q and ctx to compute K, the attention weight becomes
        HIGH where both the decoder AND navigator agree "there is a vessel here."
        At vessel boundaries, the decoder's high-res features produce a sharp Q signal
        that MODULATES the ctx contribution precisely — not a smooth gradient.
        False-positive navigator tokens (precision=0.35) get down-weighted when the
        decoder's local feature says "no vessel here," recovering precision at inference.

    Initialization:
        v_proj last layer: zeros → gamma=0, beta=0 → output = x at init.
        q_proj, k_proj: default kaiming. attn starts near sigmoid(0)=0.5 initially.
        The gate quickly learns: high attn where vessel confirmed by both streams.
    """
    def __init__(self, ctx_dim: int, feat_ch: int, gn_groups: int = 16):
        super().__init__()
        self.qk_dim = max(16, min(32, feat_ch))
        # Q from decoder features — learns what "vessel-like" looks like in feat space
        self.q_proj = nn.Conv3d(feat_ch, self.qk_dim, 1, bias=False)
        # K from ctx tokens — learns what "vessel-confirmed-by-navigator" looks like
        self.k_proj = nn.Conv3d(ctx_dim, self.qk_dim, 1, bias=False)
        # V produces the actual gamma+beta from ctx, gated by attention weight
        self.v_proj = nn.Sequential(
            nn.Conv3d(ctx_dim, feat_ch * 2, 1, bias=False),
            _gn(feat_ch * 2, gn_groups),
            nn.SiLU(),
            nn.Conv3d(feat_ch * 2, feat_ch * 2, 1),
        )
        self.scale = self.qk_dim ** -0.5
        # Init: v_proj output near zero → gamma≈0, beta≈0 → output ≈ x at start
        nn.init.zeros_(self.v_proj[-1].weight)
        nn.init.zeros_(self.v_proj[-1].bias)

    def forward(self, x: torch.Tensor, ctx_tok: torch.Tensor,
                return_attn: bool = False):
        """
        Args:
            x:           decoder feature map  [B, C, D, H, W]
            ctx_tok:     navigator ctx tokens [B, tok_dim, Dt, Ht, Wt]
            return_attn: if True, also return attn gate [B, 1, D, H, W] for diagnostics
        Returns:
            modulated feature map [B, C, D, H, W]
            (optionally) attn gate [B, 1, D, H, W]
        """
        # Step 1: upsample ctx tokens to feat resolution (trilinear — spatial alignment only)
        if ctx_tok.shape[2:] != x.shape[2:]:
            ctx_up = F.interpolate(ctx_tok.float(), size=x.shape[2:],
                                   mode="trilinear", align_corners=False).to(x.dtype)
        else:
            ctx_up = ctx_tok

        # Step 2: project decoder feat → Q, ctx_up → K
        q = self.q_proj(x)        # [B, qk, D, H, W]
        k = self.k_proj(ctx_up)   # [B, qk, D, H, W]

        # Step 3: per-voxel attention gate — element-wise dot product, no N×N matrix
        # (q * k).sum(dim=1) is O(N·qk), not O(N²)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) * self.scale)  # [B,1,D,H,W]

        # Step 4: ctx → V (gamma + beta), gated by attention weight
        gb = self.v_proj(ctx_up)  # [B, 2C, D, H, W]
        C = x.shape[1]
        gamma = attn * gb[:, :C]   # [B, C, D, H, W]
        beta  = attn * gb[:, C:]   # [B, C, D, H, W]

        out = x * (1.0 + gamma) + beta
        if return_attn:
            return out, attn
        return out


# Keep old name as alias for checkpoint compatibility during transition
SpatialFiLM3D = QueryGuidedFiLM3D


# -----------------------------------------------------------------------------
# Mamba navigator
# -----------------------------------------------------------------------------
def _parse_mamba_axes(s: str) -> List[str]:
    axes = []
    for t in str(s).split(","):
        t = t.strip().lower()
        if not t:
            continue
        if len(t) != 3 or set(t) != set("dhw"):
            raise ValueError(f"Bad mamba axis order: {t}")
        axes.append(t)
    return axes


def _permute_for_axis(x: torch.Tensor, order: str) -> torch.Tensor:
    axes = {"d": 2, "h": 3, "w": 4}
    perm = [0, 1, axes[order[0]], axes[order[1]], axes[order[2]]]
    return x.permute(*perm).contiguous()


class BiMambaSSM(nn.Module):
    def __init__(self, dim: int, axes: List[str], dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.axes = axes
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise RuntimeError("mamba_ssm not installed. Run: pip install mamba-ssm") from e
        self.mf = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)
        self.mb = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)

    def _run_seq(self, seq: torch.Tensor) -> torch.Tensor:
        y = self.norm(seq)
        y_f = self.mf(y)
        y_r = torch.flip(self.mb(torch.flip(y, dims=(1,))), dims=(1,))
        y = y_f + y_r
        y = self.dropout(y)
        return seq + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        outs = []
        for ax in self.axes:
            xp = _permute_for_axis(x, ax)
            _, _, Dp, Hp, Wp = xp.shape
            seq = xp.permute(0, 2, 3, 4, 1).reshape(B, Dp * Hp * Wp, C)
            seq = self._run_seq(seq)
            y = seq.reshape(B, Dp, Hp, Wp, C).permute(0, 4, 1, 2, 3).contiguous()
            cur = {"d": 2 + ax.index("d"), "h": 2 + ax.index("h"), "w": 2 + ax.index("w")}
            y = y.permute(0, 1, cur["d"], cur["h"], cur["w"]).contiguous()
            outs.append(y)
        return torch.stack(outs, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Adaptive-Resolution Thin-Vessel Components
# Novel contribution: learned thinness confidence head that predicts WHERE
# to allocate a fine-scale Mamba pass, guided purely by geometric thinness
# signal with no organ-specific prior.
# Differentiator vs COMMA (arXiv 2503.02332): COMMA uses randomly sampled
# local crops; here crops are STEERED by a learned per-voxel thinness score.
# ---------------------------------------------------------------------------

def thin_vessel_proximity_label(
    seg: torch.Tensor,         # [B, D, H, W] long, ground-truth labels
    thin_max_diameter: int = 4,
    proximity_radius: int = 3,
    global_stride: int = 4,
) -> torch.Tensor:
    """Compute a binary label: 1 where a thin vessel (diameter < thin_max_diameter
    voxels) is present within proximity_radius voxels.

    This label trains the ThinVesselConfidenceHead to predict where the coarse
    navigator is likely under-resolved, triggering the fine-scale Mamba pass.

    The label is computed at the NAVIGATOR token resolution (divided by
    global_stride) to match the head's output grid.

    Args:
        seg:              [B, D, H, W] full-resolution label
        thin_max_diameter: vessels with erosion-surviving radius < this are thin
        proximity_radius: label is 1 within this many voxels of a thin vessel
        global_stride:    navigator downsampling factor

    Returns:
        thin_label: [B, 1, D//gs, H//gs, W//gs] float32 binary label
    """
    if seg.ndim == 5 and seg.shape[1] == 1:
        seg = seg[:, 0]
    if seg.ndim == 3:
        seg = seg.unsqueeze(0)
    if seg.ndim != 4:
        raise ValueError(f"seg must have shape [D,H,W], [B,D,H,W], or [B,1,D,H,W], got {tuple(seg.shape)}")

    fg = (seg > 0).float().unsqueeze(1)  # [B, 1, D, H, W]

    # Erosion proxy: if a voxel survives erosion by thin_max_diameter//2,
    # its vessel is thick. Thin = present in fg but gone after erosion.
    k = thin_max_diameter // 2
    if k > 0:
        # Erode: voxel is 1 only if entire k-radius neighbourhood is fg.
        eroded = -F.max_pool3d(-fg, kernel_size=2*k+1, stride=1, padding=k)
        thin_fg = (fg - eroded).clamp(0, 1)   # only thin vessel voxels
    else:
        thin_fg = fg

    # Dilate to proximity_radius to mark neighbourhood.
    if proximity_radius > 0:
        thin_label = F.max_pool3d(
            thin_fg, kernel_size=2*proximity_radius+1,
            stride=1, padding=proximity_radius
        ).clamp(0, 1)
    else:
        thin_label = thin_fg

    # Downsample to navigator token grid: max-pool preserves presence.
    if global_stride > 1:
        thin_label = F.max_pool3d(
            thin_label, kernel_size=global_stride, stride=global_stride
        ).clamp(0, 1)

    return thin_label.detach()   # label is geometry, never needs grad


class ThinVesselConfidenceHead(nn.Module):
    """1×1×1 conv head predicting per-token 'needs fine-scale zoom' score.

    Trained with BCE against the geometric thin-vessel proximity label.
    At inference, tokens above `threshold` trigger the AdaptiveFinePass.

    Keeping this a single conv layer is intentional:
      - Lightweight: < 0.1% of total model parameters
      - Interpretable: one linear projection of navigator features
      - Generalisable: no organ-specific capacity, purely geometric
    """

    def __init__(self, tok_dim: int, threshold: float = 0.5):
        super().__init__()
        self.head = nn.Conv3d(tok_dim, 1, kernel_size=1, bias=True)
        self.threshold = threshold
        # Init bias to -2: start with low confidence (most voxels are background).
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        """tok: [B, tok_dim, Dt, Ht, Wt] → score: [B, 1, Dt, Ht, Wt] logits"""
        return self.head(tok)

    def zoom_mask(self, tok: torch.Tensor) -> torch.Tensor:
        """Binary mask: 1 where fine-scale zoom is needed."""
        return (torch.sigmoid(self.forward(tok)) > self.threshold).float()


class AdaptiveFinePass(nn.Module):
    """Runs a lightweight BiMambaSSM on flagged thin-vessel sub-volumes.

    For each connected region in the zoom_mask, extract a local sub-volume
    at full (or stride=2) resolution, process with a fine-scale Mamba block,
    and add the refined features back into the coarse token map (residual).

    This is the key differentiator from COMMA: the fine-scale pass ONLY
    executes where the thinness head says it is needed, not everywhere.

    Memory: with gradient checkpointing, the fine pass adds < 1 GB even
    when N_zoom regions × sub_size^3 is large, because only one region
    is live in memory at a time.
    """

    def __init__(
        self,
        tok_dim: int,
        fine_mamba_layers: int = 1,
        sub_size: int = 8,           # fine sub-volume size in TOKEN space
        gn_groups: int = 16,
        mamba_dropout: float = 0.0,
        mamba_axes: Optional[List[str]] = None,
        use_grad_ckpt: bool = True,
    ):
        super().__init__()
        self.sub_size = sub_size
        self.use_grad_ckpt = use_grad_ckpt

        axes = mamba_axes or ["z", "y", "x"]
        self.mamba = nn.Sequential(*[
            BiMambaSSM(tok_dim, axes=axes, dropout=mamba_dropout)
            for _ in range(fine_mamba_layers)
        ])
        # Project fine features back to tok_dim for residual addition.
        self.proj = nn.Sequential(
            nn.Conv3d(tok_dim, tok_dim, 1, bias=False),
            _gn(tok_dim, gn_groups),
        )
        # Zero-init so the fine pass starts as identity (safe warm start).
        nn.init.zeros_(self.proj[0].weight)

    def forward(
        self,
        tok: torch.Tensor,        # [B, tok_dim, Dt, Ht, Wt] coarse features
        zoom_mask: torch.Tensor,  # [B, 1, Dt, Ht, Wt] binary, from ThinHead
    ) -> torch.Tensor:
        """Return enriched coarse features with fine-scale refinement applied
        only in regions flagged by zoom_mask. Output shape == tok shape."""
        B, C, Dt, Ht, Wt = tok.shape
        S = self.sub_size
        out = tok.clone()

        for b in range(B):
            mask_b = zoom_mask[b, 0]   # [Dt, Ht, Wt]
            if mask_b.sum() == 0:
                continue

            # Find bounding box of flagged region (one region per batch item).
            # For simplicity, use the centroid of all flagged voxels to define
            # the fine-pass sub-volume. This covers the most concentrated area.
            coords = mask_b.nonzero(as_tuple=False).float()  # [N, 3]
            center = coords.mean(dim=0).long()               # [3]

            # Clamp sub-volume bounds.
            z0 = (center[0] - S//2).clamp(0, Dt - S).item()
            y0 = (center[1] - S//2).clamp(0, Ht - S).item()
            x0 = (center[2] - S//2).clamp(0, Wt - S).item()
            z1, y1, x1 = int(z0)+S, int(y0)+S, int(x0)+S
            z0, y0, x0 = int(z0), int(y0), int(x0)

            sub = tok[b:b+1, :, z0:z1, y0:y1, x0:x1].contiguous()

            if self.use_grad_ckpt and sub.requires_grad:
                refined = cp.checkpoint(self._run_mamba, sub, use_reentrant=False)
            else:
                refined = self._run_mamba(sub)

            out[b:b+1, :, z0:z1, y0:y1, x0:x1] = (
                out[b:b+1, :, z0:z1, y0:y1, x0:x1] + self.proj(refined)
            )

        return out

    def _run_mamba(self, sub: torch.Tensor) -> torch.Tensor:
        return self.mamba(sub)


class GlobalNavigator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        nav_dim: int,
        tok_dim: int,
        nav_token_stride: int,
        mamba_layers: int,
        mamba_axes: List[str],
        gn_groups: int,
        dropout: float,
        mamba_dropout: float,
    ):
        super().__init__()
        self.stem = ConvBlock(in_ch, nav_dim, gn_groups=gn_groups, dropout=dropout)
        self.token = nn.Sequential(
            nn.Conv3d(nav_dim, tok_dim, kernel_size=nav_token_stride, stride=nav_token_stride, bias=False),
            _gn(tok_dim, gn_groups),
            nn.SiLU(),
        )
        self.mamba = nn.ModuleList([
            BiMambaSSM(tok_dim, axes=mamba_axes, dropout=mamba_dropout)
            for _ in range(int(mamba_layers))
        ])

    def forward(self, low_img: torch.Tensor) -> torch.Tensor:
        # SAFETY: do NOT force-enable cuDNN here; keep navigator stem safe.
        with torch.backends.cudnn.flags(enabled=False):
            x = self.stem(low_img)
            t = self.token(x)
        for blk in self.mamba:
            t = blk(t)
        return t

# Optional: multi-scale navigator (OFF by default for checkpoint compatibility)
class MultiScaleGlobalNavigator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        nav_dim: int,
        tok_dim: int,
        nav_token_stride: int,
        mamba_layers: int,
        mamba_axes: List[str],
        gn_groups: int,
        dropout: float,
        mamba_dropout: float,
        scales: List[int],
    ):
        super().__init__()
        self.scales = [int(s) for s in scales]
        if len(self.scales) != 3:
            raise ValueError("MultiScaleGlobalNavigator expects exactly 3 scales (e.g., [2,4,8]).")

        # split tok_dim across 3 scales, distribute remainder
        base = tok_dim // 3
        rem = tok_dim - 3 * base
        dims = [base, base, base]
        for i in range(rem):
            dims[i] += 1

        self.navigators = nn.ModuleList([
            GlobalNavigator(
                in_ch=in_ch,
                nav_dim=nav_dim,
                tok_dim=dims[i],
                nav_token_stride=nav_token_stride,
                mamba_layers=mamba_layers,
                mamba_axes=mamba_axes,
                gn_groups=gn_groups,
                dropout=dropout,
                mamba_dropout=mamba_dropout,
            )
            for i in range(3)
        ])

        self.fuse = nn.Sequential(
            nn.Conv3d(tok_dim, tok_dim, 1, bias=False),
            _gn(tok_dim, gn_groups),
            nn.SiLU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img is full-res input
        target_shape = None
        toks = []
        finest = min(self.scales)

        for nav, down in zip(self.navigators, self.scales):
            if down > 1:
                low = F.avg_pool3d(img, kernel_size=down, stride=down)
            else:
                low = img
            t = nav(low)
            if down == finest:
                target_shape = t.shape[2:]
            if t.shape[2:] != target_shape:
                t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
            toks.append(t)

        return self.fuse(torch.cat(toks, dim=1))


class BottleneckCrossAttn(nn.Module):
    """
    Cross-attention at NATIVE token resolution: bottleneck queries navigator tokens.

    Instead of upsampling ctx_tok to bottleneck size (which interpolates both sides
    to the same grid, making attention degenerate), we:
      1. Pool bottleneck features DOWN to token resolution → Q  [B, qk, Dt, Ht, Wt]
      2. Keep navigator tokens at native resolution     → K/V [B, tok/bott, Dt, Ht, Wt]
      3. Attend at token resolution (small N = Dt*Ht*Wt)
      4. Upsample attended output back to bottleneck size

    This way Q and K come from genuinely different sources and the attention
    matrix is much smaller (e.g. 8×8×8=512 tokens → 512×512 = 1MB).
    """
    def __init__(self, bott: int, tok_dim: int, gn_groups: int = 16):
        super().__init__()
        self.qk_dim = max(16, bott // 4)
        self.q_proj = nn.Conv3d(bott, self.qk_dim, 1, bias=False)
        self.k_proj = nn.Conv3d(tok_dim, self.qk_dim, 1, bias=False)
        self.v_proj = nn.Conv3d(tok_dim, bott, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv3d(bott, bott, 1, bias=False),
            _gn(bott, gn_groups),
        )
        self.scale = self.qk_dim ** -0.5
        nn.init.zeros_(self.out_proj[0].weight)

    def forward(self, x: torch.Tensor, ctx_tok: torch.Tensor) -> torch.Tensor:
        """
        x: [B, bott, D, H, W]       — bottleneck features
        ctx_tok: [B, tok_dim, Dt, Ht, Wt] — navigator tokens (native resolution)
        Returns: x + cross-attended context (same shape as x)
        """
        B, C, D, H, W = x.shape
        Dt, Ht, Wt = ctx_tok.shape[2:]

        # Pool bottleneck features DOWN to token grid resolution for Q
        if (D, H, W) != (Dt, Ht, Wt):
            q_src = F.adaptive_avg_pool3d(x, (Dt, Ht, Wt))
        else:
            q_src = x

        # Q from bottleneck (pooled), K/V from navigator tokens (native)
        q = self.q_proj(q_src)        # [B, qk_dim, Dt, Ht, Wt]
        k = self.k_proj(ctx_tok)      # [B, qk_dim, Dt, Ht, Wt]
        v = self.v_proj(ctx_tok)      # [B, bott,   Dt, Ht, Wt]

        Nt = Dt * Ht * Wt
        q_flat = q.reshape(B, self.qk_dim, Nt).permute(0, 2, 1)  # [B, Nt, qk_dim]
        k_flat = k.reshape(B, self.qk_dim, Nt).permute(0, 2, 1)
        v_flat = v.reshape(B, C, Nt).permute(0, 2, 1)             # [B, Nt, bott]

        attn = torch.bmm(q_flat, k_flat.transpose(1, 2)) * self.scale  # [B, Nt, Nt]
        attn = torch.softmax(attn.float(), dim=-1).to(x.dtype)
        out_flat = torch.bmm(attn, v_flat)                        # [B, Nt, bott]
        out_tok = out_flat.permute(0, 2, 1).reshape(B, C, Dt, Ht, Wt)

        # Upsample attended result back to bottleneck spatial size
        if (Dt, Ht, Wt) != (D, H, W):
            out = F.interpolate(out_tok, size=(D, H, W),
                                mode="trilinear", align_corners=False)
        else:
            out = out_tok

        return x + self.out_proj(out)


# -----------------------------------------------------------------------------
# Local brush UNet + ctx
# -----------------------------------------------------------------------------
class LocalSnakeUNetWithCtx(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        base: int,
        gn_groups: int,
        tok_dim: int,
        snake_k_u4: int,
        snake_k: int,
        dropout: float,
        ctx_inject: str,
        safe_cudnn_u1: bool = True,
        # v6 SOTA-aligned options
        use_residual_encoder: bool = False,
        encoder_blocks: Tuple[int, ...] = (1, 2, 3, 4, 4),  # blocks per stage (stem,d1,d2,d3,d4)
        norm_type: str = "group",  # "group" or "instance"
        # sparse encoder options
        use_sparse_encoder: bool = False,
        sparse_threshold: float = 0.0,
        # per-level snake control (0 = no snake at that level)
        snake_k_u3: int = -1,  # -1 means use snake_k
        snake_k_u2: int = -1,
        snake_k_u1: int = -1,
    ):
        super().__init__()
        self.ctx_inject = set([t.strip() for t in str(ctx_inject).split(",") if t.strip()])
        self.safe_cudnn_u1 = bool(safe_cudnn_u1)
        self.use_residual_encoder = bool(use_residual_encoder)
        self.use_sparse_encoder = bool(use_sparse_encoder)

        if use_sparse_encoder:
            # Sparse residual encoder: encoder stages run on sparse voxels (huge VRAM saving
            # when most of the patch is zero-padded background — e.g., brain MRI with mask).
            # The encoder outputs DENSE bottleneck + dense skip tensors → decoder unchanged.
            assert SPCONV_AVAILABLE, (
                "spconv is required for --sparse-encoder. "
                "Install with: pip install spconv-cu121"
            )
            assert SparseResEncoder is not None, "SparseResEncoder import failed"
            self.sparse_enc = SparseResEncoder(
                in_ch=in_ch,
                base=base,
                n_blocks=encoder_blocks,
                norm_type=norm_type,
                dropout=dropout,
                sparse_threshold=float(sparse_threshold),
            )
            # Skip building stem/d1-d4 below — use sparse_enc instead
        elif use_residual_encoder:
            # SOTA-aligned residual encoder with configurable depth
            # SOTA uses [1,3,4,6,6,6] for 6 stages. We have 5 stages (stem+4 down).
            nb = encoder_blocks
            self.stem = nn.Sequential(
                *[ResidualConvBlock(in_ch if i == 0 else base, base, gn_groups, dropout, norm_type)
                  for i in range(nb[0])]
            )
            self.d1 = ResDown(base, base * 2, n_blocks=nb[1], gn_groups=gn_groups, dropout=dropout, norm_type=norm_type)
            self.d2 = ResDown(base * 2, base * 4, n_blocks=nb[2], gn_groups=gn_groups, dropout=dropout, norm_type=norm_type)
            self.d3 = ResDown(base * 4, base * 8, n_blocks=nb[3], gn_groups=gn_groups, dropout=dropout, norm_type=norm_type)
            self.d4 = ResDown(base * 8, base * 16, n_blocks=nb[4], gn_groups=gn_groups, dropout=dropout, norm_type=norm_type)
        else:
            self.stem = ConvBlock(in_ch, base, gn_groups, dropout=dropout)
            self.d1 = Down(base, base * 2, gn_groups, dropout=dropout)
            self.d2 = Down(base * 2, base * 4, gn_groups, dropout=dropout)
            self.d3 = Down(base * 4, base * 8, gn_groups, dropout=dropout)
            self.d4 = Down(base * 8, base * 16, gn_groups, dropout=dropout)

        bott = base * 16
        # Use norm_type for bottleneck when sparse or residual encoder is active
        _use_norm = use_sparse_encoder or use_residual_encoder
        self.bpre = ConvBlock(bott, bott, gn_groups, dropout=dropout, norm_type=norm_type) if _use_norm else ConvBlock(bott, bott, gn_groups, dropout=dropout)
        self.ctx_proj = nn.Conv3d(tok_dim, bott, 1, bias=False)
        self.ctx_fuse = nn.Sequential(
            nn.Conv3d(bott + bott, bott, 1, bias=False),
            _norm(bott, norm_type, gn_groups) if _use_norm else _gn(bott, gn_groups),
            nn.SiLU(),
            ConvBlock(bott, bott, gn_groups, dropout=dropout, norm_type=norm_type) if _use_norm else ConvBlock(bott, bott, gn_groups, dropout=dropout),
        )
        self.bpost = ConvBlock(bott, bott, gn_groups, dropout=dropout, norm_type=norm_type) if _use_norm else ConvBlock(bott, bott, gn_groups, dropout=dropout)
        self.bott_snake = SnakeRefine3DXYZ(bott, K=3, offset_scale=3.0)
        self.bott_cross_attn = BottleneckCrossAttn(bott, tok_dim, gn_groups)

        # Per-level snake K: -1 means inherit from snake_k, 0 means no snake
        _sk_u3 = snake_k if snake_k_u3 < 0 else snake_k_u3
        _sk_u2 = snake_k if snake_k_u2 < 0 else snake_k_u2
        _sk_u1 = snake_k if snake_k_u1 < 0 else snake_k_u1

        self.u4 = Up(bott, base * 8, base * 8, use_snake=True, snake_k=snake_k_u4,
                     gn_groups=gn_groups, dropout=dropout, safe_cudnn_block=True)
        self.u3 = Up(base * 8, base * 4, base * 4, use_snake=(_sk_u3 > 0), snake_k=max(3, _sk_u3),
                     gn_groups=gn_groups, dropout=dropout, safe_cudnn_block=True)
        self.u2 = Up(base * 4, base * 2, base * 2, use_snake=(_sk_u2 > 0), snake_k=max(3, _sk_u2),
                     gn_groups=gn_groups, dropout=dropout, safe_cudnn_block=True)
        self.u1 = Up(base * 2, base, base, use_snake=(_sk_u1 > 0), snake_k=max(3, _sk_u1),
                     gn_groups=gn_groups, dropout=dropout, safe_cudnn_block=self.safe_cudnn_u1)

        self.film_u4 = SpatialFiLM3D(tok_dim, base * 8, gn_groups)
        self.film_u3 = SpatialFiLM3D(tok_dim, base * 4, gn_groups)
        self.film_u2 = SpatialFiLM3D(tok_dim, base * 2, gn_groups)
        self.film_u1 = SpatialFiLM3D(tok_dim, base, gn_groups)

        self.head = nn.Conv3d(base, num_classes, 1)
        self.aux2 = nn.Conv3d(base * 2, num_classes, 1)
        self.aux3 = nn.Conv3d(base * 4, num_classes, 1)
        self.aux4 = nn.Conv3d(base * 8, num_classes, 1)

    def forward(self, x: torch.Tensor, ctx_tok: torch.Tensor, grad_ckpt: bool = False) -> Dict[str, torch.Tensor]:
        if self.use_sparse_encoder:
            # Sparse encoder: compute only on active (non-zero) voxels in encoder stages.
            # Returns dense bottleneck + 4 dense skip tensors — decoder is unchanged.
            x4, (x3, x2, x1, x0) = self.sparse_enc(x)
        else:
            x0 = self.stem(x)
            x1 = self.d1(x0)
            x2 = self.d2(x1)
            x3 = self.d3(x2)
            x4 = self.d4(x3)

        b = self.bpre(x4)

        # Bottleneck: spatial fusion of context tokens (unchanged)
        if "bottleneck" in self.ctx_inject:
            ctx_bott = ctx_tok
            if ctx_bott.shape[2:] != b.shape[2:]:
                ctx_bott = F.interpolate(ctx_bott, size=b.shape[2:], mode="trilinear", align_corners=False)
            ctxp = self.ctx_proj(ctx_bott)
            b = self.ctx_fuse(torch.cat([b, ctxp], dim=1))

        b = self.bpost(b)
        b = self.bott_snake(b)
        if "bottleneck" in self.ctx_inject:
            b = self.bott_cross_attn(b, ctx_tok)

        # Decoder: SPATIALLY-VARYING modulation from navigator tokens
        # Each voxel gets unique gamma/beta based on its position in the token grid
        y4 = self.u4(b, x3, grad_ckpt=grad_ckpt)
        if "u4" in self.ctx_inject:
            y4 = self.film_u4(y4, ctx_tok)

        y3 = self.u3(y4, x2, grad_ckpt=grad_ckpt)
        if "u3" in self.ctx_inject:
            y3 = self.film_u3(y3, ctx_tok)

        y2 = self.u2(y3, x1, grad_ckpt=grad_ckpt)
        if "u2" in self.ctx_inject:
            y2 = self.film_u2(y2, ctx_tok)

        y1 = self.u1(y2, x0, grad_ckpt=grad_ckpt)
        attn_gate = None
        if "u1" in self.ctx_inject:
            if self.training and isinstance(self.film_u1, QueryGuidedFiLM3D):
                y1, attn_gate = self.film_u1(y1, ctx_tok, return_attn=True)
            else:
                y1 = self.film_u1(y1, ctx_tok)

        out = {"logits": self.head(y1)}
        if attn_gate is not None:
            out["attn_gate"] = attn_gate  # [B, 1, D, H, W] — for gate_reg_loss
        if self.training:
            out["aux2"] = self.aux2(y2)
            out["aux3"] = self.aux3(y3)
            out["aux4"] = self.aux4(y4)
        return out


# ctx halo mixer: depthwise conv initialized as identity (residual starts at 0)
class TokenHaloMixer(nn.Module):
    def __init__(self, tok_dim: int, kernel: int):
        super().__init__()
        k = int(kernel)
        if k < 1 or k % 2 == 0:
            raise ValueError("ctx-halo-kernel must be odd and >= 1 (e.g., 3,5).")
        self.dw = nn.Conv3d(tok_dim, tok_dim, kernel_size=k, padding=k // 2, groups=tok_dim, bias=False)
        nn.init.zeros_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw(x)


class NavBrushModel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        tok_dim: int,
        nav_dim: int,
        nav_down: int,
        nav_token_stride: int,
        nav_mamba_layers: int,
        nav_mamba_axes: List[str],
        snake_k: int,
        snake_k_u4: int,
        gn_groups: int,
        dropout: float,
        mamba_dropout: float,
        ctx_inject: str,
        safe_cudnn_u1: bool,
        # ctx halo
        ctx_halo_tokens: int,
        ctx_halo_kernel: int,
        # optional navigator
        nav_multiscale: bool,
        # pretrained VMamba navigator
        pretrained_nav: Optional[str] = None,
        # v6 SOTA-aligned options
        use_residual_encoder: bool = False,
        encoder_blocks: Tuple[int, ...] = (1, 2, 3, 4, 4),
        norm_type: str = "group",
        # sparse encoder options
        use_sparse_encoder: bool = False,
        sparse_threshold: float = 0.0,
        # per-level snake control
        snake_k_u3: int = -1,
        snake_k_u2: int = -1,
        snake_k_u1: int = -1,
    ):
        super().__init__()
        self.nav_down = int(max(1, nav_down))
        self.nav_token_stride = int(max(1, nav_token_stride))
        # IMPORTANT: token grid stride is based on the FINEST scale
        self.global_stride = int(self.nav_down * self.nav_token_stride)

        self.ctx_halo_tokens = int(max(0, ctx_halo_tokens))
        self.ctx_halo_kernel = int(ctx_halo_kernel)
        self.ctx_halo_mixer = TokenHaloMixer(tok_dim, ctx_halo_kernel) if self.ctx_halo_tokens > 0 else nn.Identity()

        self.nav_multiscale = bool(nav_multiscale)
        self.use_pretrained_nav = pretrained_nav is not None

        if self.use_pretrained_nav:
            # Pretrained VMamba-Tiny navigator with 2D→3D inflation + 3D DWConv locality
            # gn_groups for navigator must divide tok_dim (128); find largest divisor <= gn_groups
            _nav_gn = next(g for g in range(gn_groups, 0, -1) if tok_dim % g == 0)
            self.nav = PretrainedVMamba3DNavigator(
                in_ch=in_ch,
                tok_dim=tok_dim,
                vmamba_dim=96,  # VMamba-Tiny Stage 1 dim
                ssm_ratio=2.0,
                mlp_ratio=4.0,
                num_vss_blocks=2,
                num_mamba_layers=2,
                mamba_axes=nav_mamba_axes,
                gn_groups=_nav_gn,
                mamba_dropout=mamba_dropout,
            )
            # Load and inflate pretrained weights (skip if inference-only mode)
            if pretrained_nav != "__inference_mode__" and Path(pretrained_nav).exists():
                inflate_vmamba_tiny_to_3d(self.nav, pretrained_nav, num_vss_blocks=2)
                print(f"[nav] Loaded pretrained VMamba-Tiny from {pretrained_nav}", flush=True)
            elif pretrained_nav == "__inference_mode__":
                print("[nav] PretrainedVMamba3DNavigator created (inference mode, weights from checkpoint)", flush=True)
            else:
                print(f"[nav] WARNING: pretrained weights not found at {pretrained_nav}, using random init", flush=True)
        elif self.nav_multiscale:
            scales = [self.nav_down, self.nav_down * 2, self.nav_down * 4]  # e.g., 2,4,8 when nav_down=2
            self.nav = MultiScaleGlobalNavigator(
                in_ch=in_ch,
                nav_dim=nav_dim,
                tok_dim=tok_dim,
                nav_token_stride=nav_token_stride,
                mamba_layers=nav_mamba_layers,
                mamba_axes=nav_mamba_axes,
                gn_groups=gn_groups,
                dropout=dropout,
                mamba_dropout=mamba_dropout,
                scales=scales,
            )
        else:
            self.nav = GlobalNavigator(
                in_ch=in_ch,
                nav_dim=nav_dim,
                tok_dim=tok_dim,
                nav_token_stride=nav_token_stride,
                mamba_layers=nav_mamba_layers,
                mamba_axes=nav_mamba_axes,
                gn_groups=gn_groups,
                dropout=dropout,
                mamba_dropout=mamba_dropout,
            )

        self.local = LocalSnakeUNetWithCtx(
            in_ch=in_ch,
            num_classes=2,
            base=base_ch,
            gn_groups=gn_groups,
            tok_dim=tok_dim,
            snake_k_u4=snake_k_u4,
            snake_k=snake_k,
            dropout=dropout,
            ctx_inject=ctx_inject,
            safe_cudnn_u1=safe_cudnn_u1,
            use_residual_encoder=use_residual_encoder,
            encoder_blocks=encoder_blocks,
            norm_type=norm_type,
            use_sparse_encoder=use_sparse_encoder,
            sparse_threshold=sparse_threshold,
            snake_k_u3=snake_k_u3,
            snake_k_u2=snake_k_u2,
            snake_k_u1=snake_k_u1,
        )

        # Navigator vessel detection head: predicts binary vessel at token resolution
        self.nav_head = nn.Sequential(
            nn.Conv3d(tok_dim, tok_dim // 2, 1, bias=False),
            _gn(tok_dim // 2, gn_groups),
            nn.SiLU(),
            nn.Conv3d(tok_dim // 2, 1, 1),
        )

        # Thin-vessel confidence head + adaptive fine-scale pass.
        # Novel contribution: learned per-voxel thinness detector that steers
        # a conditional fine-scale Mamba pass to thin-vessel corridors only.
        self.thin_head = ThinVesselConfidenceHead(tok_dim, threshold=0.5)
        self.adaptive_fine = AdaptiveFinePass(
            tok_dim=tok_dim,
            fine_mamba_layers=1,
            sub_size=8,           # 8 tokens × global_stride = 32 voxels at stride=4
            gn_groups=gn_groups,
            mamba_dropout=mamba_dropout,
            mamba_axes=nav_mamba_axes,
            use_grad_ckpt=True,
        )

        # Absolute position encoding for context tokens
        # Tells the local U-Net which brain region the patch comes from
        # max_tokens_per_axis must cover largest axis: e.g. 512/global_stride
        _max_tok = max(40, 512 // self.global_stride + 4)
        self.pos_enc = TokenAbsolutePositionEncoding(tok_dim, max_tokens_per_axis=_max_tok)

    def add_position_encoding(
        self,
        ctx_tok: torch.Tensor,
        starts_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """Add absolute position encoding to context tokens.

        Args:
            ctx_tok: [B, C, td, th, tw] cropped context tokens
            starts_zyx: [B, 3] patch start positions in voxel coordinates
        Returns:
            Position-encoded context tokens (same shape)
        """
        gs = self.global_stride
        halo = int(self.ctx_halo_tokens)
        # Convert voxel starts to token-grid offsets (accounting for halo)
        token_offsets = (starts_zyx // gs - halo).long()
        token_offsets = token_offsets.clamp_min(0)
        return self.pos_enc(ctx_tok, token_offsets)

    def downsample_for_nav(self, img: torch.Tensor) -> torch.Tensor:
        if self.nav_down <= 1:
            return img
        k = self.nav_down
        return F.avg_pool3d(img, kernel_size=k, stride=k)

    def nav_forward(
        self,
        img: torch.Tensor,
        use_adaptive: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run navigator and optionally apply adaptive fine-scale pass.

        Returns:
            tok:          [B, tok_dim, Dt, Ht, Wt] enriched context tokens
            thin_logits:  [B, 1, Dt, Ht, Wt] thinness head logits (or None)
        """
        if self.nav_multiscale:
            tok = self.nav(img)
        else:
            low = self.downsample_for_nav(img)
            tok = self.nav(low)

        # Thinness confidence head: always run (needed for loss even if
        # adaptive pass is disabled, e.g. during nav-only Phase 1).
        thin_logits = self.thin_head(tok)

        if use_adaptive:
            zoom_mask = self.thin_head.zoom_mask(tok)
            tok = self.adaptive_fine(tok, zoom_mask)

        return tok, thin_logits

    def crop_ctx_tokens_for_patches(
        self,
        tokens: torch.Tensor,      # [1,C,Dt,Ht,Wt] or [B,C,Dt,Ht,Wt]
        starts_zyx: torch.Tensor,  # [K,3] starts in original voxel space
        patch_dhw: Tuple[int, int, int],
    ) -> torch.Tensor:
        Bt, C, Dt, Ht, Wt = tokens.shape
        K = int(starts_zyx.shape[0])

        pd, ph, pw = patch_dhw
        sd = int(self.global_stride)

        td = int(math.ceil(pd / sd))
        th = int(math.ceil(ph / sd))
        tw = int(math.ceil(pw / sd))

        halo = int(self.ctx_halo_tokens)
        td2 = td + 2 * halo
        th2 = th + 2 * halo
        tw2 = tw + 2 * halo

        zt = (starts_zyx[:, 0] // sd).long()
        yt = (starts_zyx[:, 1] // sd).long()
        xt = (starts_zyx[:, 2] // sd).long()

        out = []
        for i in range(K):
            tb = 0 if Bt == 1 else i
            z0 = int(zt[i].item()) - halo
            y0 = int(yt[i].item()) - halo
            x0 = int(xt[i].item()) - halo

            z0 = max(0, min(z0, max(0, Dt - td2)))
            y0 = max(0, min(y0, max(0, Ht - th2)))
            x0 = max(0, min(x0, max(0, Wt - tw2)))

            out.append(tokens[tb:tb+1, :, z0:z0+td2, y0:y0+th2, x0:x0+tw2])

        ctx = torch.cat(out, dim=0)
        ctx = self.ctx_halo_mixer(ctx)
        return ctx


def freeze_nav_(model: nn.Module) -> None:
    # Freeze all navigator weights and disable its dropout by putting it in eval()
    for n, p in model.named_parameters():
        if n.startswith("nav."):
            p.requires_grad_(False)
    if hasattr(model, "nav"):
        model.nav.eval()


def freeze_local_(model: nn.Module) -> None:
    """Freeze the entire local decoder (U-Net + film layers + head).
    Used for Phase 1 (navigator-only training): only nav.* and nav_head.* receive gradients.
    The decoder is put in eval() so BN/dropout are deterministic during nav forward passes.
    """
    for n, p in model.named_parameters():
        if not n.startswith("nav.") and not n.startswith("nav_head."):
            p.requires_grad_(False)
    if hasattr(model, "local"):
        model.local.eval()


def build_param_groups(model: NavBrushModel, base_lr: float, wd: float,
                        freeze_nav: bool, freeze_local: bool = False):
    # If using pretrained VMamba navigator, use finer-grained LR groups
    if hasattr(model, 'use_pretrained_nav') and model.use_pretrained_nav:
        nav_groups = get_pretrained_nav_param_groups(
            model.nav,
            lr_pretrained=0.10,  # very gentle for pretrained layers
            lr_new=1.00,         # full LR for new projection/mamba layers
            lr_mamba=0.30,       # moderate for SSM params
        )
        groups = []
        for ng in nav_groups:
            groups.append({
                "params": ng["params"],
                "weight_decay": float(wd),
                "lr_scale": ng["lr_scale"],
            })
    else:
        nav_backbone = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("nav."):
                nav_backbone.append(p)
        groups = []
        if (not freeze_nav) and len(nav_backbone) > 0:
            groups.append({"params": nav_backbone, "weight_decay": float(wd), "lr_scale": 0.30})

    # Nav head and local U-Net (same for both pretrained and from-scratch)
    # When freeze_local=True (Phase 1 nav training), skip local decoder entirely.
    nav_head, local = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("nav."):
            continue  # already handled above
        if n.startswith("nav_head."):
            nav_head.append(p)
        else:
            if not freeze_local:
                local.append(p)

    if len(nav_head) > 0:
        groups.append({"params": nav_head, "weight_decay": float(wd), "lr_scale": 1.00})
    if len(local) > 0:
        groups.append({"params": local, "weight_decay": float(wd), "lr_scale": 1.00})
    return groups


# -----------------------------------------------------------------------------
# Checkpoint IO (safe warmstart)
# -----------------------------------------------------------------------------
def _load_checkpoint_obj(ckpt_path: Path) -> Dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def _get_model_state_from_ckpt(ckpt_obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("Unsupported checkpoint format (expected {'model': state_dict} or raw state_dict)")


def load_state_dict_match_shapes(model: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    model_sd = model.state_dict()
    load_sd = {}
    skipped = []
    for k, v in state.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            load_sd[k] = v
        else:
            skipped.append(k)
    missing, _unexpected = model.load_state_dict(load_sd, strict=False)
    return skipped, list(missing)


def save_checkpoint(path: Path, payload: Dict[str, Any]):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Chunked navigator for large volumes
# -----------------------------------------------------------------------------
def _chunk_starts_nav(dim_size: int, chunk_size: int, stride: int, gs: int) -> List[int]:
    """Generate aligned chunk start positions for navigator chunking."""
    starts = []
    pos = 0
    while pos + chunk_size <= dim_size:
        starts.append(pos)
        pos += stride
    last = max(0, dim_size - chunk_size)
    last = (last // gs) * gs
    if last not in starts:
        starts.append(last)
    return sorted(set(starts))


@torch.inference_mode()
def chunked_nav_forward_train(
    model: NavBrushModel,
    img: torch.Tensor,           # [1,1,D,H,W]
    patch_size: Tuple[int, int, int],
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_adaptive: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run navigator using 3D chunks matching the training patch_size.

    The Mamba SSM trains on ~patch_size³ patches (e.g. 160³ → 10x10x10=1000 tokens).
    For volumes larger than patch_size in any dimension, the token sequence explodes,
    causing Mamba to produce poor context or OOM.

    This function chunks the volume into overlapping 3D blocks matching patch_size,
    processes each independently through the navigator, and averages overlap regions.

    Returns: context tokens [1,C,Dt,Ht,Wt]
    """
    _, _, D, H, W = img.shape
    gs = model.global_stride  # nav_down * nav_token_stride (e.g. 8 or 16)
    pd, ph, pw = patch_size

    # FIXED: Always use chunked navigation for consistency between training and inference
    # Even small volumes are processed in chunks to ensure navigator sees same patterns
    # during training as it will during inference on any volume size.
    # This eliminates the training/inference mismatch that caused poor performance on large volumes.

    # Chunk size per dimension: use patch_size, aligned to gs
    cd = min(D, max(gs * 4, (pd // gs) * gs))
    ch = min(H, max(gs * 4, (ph // gs) * gs))
    cw = min(W, max(gs * 4, (pw // gs) * gs))

    # 25% overlap in each dimension
    ov_d = max(gs, (cd // 4 // gs) * gs)
    ov_h = max(gs, (ch // 4 // gs) * gs)
    ov_w = max(gs, (cw // 4 // gs) * gs)

    stride_d = cd - ov_d
    stride_h = ch - ov_h
    stride_w = cw - ov_w

    Dt = D // gs
    Ht = H // gs
    Wt = W // gs

    C = model.local.ctx_proj.in_channels
    device = img.device

    full_tokens      = torch.zeros((1, C, Dt, Ht, Wt), device=device, dtype=torch.float32)
    full_thin_logits = torch.zeros((1, 1, Dt, Ht, Wt), device=device, dtype=torch.float32)
    token_count      = torch.zeros((1, 1, Dt, Ht, Wt), device=device, dtype=torch.float32)

    d_starts = _chunk_starts_nav(D, cd, stride_d, gs)
    h_starts = _chunk_starts_nav(H, ch, stride_h, gs)
    w_starts = _chunk_starts_nav(W, cw, stride_w, gs)

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                chunk = img[:, :, d0:d0+cd, h0:h0+ch, w0:w0+cw]

                with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
                    chunk_tokens, chunk_thin = model.nav_forward(
                        chunk, use_adaptive=use_adaptive
                    )
                    chunk_tokens = chunk_tokens.float()
                    if chunk_thin is not None:
                        chunk_thin = chunk_thin.float()

                td, th, tw = chunk_tokens.shape[2:]
                td0, th0, tw0 = d0//gs, h0//gs, w0//gs
                td1 = min(td0 + td, Dt)
                th1 = min(th0 + th, Ht)
                tw1 = min(tw0 + tw, Wt)

                full_tokens[:, :, td0:td1, th0:th1, tw0:tw1] += \
                    chunk_tokens[:, :, :td1-td0, :th1-th0, :tw1-tw0]
                if chunk_thin is not None:
                    full_thin_logits[:, :, td0:td1, th0:th1, tw0:tw1] += \
                        chunk_thin[:, :, :td1-td0, :th1-th0, :tw1-tw0]
                token_count[:, :, td0:td1, th0:th1, tw0:tw1] += 1.0

                del chunk, chunk_tokens, chunk_thin

    cnt = token_count.clamp_min(1.0)
    full_tokens = full_tokens / cnt
    full_thin_logits = full_thin_logits / cnt
    torch.cuda.empty_cache()
    return full_tokens, full_thin_logits


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
@torch.inference_mode()
def validate_patchwise(model: NavBrushModel, ema: Optional[EMA], loader, cfg, device) -> Dict[str, float]:
    model.eval()
    backup = ema.apply_to(model) if ema is not None else None
    amp_dtype = _autocast_dtype(cfg.amp_dtype)
    main_loss = MainTubularLoss().to(device)

    total_loss = 0.0
    total_dice = 0.0

    # Navigator token-level detection metrics (accumulated over all val volumes)
    nav_tp = 0
    nav_fp = 0
    nav_fn = 0
    # Navigator calibration: mean logit on TP tokens vs FP tokens
    nav_logit_tp_sum = 0.0
    nav_logit_fp_sum = 0.0
    nav_tp_cnt = 0
    nav_fp_cnt = 0

    # Cross-attention gate quality: mean attn at vessel voxels vs background
    # Collected from film_u1 (highest resolution = most informative for vessel boundary)
    attn_vessel_sum = 0.0
    attn_bg_sum = 0.0
    attn_vessel_cnt = 0
    attn_bg_cnt = 0
    # Track whether QueryGuidedFiLM3D is present (vs old SpatialFiLM3D)
    has_qfilm = isinstance(getattr(model.local, 'film_u1', None), QueryGuidedFiLM3D)

    n = 0

    for batch in loader:
        img_cpu = batch["image"].float()
        lab_cpu = (batch["label"] > 0).to(torch.uint8)
        pools = batch["pools"]

        img_cpu = _pad_to_min_shape(img_cpu, cfg.patch_size, pad_value=0.0)
        lab_cpu = _pad_to_min_shape(lab_cpu, cfg.patch_size, pad_value=0)

        D, H, W = lab_cpu.shape
        fg = pools.get("fg", np.zeros((0, 3), np.int32))
        if fg is not None and fg.shape[0] > 0:
            m = fg.mean(axis=0)
            centers = np.tile(np.array([[int(m[0]), int(m[1]), int(m[2])]], np.int32), (cfg.patches_per_volume, 1))
        else:
            centers = np.tile(np.array([[D // 2, H // 2, W // 2]], np.int32), (cfg.patches_per_volume, 1))

        starts = centers_to_starts_aligned(centers, (D, H, W), cfg.patch_size, align=model.global_stride)

        img_gpu = img_cpu.unsqueeze(0).to(device, non_blocking=True)  # [1,1,D,H,W]

        tokens, thin_logits = chunked_nav_forward_train(
            model, img_gpu, cfg.patch_size,
            amp=cfg.amp, amp_dtype=amp_dtype,
        )

        # ---------------------------------------------------------------
        # 1. Navigator token-level quality (full volume)
        # ---------------------------------------------------------------
        with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=amp_dtype):
            nav_logits = model.nav_head(tokens)  # [1,1,Dt,Ht,Wt]

        gs = model.global_stride
        lab_gpu = lab_cpu.unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,D,H,W]
        # Soft token GT: avg_pool gives vessel fraction per token.
        # Binary threshold at 5% for precision/recall metrics (token must have >5% vessel).
        token_gt_soft = F.avg_pool3d(lab_gpu, kernel_size=gs, stride=gs)  # [1,1,Dt,Ht,Wt]
        if token_gt_soft.shape[2:] != nav_logits.shape[2:]:
            token_gt_soft = F.interpolate(token_gt_soft, size=nav_logits.shape[2:], mode="trilinear", align_corners=False)

        token_gt_bin  = (token_gt_soft > 0.05).float()                  # [1,1,Dt,Ht,Wt] threshold at 5%
        nav_pred_bin  = (torch.sigmoid(nav_logits) > 0.5).float()

        nav_tp += int((nav_pred_bin * token_gt_bin).sum().item())
        nav_fp += int((nav_pred_bin * (1 - token_gt_bin)).sum().item())
        nav_fn += int(((1 - nav_pred_bin) * token_gt_bin).sum().item())

        # Calibration: mean raw logit on TP vs FP tokens
        # High gap (e.g. TP=+2.5, FP=-0.5) → navigator is confident and discriminative
        tp_mask = (nav_pred_bin * token_gt_bin).bool().squeeze()
        fp_mask = (nav_pred_bin * (1 - token_gt_bin)).bool().squeeze()
        nav_logits_flat = nav_logits.squeeze()
        if tp_mask.any():
            nav_logit_tp_sum += float(nav_logits_flat[tp_mask].mean().item())
            nav_tp_cnt += 1
        if fp_mask.any():
            nav_logit_fp_sum += float(nav_logits_flat[fp_mask].mean().item())
            nav_fp_cnt += 1

        # ---------------------------------------------------------------
        # 2. Decoder patch forward + cross-attention gate quality
        # ---------------------------------------------------------------
        z0, y0, x0 = [int(v) for v in starts[0]]
        ip, lp = crop_patch_from_start(img_cpu, lab_cpu, (z0, y0, x0), cfg.patch_size)
        x_patch = ip.unsqueeze(0).to(device, non_blocking=True)
        y_patch = lp.unsqueeze(0).to(device, non_blocking=True).long()

        starts_t = torch.tensor([[z0, y0, x0]], device=device, dtype=torch.long)
        ctx = model.crop_ctx_tokens_for_patches(tokens, starts_t, cfg.patch_size)

        if has_qfilm and "u1" in model.local.ctx_inject:
            # Run decoder manually up to u1 to intercept attn gate
            # This mirrors LocalSnakeUNetWithCtx.forward exactly
            local = model.local
            x0_enc = local.stem(x_patch)
            x1_enc = local.d1(x0_enc)
            x2_enc = local.d2(x1_enc)
            x3_enc = local.d3(x2_enc)
            x4_enc = local.d4(x3_enc)
            b = local.bpre(x4_enc)
            if "bottleneck" in local.ctx_inject:
                ctx_bott = ctx
                if ctx_bott.shape[2:] != b.shape[2:]:
                    ctx_bott = F.interpolate(ctx_bott, size=b.shape[2:], mode="trilinear", align_corners=False)
                ctxp = local.ctx_proj(ctx_bott)
                b = local.ctx_fuse(torch.cat([b, ctxp], dim=1))
            b = local.bpost(b)
            b = local.bott_snake(b)
            if "bottleneck" in local.ctx_inject:
                b = local.bott_cross_attn(b, ctx)
            y4_dec = local.u4(b, x3_enc)
            if "u4" in local.ctx_inject:
                y4_dec = local.film_u4(y4_dec, ctx)
            y3_dec = local.u3(y4_dec, x2_enc)
            if "u3" in local.ctx_inject:
                y3_dec = local.film_u3(y3_dec, ctx)
            y2_dec = local.u2(y3_dec, x1_enc)
            if "u2" in local.ctx_inject:
                y2_dec = local.film_u2(y2_dec, ctx)
            y1_dec = local.u1(y2_dec, x0_enc)
            # film_u1 with return_attn=True to get the gate
            y1_dec, attn_gate = local.film_u1(y1_dec, ctx, return_attn=True)  # attn: [1,1,D,H,W]
            logits = local.head(y1_dec)
            out = {"logits": logits}
            loss = main_loss(logits, y_patch)

            # Measure gate quality: vessel mask at full patch resolution
            vessel_mask = (y_patch[0] > 0)       # [D,H,W] bool
            attn_flat = attn_gate[0, 0]           # [D,H,W]
            if vessel_mask.any():
                attn_vessel_sum += float(attn_flat[vessel_mask].mean().item())
                attn_vessel_cnt += 1
            bg_mask = ~vessel_mask
            if bg_mask.any():
                attn_bg_sum += float(attn_flat[bg_mask].mean().item())
                attn_bg_cnt += 1
        else:
            with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=amp_dtype):
                out = model.local(x_patch, ctx, grad_ckpt=False)
                loss = main_loss(out["logits"], y_patch)

        total_loss += float(loss.item())
        total_dice += dice_fg_from_logits(out["logits"], y_patch)
        n += 1

        if cfg.val_patch_max_batches > 0 and n >= cfg.val_patch_max_batches:
            break

    if ema is not None and backup is not None:
        ema.restore(model, backup)

    nav_recall = nav_tp / max(1, nav_tp + nav_fn)
    nav_prec   = nav_tp / max(1, nav_tp + nav_fp)
    nav_f1     = 2 * nav_recall * nav_prec / max(1e-6, nav_recall + nav_prec)
    nav_logit_tp = nav_logit_tp_sum / max(1, nav_tp_cnt)   # mean logit on true-positive tokens
    nav_logit_fp = nav_logit_fp_sum / max(1, nav_fp_cnt)   # mean logit on false-positive tokens
    attn_vessel  = attn_vessel_sum  / max(1, attn_vessel_cnt)
    attn_bg      = attn_bg_sum      / max(1, attn_bg_cnt)

    return {
        "val_patch_loss":      total_loss / max(1, n),
        "val_patch_fg_dice":   total_dice / max(1, n),
        # Navigator binary detection
        "nav_token_recall":    nav_recall,
        "nav_token_prec":      nav_prec,
        "nav_token_f1":        nav_f1,
        # Navigator calibration (higher gap = more confident navigator)
        "nav_logit_tp":        nav_logit_tp,
        "nav_logit_fp":        nav_logit_fp,
        "nav_logit_gap":       nav_logit_tp - nav_logit_fp,
        # Cross-attention gate quality (higher gap = gate correctly focuses on vessels)
        "attn_vessel":         attn_vessel,
        "attn_bg":             attn_bg,
        "attn_gap":            attn_vessel - attn_bg,
    }


# -----------------------------------------------------------------------------
# Gaussian importance weighting for sliding-window aggregation
# (nnU-Net-style: center voxels weighted higher, border voxels lower)
# -----------------------------------------------------------------------------
def gaussian_weight_3d(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a 3D Gaussian importance map for a patch.

    Center voxels get weight ~1.0, border voxels get exponentially lower weights.
    This reduces border artifacts when averaging overlapping patches, matching
    the nnU-Net sliding-window inference strategy.

    Args:
        patch_size: (D, H, W) patch dimensions
        sigma_scale: sigma = dim_size * sigma_scale (default 0.125 = nnU-Net)
        device: target device
    Returns:
        [D, H, W] weight tensor
    """
    weight = torch.ones(1, device=device)
    for dim_size in patch_size:
        sigma = dim_size * sigma_scale
        center = (dim_size - 1) / 2.0
        coords = torch.arange(dim_size, dtype=torch.float32, device=device)
        g = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
        # Reshape for broadcasting: [dim_size, 1, 1] or [1, dim_size, 1] etc.
        shape = [1] * len(patch_size)
        shape[len(shape) - len(patch_size) + list(patch_size).index(dim_size)] = dim_size
        # Actually, build sequentially:
        weight = weight.unsqueeze(-1) * g
    # weight is now [D, H, W]
    weight = weight.clamp_min(1e-6)
    # Normalize so max = 1
    weight = weight / weight.max()
    return weight


def gaussian_weight_3d_fast(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Vectorized Gaussian weight computation for 3D patches."""
    d, h, w = patch_size
    sd, sh, sw = d * sigma_scale, h * sigma_scale, w * sigma_scale
    cd, ch, cw = (d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0

    zz = torch.arange(d, dtype=torch.float32, device=device)
    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)

    gz = torch.exp(-0.5 * ((zz - cd) / max(sd, 1e-6)) ** 2)
    gy = torch.exp(-0.5 * ((yy - ch) / max(sh, 1e-6)) ** 2)
    gx = torch.exp(-0.5 * ((xx - cw) / max(sw, 1e-6)) ** 2)

    # Outer product: [D] x [H] x [W] -> [D, H, W]
    weight = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    weight = weight.clamp_min(1e-6)
    weight = weight / weight.max()
    return weight


# -----------------------------------------------------------------------------
# Absolute position encoding for context tokens
# (tells the model which brain region each patch comes from)
# -----------------------------------------------------------------------------
class TokenAbsolutePositionEncoding(nn.Module):
    """
    Factored 3D absolute position encoding for navigator context tokens.

    Each token at absolute position (z, y, x) in the full-volume token grid
    receives a unique embedding = z_emb[z] + y_emb[y] + x_emb[x].

    This tells the local U-Net WHICH brain region the patch comes from
    (frontal lobe, posterior fossa, etc.), enabling region-specific vessel
    prediction patterns.

    Applied BEFORE augmentation, so flips/rotations transform both content
    and position encoding together, maintaining alignment.

    Memory: 3 * max_tokens * tok_dim * 4 bytes ≈ 3 * 40 * 128 * 4 = 60KB.
    """
    def __init__(self, tok_dim: int, max_tokens_per_axis: int = 40):
        super().__init__()
        self.max_tok = max_tokens_per_axis
        self.z_emb = nn.Embedding(max_tokens_per_axis, tok_dim)
        self.y_emb = nn.Embedding(max_tokens_per_axis, tok_dim)
        self.x_emb = nn.Embedding(max_tokens_per_axis, tok_dim)
        # Small initialization for stable startup
        for emb in [self.z_emb, self.y_emb, self.x_emb]:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(
        self,
        ctx_tok: torch.Tensor,
        token_offset_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ctx_tok: [B, C, td, th, tw] context tokens
            token_offset_zyx: [B, 3] absolute token grid offset for each patch
        Returns:
            ctx_tok + position_encoding (same shape)
        """
        B, C, td, th, tw = ctx_tok.shape
        device = ctx_tok.device
        out = ctx_tok.clone()
        for b in range(B):
            z0 = int(token_offset_zyx[b, 0].item())
            y0 = int(token_offset_zyx[b, 1].item())
            x0 = int(token_offset_zyx[b, 2].item())
            z_ids = torch.arange(z0, z0 + td, device=device).clamp(0, self.max_tok - 1)
            y_ids = torch.arange(y0, y0 + th, device=device).clamp(0, self.max_tok - 1)
            x_ids = torch.arange(x0, x0 + tw, device=device).clamp(0, self.max_tok - 1)
            # Factored 3D: sum of axis-aligned embeddings
            pos = (
                self.z_emb(z_ids).view(td, 1, 1, C)
                + self.y_emb(y_ids).view(1, th, 1, C)
                + self.x_emb(x_ids).view(1, 1, tw, C)
            ).permute(3, 0, 1, 2)  # [C, td, th, tw]
            out[b] = out[b] + pos
        return out


@torch.inference_mode()
def validate_fullvolume(
    model: NavBrushModel,
    ema: Optional[EMA],
    val_items: List[Dict[str, str]],
    cfg,
    device,
) -> Dict[str, float]:
    model.eval()
    backup = ema.apply_to(model) if ema is not None else None
    amp_dtype = _autocast_dtype(cfg.amp_dtype)

    dices = []
    max_cases = int(cfg.val_fullvol_max_cases)

    for it in val_items[:max_cases]:
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], cfg.cache_dir, "image")
        lp = _resolve_cached_path(uid, it["label_pt"], cfg.cache_dir, "label")
        img = torch.load(ip, map_location="cpu").float()
        lab = torch.load(lp, map_location="cpu")

        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.clamp(0.0, 1.0)

        if lab.ndim == 4 and lab.shape[0] == 1:
            lab = lab[0]
        lab = (lab > 0).long()

        D, H, W = lab.shape
        vox = int(D * H * W)
        if cfg.val_fullvol_max_voxels > 0 and vox > int(cfg.val_fullvol_max_voxels):
            print(f"[val_fullvol] skip {uid}: vox={vox} > max_voxels={cfg.val_fullvol_max_voxels}", flush=True)
            continue

        img_gpu = img.unsqueeze(0).to(device, non_blocking=True)  # [1,1,D,H,W]
        lab_gpu = lab.unsqueeze(0).to(device, non_blocking=True)  # [1,D,H,W]

        tokens, thin_logits = chunked_nav_forward_train(
            model, img_gpu, cfg.patch_size,
            amp=cfg.amp, amp_dtype=amp_dtype,
        )

        pd, ph, pw = cfg.patch_size
        overlap = float(cfg.val_fullvol_overlap)
        stride_d = max(1, int(pd * (1.0 - overlap)))
        stride_h = max(1, int(ph * (1.0 - overlap)))
        stride_w = max(1, int(pw * (1.0 - overlap)))

        zs = list(range(0, max(1, D - pd + 1), stride_d)) + ([D - pd] if D > pd else [])
        ys = list(range(0, max(1, H - ph + 1), stride_h)) + ([H - ph] if H > ph else [])
        xs = list(range(0, max(1, W - pw + 1), stride_w)) + ([W - pw] if W > pw else [])

        # Gaussian importance weighting (nnU-Net-style)
        gw = gaussian_weight_3d_fast(cfg.patch_size, sigma_scale=0.125, device=device)
        gw = gw.unsqueeze(0).unsqueeze(0)  # [1, 1, pd, ph, pw]

        out_logits = torch.zeros((1, 2, D, H, W), device=device, dtype=torch.float16)
        out_weight = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float16)

        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    patch = img_gpu[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw]
                    st = torch.tensor([[z0, y0, x0]], device=device, dtype=torch.long)
                    ctx = model.crop_ctx_tokens_for_patches(tokens, st, cfg.patch_size)
                    # Add absolute position encoding
                    ctx = model.add_position_encoding(ctx, st)
                    with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=amp_dtype):
                        logits = model.local(patch, ctx, grad_ckpt=False)["logits"].to(torch.float16)
                    # Crop Gaussian weight to actual patch size (boundary patches may be smaller)
                    ad, ah, aw = logits.shape[2], logits.shape[3], logits.shape[4]
                    gw_crop = gw[:, :, :ad, :ah, :aw].half()
                    # Gaussian-weighted accumulation (center > borders)
                    out_logits[:, :, z0:z0+ad, y0:y0+ah, x0:x0+aw] += logits * gw_crop
                    out_weight[:, :, z0:z0+ad, y0:y0+ah, x0:x0+aw] += gw_crop

        out_logits = (out_logits / out_weight.clamp_min(1e-6)).to(torch.float32)
        dice = dice_fg_from_logits(out_logits, lab_gpu)
        dices.append(dice)
        print(f"[val_fullvol] {uid} dice={dice:.4f}", flush=True)
        del img_gpu, lab_gpu, tokens, out_logits, out_weight
        torch.cuda.empty_cache()

    if ema is not None and backup is not None:
        ema.restore(model, backup)

    return {"val_fullvol_fg_dice": float(np.mean(dices)) if dices else 0.0}


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    cache_dir: Path
    out_dir: Path
    sampling_db_dir: Optional[Path]

    epochs: int
    epoch_size: int
    patches_per_volume: int
    patch_micro_batch: int
    accum_steps: int
    num_workers: int
    seed: int
    val_ratio: float
    patch_size: Tuple[int, int, int]

    lr: float
    weight_decay: float
    warmup_epochs: int
    lr_restart: bool
    lr_restart_cycles: int
    grad_clip: float

    amp: bool
    amp_dtype: str

    ram_cache_items: int
    prefer_sampling_base_uid: bool

    # sampling
    p_fg: float
    p_bg_boundary: float
    p_bg_hard: float
    p_bg_easy: float
    p_vessel_boundary: float

    # augmentation
    flip_prob: float
    rot_prob: float
    rot_deg: float
    elastic_prob: float
    elastic_alpha: float
    elastic_coarse: int
    intensity_aug_prob: float
    mixup_prob: float
    mixup_alpha: float

    # model
    base_ch: int
    tok_dim: int
    nav_dim: int
    nav_down: int
    nav_token_stride: int
    nav_mamba_layers: int
    nav_mamba_axes: Tuple[str, ...]
    mamba_dropout: float
    snake_k: int
    snake_k_u4: int
    snake_k_u3: int
    snake_k_u2: int
    snake_k_u1: int
    ctx_inject: str
    gn_groups: int
    dropout: float
    nav_multiscale: bool
    pretrained_nav: Optional[str]  # path to VMamba-Tiny pretrained weights (None=from scratch)

    # v6 SOTA-aligned
    use_residual_encoder: bool
    encoder_blocks: Tuple[int, ...]
    norm_type: str
    use_skeleton_recall: bool
    use_strong_aug: bool
    # sparse encoder
    use_sparse_encoder: bool
    sparse_threshold: float

    # ctx halo
    ctx_halo_tokens: int
    ctx_halo_kernel: int

    # deep sup base weights
    ds_w2: float
    ds_w3: float
    ds_w4: float

    # misc
    input_clamp_01: bool
    ema: bool
    ema_decay: float
    grad_checkpoint: bool
    freeze_nav: bool
    freeze_local: bool   # Phase 1: freeze decoder, train navigator only
    # Navigator supervision weights
    nav_w: float         # overall weight for navigator loss (0 = disabled)
    nav_cl_w: float      # clDice weight within navigator loss
    nav_conn_w: float    # graph connectivity weight within navigator loss
    nav_ce_fg_w: float   # FG class weight in navigator BCE (30=recall-biased; 8-10 for Phase 1 precision)
    gate_reg_w: float    # Weight for gate regularization loss (0=off; 0.05-0.10 for Phase 2)
    thin_w: float        # Weight for thin-vessel confidence head BCE loss (0=off; 0.1-0.5)
    use_compile: bool    # torch.compile the model for ~15-20% speedup
    # SAFE defaults
    channels_last: bool
    safe_cudnn: bool
    safe_cudnn_u1: bool

    # validation
    val_patch_max_batches: int
    val_fullvol_every: int
    val_fullvol_max_cases: int
    val_fullvol_overlap: float
    val_fullvol_max_voxels: int

    log_mem_every: int


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _default_out_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("/workspace/mamba_snake/logs") / f"navbrush_run_{ts}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--sampling-db-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. If omitted, a timestamped folder under /workspace/mamba_snake/logs/ is used.")

    ap.add_argument("--init-ckpt", type=Path, default=None)
    ap.add_argument("--init-mode", choices=["strict", "match"], default="strict")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--epoch-size", type=int, default=256)
    ap.add_argument("--patches-per-volume", type=int, default=4)
    ap.add_argument("--patch-micro-batch", type=int, default=1)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[192, 192, 192])
    ap.add_argument("--accum-steps", type=int, default=2)

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.10)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-restart", action="store_true", default=False,
                    help="Use cosine annealing with warm restarts (2 cycles by default).")
    ap.add_argument("--lr-restart-cycles", type=int, default=2,
                    help="Number of cosine cycles for --lr-restart (default: 2).")

    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--no-amp", action="store_true")

    ap.add_argument("--ram-cache-items", type=int, default=2)
    ap.add_argument("--prefer-sampling-base-uid", action="store_true", default=True)
    ap.add_argument("--no-prefer-sampling-base-uid", action="store_false", dest="prefer_sampling_base_uid")

    # aug
    ap.add_argument("--flip-prob", type=float, default=0.5)
    ap.add_argument("--rot-prob", type=float, default=0.25)
    ap.add_argument("--rot-deg", type=float, default=12.0)
    ap.add_argument("--elastic-prob", type=float, default=0.30)
    ap.add_argument("--elastic-alpha", type=float, default=1.6)
    ap.add_argument("--elastic-coarse", type=int, default=8)
    ap.add_argument("--intensity-aug-prob", type=float, default=0.50)
    ap.add_argument("--mixup-prob", type=float, default=0.0, help="Probability of applying MixUp augmentation (disabled by default)")
    ap.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp alpha parameter for Beta distribution")

    # sampling
    ap.add_argument("--p-fg", type=float, default=0.80)
    ap.add_argument("--p-bg-boundary", type=float, default=0.15)
    ap.add_argument("--p-bg-hard", type=float, default=0.05)
    ap.add_argument("--p-bg-easy", type=float, default=0.00)
    ap.add_argument("--p-vessel-boundary", type=float, default=0.00,
                    help="Fraction of patches centered near vessel bbox edges (carved from p_fg)")

    # model
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--gn-groups", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--tok-dim", type=int, default=128)
    ap.add_argument("--nav-dim", type=int, default=48)
    ap.add_argument("--nav-down", type=int, default=4)
    ap.add_argument("--nav-token-stride", type=int, default=4)
    ap.add_argument("--nav-mamba-layers", type=int, default=4)
    ap.add_argument("--nav-mamba-axes", type=str, default="dhw,hwd,wdh")
    ap.add_argument("--mamba-dropout", type=float, default=0.05)

    ap.add_argument("--snake-k", type=int, default=5)
    ap.add_argument("--snake-k-u4", type=int, default=3)
    ap.add_argument("--snake-k-u3", type=int, default=-1, help="Snake K for u3 (-1=inherit snake-k, 0=disable)")
    ap.add_argument("--snake-k-u2", type=int, default=-1, help="Snake K for u2 (-1=inherit snake-k, 0=disable)")
    ap.add_argument("--snake-k-u1", type=int, default=-1, help="Snake K for u1 (-1=inherit snake-k, 0=disable)")
    ap.add_argument("--ctx-inject", type=str, default="bottleneck,u4,u3,u2,u1")

    # ctx halo
    ap.add_argument("--ctx-halo-tokens", type=int, default=0)
    ap.add_argument("--ctx-halo-kernel", type=int, default=3)

    ap.add_argument("--no-input-clamp-01", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.9995)

    ap.add_argument("--grad-checkpoint", action="store_true", default=False,
                        help="Use gradient checkpointing (recommended for A100 40GB)")
    ap.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint",
                        help="Disable gradient checkpointing")

    ap.add_argument("--channels-last", action="store_true")

    # cuDNN safety
    ap.add_argument("--enable-cudnn", action="store_true")
    ap.add_argument("--unsafe-cudnn", action="store_true")
    ap.add_argument("--no-safe-cudnn-u1", action="store_true")

    # optional nav multiscale
    ap.add_argument("--nav-multiscale", action="store_true")
    ap.add_argument("--pretrained-nav", type=str, default=None,
                    help="Path to VMamba-Tiny pretrained weights (.pth). Enables pretrained navigator with 2D→3D inflation.")

    # v6 SOTA-aligned
    ap.add_argument("--residual-encoder", action="store_true", default=False,
                    help="Use residual encoder blocks (SOTA-aligned). Much deeper encoder.")
    ap.add_argument("--encoder-blocks", type=int, nargs=5, default=[1, 2, 3, 4, 4],
                    help="Number of residual blocks per encoder stage [stem,d1,d2,d3,d4]. SOTA uses [1,3,4,6,6].")
    ap.add_argument("--norm-type", choices=["group", "instance"], default="group",
                    help="Normalization type. SOTA uses instance.")
    ap.add_argument("--skeleton-recall", action="store_true", default=False,
                    help="Enable true skeleton recall loss (precomputes skeletons via skimage).")
    ap.add_argument("--strong-aug", action="store_true", default=False,
                    help="Enable SOTA-level augmentation (SimLowRes, Sharpening, BrightnessGradient, etc.).")

    # sparse encoder
    ap.add_argument("--sparse-encoder", action="store_true", default=False,
                    help="Use sparse convolution in encoder stages (requires spconv-cu121). "
                         "Saves VRAM when input has large zero-padded background regions.")
    ap.add_argument("--sparse-threshold", type=float, default=0.0,
                    help="Sparsification threshold: voxels with |x| > threshold are active. "
                         "0.0 = all non-zero voxels active (any brain tissue). Higher values "
                         "restrict to bright regions only. Default: 0.0")

    # validation
    ap.add_argument("--val-patch-max-batches", type=int, default=16)
    ap.add_argument("--val-fullvol-every", type=int, default=5)
    ap.add_argument("--val-fullvol-max-cases", type=int, default=4)
    ap.add_argument("--val-fullvol-overlap", type=float, default=0.5)
    ap.add_argument("--val-fullvol-max-voxels", type=int, default=0,
                    help="Max voxels for fullvol validation (0=unlimited, was 22M)")

    ap.add_argument("--log-mem-every", type=int, default=50)

    # NEW: safety knobs
    ap.add_argument("--min-nav-down", type=int, default=0,
                    help="Force nav_down >= this value to reduce navigator memory/workspace risk.")
    ap.add_argument("--freeze-nav", action="store_true",
                    help="Freeze navigator weights (Phase 2: decoder-only training).")
    ap.add_argument("--freeze-local", action="store_true",
                    help="Freeze decoder weights (Phase 1: navigator-only training).")

    # Navigator supervision
    ap.add_argument("--nav-w", type=float, default=0.15,
                    help="Navigator loss weight (0=disabled, 0.15=recommended). Warms up from 0 over first 30%% epochs.")
    ap.add_argument("--nav-cl-w", type=float, default=0.10,
                    help="clDice weight within navigator loss.")
    ap.add_argument("--nav-conn-w", type=float, default=0.10,
                    help="Graph connectivity weight within navigator loss.")
    ap.add_argument("--nav-ce-fg-w", type=float, default=30.0,
                    help="FG class weight in navigator BCE loss (default=30 for recall-biased; use 8-10 in Phase 1 to improve precision).")
    ap.add_argument("--gate-reg-w", type=float, default=0.0,
                    help="Weight for QueryGuidedFiLM gate regularization loss (penalizes gate at bg voxels). Use 0.05-0.10 in Phase 2 to prevent gate saturation.")
    ap.add_argument("--thin-w", type=float, default=0.0,
                    help="Weight for thin-vessel confidence head BCE loss. "
                         "Trains the thinness head to predict which tokens need "
                         "the adaptive fine-scale Mamba pass. Recommended: 0.2-0.5. "
                         "Set to 0 to disable adaptive resolution (backward compatible).")
    ap.add_argument("--use-compile", action="store_true", default=False,
                    help="Enable torch.compile (reduce-overhead mode). Gives ~15-20%% speedup "
                         "after a one-time compilation cost (~5 min first epoch).")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    dev = torch.device("cuda:0")
    print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')} -> using {dev}", flush=True)
    print(f"[gpu] torch sees {torch.cuda.device_count()} CUDA device(s). Current=0 Name={torch.cuda.get_device_name(0)}", flush=True)

    seed_everything(int(args.seed))
    safe_cudnn = (not args.unsafe_cudnn)
    configure_cuda_for_a100(safe_cudnn=safe_cudnn, allow_tf32=True, enable_cudnn=bool(args.enable_cudnn))
    print(f"[cudnn] enabled={torch.backends.cudnn.enabled} benchmark={torch.backends.cudnn.benchmark} deterministic={torch.backends.cudnn.deterministic}", flush=True)

    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir()
    out_dir = Path(out_dir)

    nav_down_eff = int(args.nav_down)
    if int(args.min_nav_down) > 0:
        nav_down_eff = max(nav_down_eff, int(args.min_nav_down))

    cfg = TrainConfig(
        cache_dir=Path(args.cache_dir),
        out_dir=out_dir,
        sampling_db_dir=Path(args.sampling_db_dir) if args.sampling_db_dir is not None else None,

        epochs=int(args.epochs),
        epoch_size=int(args.epoch_size),
        patches_per_volume=int(args.patches_per_volume),
        patch_micro_batch=max(1, int(args.patch_micro_batch)),
        accum_steps=max(1, int(args.accum_steps)),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        patch_size=tuple(int(x) for x in args.patch_size),

        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_epochs=int(args.warmup_epochs),
        lr_restart=bool(args.lr_restart),
        lr_restart_cycles=int(args.lr_restart_cycles),
        grad_clip=float(args.grad_clip),

        amp=(not bool(args.no_amp)),
        amp_dtype=str(args.amp_dtype),

        ram_cache_items=int(args.ram_cache_items),
        prefer_sampling_base_uid=bool(args.prefer_sampling_base_uid),

        p_fg=float(args.p_fg),
        p_bg_boundary=float(args.p_bg_boundary),
        p_bg_hard=float(args.p_bg_hard),
        p_bg_easy=float(args.p_bg_easy),
        p_vessel_boundary=float(args.p_vessel_boundary),

        flip_prob=float(args.flip_prob),
        rot_prob=float(args.rot_prob),
        rot_deg=float(args.rot_deg),
        elastic_prob=float(args.elastic_prob),
        elastic_alpha=float(args.elastic_alpha),
        elastic_coarse=int(args.elastic_coarse),
        intensity_aug_prob=float(args.intensity_aug_prob),
        mixup_prob=float(args.mixup_prob),
        mixup_alpha=float(args.mixup_alpha),

        base_ch=int(args.base_ch),
        tok_dim=int(args.tok_dim),
        nav_dim=int(args.nav_dim),
        nav_down=int(nav_down_eff),
        nav_token_stride=int(args.nav_token_stride),
        nav_mamba_layers=int(args.nav_mamba_layers),
        nav_mamba_axes=tuple(_parse_mamba_axes(args.nav_mamba_axes)),
        mamba_dropout=float(args.mamba_dropout),
        snake_k=int(args.snake_k),
        snake_k_u4=int(args.snake_k_u4),
        snake_k_u3=int(args.snake_k_u3),
        snake_k_u2=int(args.snake_k_u2),
        snake_k_u1=int(args.snake_k_u1),
        ctx_inject=str(args.ctx_inject),
        gn_groups=int(args.gn_groups),
        dropout=float(args.dropout),
        nav_multiscale=bool(args.nav_multiscale),
        pretrained_nav=args.pretrained_nav,

        use_residual_encoder=bool(args.residual_encoder),
        encoder_blocks=tuple(int(x) for x in args.encoder_blocks),
        norm_type=str(args.norm_type),
        use_skeleton_recall=bool(args.skeleton_recall),
        use_strong_aug=bool(args.strong_aug),
        use_sparse_encoder=bool(args.sparse_encoder),
        sparse_threshold=float(args.sparse_threshold),

        ctx_halo_tokens=int(args.ctx_halo_tokens),
        ctx_halo_kernel=int(args.ctx_halo_kernel),

        ds_w2=0.20,
        ds_w3=0.10,
        ds_w4=0.05,

        input_clamp_01=(not bool(args.no_input_clamp_01)),
        ema=(not bool(args.no_ema)),
        ema_decay=float(args.ema_decay),
        grad_checkpoint=bool(args.grad_checkpoint),
        freeze_nav=bool(args.freeze_nav),
        freeze_local=bool(args.freeze_local),
        nav_w=float(args.nav_w),
        nav_cl_w=float(args.nav_cl_w),
        nav_conn_w=float(args.nav_conn_w),
        nav_ce_fg_w=float(args.nav_ce_fg_w),
        gate_reg_w=float(args.gate_reg_w),
        thin_w=float(args.thin_w),
        use_compile=bool(args.use_compile),

        channels_last=bool(args.channels_last),
        safe_cudnn=bool(safe_cudnn),
        safe_cudnn_u1=(not bool(args.no_safe_cudnn_u1)),

        val_patch_max_batches=int(args.val_patch_max_batches),
        val_fullvol_every=int(args.val_fullvol_every),
        val_fullvol_max_cases=int(args.val_fullvol_max_cases),
        val_fullvol_overlap=float(args.val_fullvol_overlap),
        val_fullvol_max_voxels=int(args.val_fullvol_max_voxels),

        log_mem_every=int(args.log_mem_every),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("[cfg] " + json_dump_safe(asdict(cfg), indent=2), flush=True)

    # -------------------------------------------------------------------------
    # Build dataset
    # -------------------------------------------------------------------------
    all_items = list_cases_from_cache(cfg.cache_dir)
    train_items, val_items = split_train_val_baseuids(
        all_items,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        val_original_only=True,
    )
    print(f"[data] total={len(all_items)} train={len(train_items)} val={len(val_items)}", flush=True)

    train_ds = CachedVolumeDataset(
        items=train_items,
        cache_dir=cfg.cache_dir,
        sampling_db_dir=cfg.sampling_db_dir,
        ram_cache_items=cfg.ram_cache_items,
        input_clamp_01=cfg.input_clamp_01,
        prefer_sampling_base_uid=cfg.prefer_sampling_base_uid,
        pool_cache_items=1,
        compute_skeleton=cfg.use_skeleton_recall,
    )
    if cfg.use_skeleton_recall:
        print("[skel] Skeleton recall enabled — precomputing TRUE skeletons via skimage", flush=True)

    fold_all = len(val_items) == 0
    if fold_all:
        print("[fold_all] No validation holdout — training on ALL data. "
              "Best checkpoint saved every val-fullvol-every epochs.", flush=True)

    if not fold_all:
        val_ds = CachedVolumeDataset(
            items=val_items,
            cache_dir=cfg.cache_dir,
            sampling_db_dir=cfg.sampling_db_dir,
            ram_cache_items=1,
            input_clamp_01=cfg.input_clamp_01,
            prefer_sampling_base_uid=cfg.prefer_sampling_base_uid,
            pool_cache_items=1,
        )
    else:
        val_ds = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_bs1,
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, cfg.num_workers // 2),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_bs1,
            persistent_workers=(cfg.num_workers > 0),
        )

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    model = NavBrushModel(
        in_ch=1,
        base_ch=cfg.base_ch,
        tok_dim=cfg.tok_dim,
        nav_dim=cfg.nav_dim,
        nav_down=cfg.nav_down,
        nav_token_stride=cfg.nav_token_stride,
        nav_mamba_layers=cfg.nav_mamba_layers,
        nav_mamba_axes=list(cfg.nav_mamba_axes),
        snake_k=cfg.snake_k,
        snake_k_u4=cfg.snake_k_u4,
        gn_groups=cfg.gn_groups,
        dropout=cfg.dropout,
        mamba_dropout=cfg.mamba_dropout,
        ctx_inject=cfg.ctx_inject,
        safe_cudnn_u1=cfg.safe_cudnn_u1,
        ctx_halo_tokens=cfg.ctx_halo_tokens,
        ctx_halo_kernel=cfg.ctx_halo_kernel,
        nav_multiscale=cfg.nav_multiscale,
        pretrained_nav=cfg.pretrained_nav,
        use_residual_encoder=cfg.use_residual_encoder,
        encoder_blocks=cfg.encoder_blocks,
        norm_type=cfg.norm_type,
        use_sparse_encoder=cfg.use_sparse_encoder,
        sparse_threshold=cfg.sparse_threshold,
        snake_k_u3=cfg.snake_k_u3,
        snake_k_u2=cfg.snake_k_u2,
        snake_k_u1=cfg.snake_k_u1,
    ).to(dev)

    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)

    # Losses
    main_loss = MainTubularLoss().to(dev)
    aux_loss = AuxDiceCELoss(ce_weight_fg=20.0).to(dev)
    nav_loss_fn = NavigatorLoss(ce_weight_fg=cfg.nav_ce_fg_w, cldice_iters=5).to(dev)

    # Optimizer (created before load; safe to load state after)
    param_groups = build_param_groups(model, cfg.lr, cfg.weight_decay,
                                      freeze_nav=cfg.freeze_nav, freeze_local=cfg.freeze_local)
    for g in param_groups:
        g["lr"] = cfg.lr * float(g.get("lr_scale", 1.0))
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))

    # AMP scaler (fp16 only)
    amp_dtype = _autocast_dtype(cfg.amp_dtype)
    use_scaler = cfg.amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # State
    start_epoch = 1
    global_step = 0
    best_fullvol = -1.0
    best_patch = -1.0

    history_path = out_dir / "history.json"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []

    ckpt_dir = out_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = out_dir / "best.pt"

    resume_obj = None  # keep around to restore EMA correctly

    # -------------------------------------------------------------------------
    # Resume / init (IMPORTANT: do this BEFORE creating EMA)
    # -------------------------------------------------------------------------
    if args.resume:
        if not last_ckpt.exists():
            print("[resume] last.pt not found, skipping resume", flush=True)
        else:
            resume_obj = _load_checkpoint_obj(last_ckpt)
            state = _get_model_state_from_ckpt(resume_obj)
            model.load_state_dict(state, strict=True)
            if "opt" in resume_obj:
                opt.load_state_dict(resume_obj["opt"])
            if "scaler" in resume_obj and use_scaler:
                try:
                    scaler.load_state_dict(resume_obj["scaler"])
                except Exception:
                    pass
            start_epoch = int(resume_obj.get("epoch", 1)) + 1
            global_step = int(resume_obj.get("global_step", 0))
            best_fullvol = float(resume_obj.get("best_fullvol", -1.0))
            best_patch = float(resume_obj.get("best_patch", -1.0))
            if "history" in resume_obj and isinstance(resume_obj["history"], list):
                history = resume_obj["history"]
            print(f"[resume] loaded last.pt epoch={start_epoch-1} global_step={global_step}", flush=True)
    else:
        if args.init_ckpt is not None:
            ckpt_path = Path(args.init_ckpt)
            if not ckpt_path.exists():
                raise RuntimeError(f"--init-ckpt not found: {ckpt_path}")
            obj = _load_checkpoint_obj(ckpt_path)
            state = _get_model_state_from_ckpt(obj)

            if args.init_mode == "strict":
                model.load_state_dict(state, strict=True)
                print(f"[init] strict loaded: {ckpt_path}", flush=True)
            else:
                skipped, missing = load_state_dict_match_shapes(model, state)
                print(f"[init] match loaded: {ckpt_path}", flush=True)
                print(f"[init] skipped={len(skipped)} missing={len(missing)}", flush=True)
                if len(skipped) > 0:
                    print(f"[init] first skipped keys: {skipped[:8]}", flush=True)
                if len(missing) > 0:
                    print(f"[init] first missing keys: {missing[:8]}", flush=True)

    # Apply freeze AFTER loading weights (so it freezes the correct tensors)
    if cfg.freeze_nav:
        freeze_nav_(model)
        print("[nav] frozen navigator params — Phase 2 (decoder-only training)", flush=True)
    if cfg.freeze_local:
        freeze_local_(model)
        print("[local] frozen decoder params — Phase 1 (navigator-only training)", flush=True)

    # -------------------------------------------------------------------------
    # EMA (IMPORTANT: create AFTER init/resume so it is synced to loaded weights)
    # -------------------------------------------------------------------------
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema else None
    if ema is not None:
        if resume_obj is not None and "ema" in resume_obj and isinstance(resume_obj["ema"], dict):
            # move EMA to GPU so apply_to() works without device mismatch
            ema.shadow = {k: v.to(dev, non_blocking=True) for k, v in resume_obj["ema"].items()}
            print("[ema] restored from checkpoint", flush=True)
        else:
            print("[ema] initialized from current model weights", flush=True)

    # torch.compile AFTER checkpoint loading so key names are consistent
    if getattr(cfg, 'use_compile', False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[compile] torch.compile enabled (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"[compile] torch.compile failed, skipping: {e}", flush=True)

    # -------------------------------------------------------------------------
    # Training helpers
    # -------------------------------------------------------------------------
    def _compute_deep_sup_loss(out: Dict[str, torch.Tensor], y: torch.Tensor, w2, w3, w4,
                               soft_labels: bool = False,
                               skel_gt: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = main_loss(out["logits"], y, soft_labels=soft_labels, skel_gt=skel_gt)
        if "aux2" in out:
            if soft_labels:
                y2 = F.avg_pool3d(y.float().unsqueeze(1) if y.ndim == 4 else y.float(), 2, 2).squeeze(1)
            else:
                y2 = downsample_label_maxpool(y, 2)
            loss = loss + float(w2) * aux_loss(out["aux2"], y2)
        if "aux3" in out:
            if soft_labels:
                y3 = F.avg_pool3d(y.float().unsqueeze(1) if y.ndim == 4 else y.float(), 4, 4).squeeze(1)
            else:
                y3 = downsample_label_maxpool(y, 4)
            loss = loss + float(w3) * aux_loss(out["aux3"], y3)
        if "aux4" in out:
            if soft_labels:
                y4 = F.avg_pool3d(y.float().unsqueeze(1) if y.ndim == 4 else y.float(), 8, 8).squeeze(1)
            else:
                y4 = downsample_label_maxpool(y, 8)
            loss = loss + float(w4) * aux_loss(out["aux4"], y4)
        return loss

    # steps_per_epoch = actual optimizer steps per epoch, not iterations.
    # Each outer iteration processes patches_per_volume patches in micro-batches
    # of patch_micro_batch, accumulating over accum_steps before stepping.
    # optimizer steps per iteration = ceil(patches_per_volume / patch_micro_batch / accum_steps)
    _patches_per_iter = int(cfg.patches_per_volume)
    _mb = max(1, int(cfg.patch_micro_batch))
    _accum = max(1, int(cfg.accum_steps))
    _opt_steps_per_iter = max(1, (_patches_per_iter + _mb * _accum - 1) // (_mb * _accum))
    steps_per_epoch = int(cfg.epoch_size) * _opt_steps_per_iter
    total_steps = int(cfg.epochs) * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs) * steps_per_epoch
    rng = np.random.default_rng(cfg.seed + 123)

    def _set_optimizer_lr(step: int):
        for pg in opt.param_groups:
            base_lr = cfg.lr * float(pg.get("lr_scale", 1.0))
            if cfg.lr_restart:
                lr_now = cosine_with_warmup_restart(
                    step, total_steps, warmup_steps, base_lr,
                    n_cycles=cfg.lr_restart_cycles,
                )
            else:
                lr_now = cosine_with_warmup(step, total_steps, warmup_steps, base_lr)
            pg["lr"] = lr_now

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------
    model.train()
    torch.cuda.reset_peak_memory_stats()

    train_iter = iter(train_loader)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        nav_losses_epoch = []
        epoch_start_time = time.time()

        cl_w, ce_w, cl_iters, cl_down = loss_schedule(epoch, cfg.epochs)
        main_loss.set(ce_weight_fg=ce_w, cl_w=cl_w, cl_iters=cl_iters, cl_down=cl_down)
        ds_w2, ds_w3, ds_w4 = deep_sup_weights_dynamic(epoch, cfg.epochs, cfg.ds_w2, cfg.ds_w3, cfg.ds_w4)

        # Navigator loss weight warmup: 0 → nav_w over first 30% of training
        nav_warmup_end = max(1, int(0.30 * cfg.epochs))
        if epoch <= nav_warmup_end:
            nav_w_now = cfg.nav_w * (epoch / nav_warmup_end)
        else:
            nav_w_now = cfg.nav_w

        pbar = tqdm(range(cfg.epoch_size), desc=f"epoch {epoch}/{cfg.epochs}", dynamic_ncols=True)
        torch.cuda.reset_peak_memory_stats()

        for it_step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            img_cpu = batch["image"].float()
            lab_cpu = (batch["label"] > 0).to(torch.uint8)
            pools = batch["pools"]
            skel_cpu = batch.get("skeleton", None)  # [D,H,W] or None

            img_cpu = _pad_to_min_shape(img_cpu, cfg.patch_size, pad_value=0.0)
            lab_cpu = _pad_to_min_shape(lab_cpu, cfg.patch_size, pad_value=0)
            if skel_cpu is not None:
                skel_cpu = _pad_to_min_shape(skel_cpu, cfg.patch_size, pad_value=0)

            D, H, W = lab_cpu.shape

            # precompute nav tokens once per volume (chunked for large volumes)
            img_gpu_full = img_cpu.unsqueeze(0).to(dev, non_blocking=True)  # [1,1,D,H,W]
            if cfg.channels_last:
                img_gpu_full = img_gpu_full.to(memory_format=torch.channels_last_3d)

            tokens_full, thin_logits_full = chunked_nav_forward_train(
                model, img_gpu_full, cfg.patch_size,
                amp=cfg.amp, amp_dtype=amp_dtype,
            )

            centers = choose_centers(
                pools=pools,
                lab=lab_cpu,
                k=cfg.patches_per_volume,
                rng=rng,
                p_fg=cfg.p_fg,
                p_bg_boundary=cfg.p_bg_boundary,
                p_bg_hard=cfg.p_bg_hard,
                p_bg_easy=cfg.p_bg_easy,
                p_vessel_boundary=cfg.p_vessel_boundary,
            )

            starts = centers_to_starts_aligned(
                centers_zyx=centers,
                vol_dhw=(D, H, W),
                patch_dhw=cfg.patch_size,
                align=model.global_stride,
            )

            patch_losses = []
            patch_dices = []

            K = int(cfg.patches_per_volume)
            mb = int(cfg.patch_micro_batch)
            accum = int(cfg.accum_steps)

            opt.zero_grad(set_to_none=True)

    

            for k0 in range(0, K, mb):
                k1 = min(K, k0 + mb)

                xs, ys, sk = [], [], []
                for j in range(k0, k1):
                    z0, y0, x0 = [int(v) for v in starts[j]]
                    ip, lp = crop_patch_from_start(img_cpu, lab_cpu, (z0, y0, x0), cfg.patch_size)
                    xs.append(ip)
                    ys.append(lp)
                    if skel_cpu is not None:
                        pd, ph, pw = cfg.patch_size
                        sp = skel_cpu[z0:z0+pd, y0:y0+ph, x0:x0+pw].contiguous()
                        sk.append(sp)

                x = torch.stack(xs, dim=0).to(dev, non_blocking=True)
                y = torch.stack(ys, dim=0).to(dev, non_blocking=True).long()
                skel_patch = torch.stack(sk, dim=0).to(dev, non_blocking=True) if sk else None
                starts_t = torch.tensor(starts[k0:k1], device=dev, dtype=torch.long)  # [B,3]
                ctx = model.crop_ctx_tokens_for_patches(tokens_full, starts_t, cfg.patch_size)

                ctx = model.add_position_encoding(ctx, starts_t)

                if cfg.channels_last:
                    x = x.to(memory_format=torch.channels_last_3d)
                    ctx = ctx.to(memory_format=torch.channels_last_3d)

                is_soft = False
                with torch.no_grad():
                    x, y, ctx = random_flip_aligned(x, y, ctx, p=cfg.flip_prob)
                    x, y, ctx = aligned_geom_aug(
                        x, y, ctx,
                        rot_prob=cfg.rot_prob,
                        rot_deg=cfg.rot_deg,
                        elastic_prob=cfg.elastic_prob,
                        elastic_alpha=cfg.elastic_alpha,
                        elastic_coarse=cfg.elastic_coarse,
                    )
                    # Use STRONG augmentation if enabled, otherwise standard
                    if cfg.use_strong_aug:
                        x = gpu_intensity_aug_strong(x, p=cfg.intensity_aug_prob)
                    else:
                        x = gpu_intensity_aug(x, p=cfg.intensity_aug_prob)

                    # MixUp augmentation: KEEP soft labels (no hard conversion)
                    if cfg.mixup_prob > 0 and x.shape[0] > 1:
                        x, y_mixed, did_mixup = apply_mixup_batch(
                            x, y, prob=cfg.mixup_prob, alpha=cfg.mixup_alpha
                        )
                        if did_mixup:
                            y = y_mixed
                            is_soft = True
                            skel_patch = None  # can't use skeleton with MixUp soft labels

                # Forward pass — three modes:
                # freeze_local=True  → Phase 1: nav_loss ONLY, no decoder forward at all
                # freeze_nav=True    → Phase 2: seg_loss ONLY, ctx from frozen nav
                # neither frozen     → Joint: nav_loss + seg_loss (original v9 behaviour)
                with torch.amp.autocast("cuda", enabled=cfg.amp, dtype=amp_dtype):
                    if cfg.freeze_local:
                        # ---- PHASE 1: Navigator-only training ----
                        # Decoder is frozen and never called — pure vessel detection loss.
                        # No seg_loss backprop through navigator, no interference.
                        nav_loss_accum = torch.tensor(0.0, device=dev)
                        thin_loss_accum = torch.tensor(0.0, device=dev)
                        gs = model.global_stride
                        for bi in range(x.shape[0]):
                            xi = x[bi:bi+1]
                            live_tok, live_thin = model.nav_forward(xi)  # WITH gradients
                            yi = y[bi]
                            # Soft token label: avg_pool gives vessel fraction per token [0,1].
                            # Fixes 47% FP rate caused by max_pool exploding positives along vessel tubes.
                            token_label = F.avg_pool3d(
                                (yi > 0).float().unsqueeze(0).unsqueeze(0),
                                kernel_size=gs, stride=gs,
                            )
                            if token_label.shape[2:] != live_tok.shape[2:]:
                                token_label = F.interpolate(
                                    token_label, size=live_tok.shape[2:], mode="trilinear", align_corners=False)
                            nav_pred = model.nav_head(live_tok)
                            nav_loss_accum = nav_loss_accum + nav_loss_fn(
                                nav_pred, token_label,
                                nav_cl_w=cfg.nav_cl_w, nav_conn_w=cfg.nav_conn_w,
                            )
                            # Thin-vessel confidence head loss (BCE, weighted)
                            if live_thin is not None and getattr(cfg, 'thin_w', 0.0) > 0:
                                yi_4d = yi.unsqueeze(0).long()  # [1,D,H,W] — thin_vessel_proximity_label needs [B,D,H,W]
                                thin_lbl = thin_vessel_proximity_label(
                                    yi_4d, global_stride=gs
                                )
                                if thin_lbl.shape[2:] != live_thin.shape[2:]:
                                    thin_lbl = F.interpolate(thin_lbl, size=live_thin.shape[2:], mode="nearest")
                                thin_loss_accum = thin_loss_accum + F.binary_cross_entropy_with_logits(
                                    live_thin, thin_lbl,
                                    pos_weight=torch.tensor(10.0, device=dev),  # thin vessels are rare
                                )
                        nav_loss_accum = nav_loss_accum / max(x.shape[0], 1)
                        thin_loss_accum = thin_loss_accum / max(x.shape[0], 1)
                        # No decoder forward — seg_loss = 0
                        seg_loss = torch.tensor(0.0, device=dev)
                        out = {"logits": torch.zeros(
                            x.shape[0], 2, *x.shape[2:], device=dev, dtype=x.dtype)}

                    elif cfg.nav_w > 0 and not cfg.freeze_nav:
                        # ---- JOINT training (v9 style) ----
                        # nav_loss from live patch tokens + seg_loss from full-vol ctx tokens
                        nav_loss_accum = torch.tensor(0.0, device=dev)
                        thin_loss_accum = torch.tensor(0.0, device=dev)
                        gs = model.global_stride
                        for bi in range(x.shape[0]):
                            xi = x[bi:bi+1]
                            live_tok, live_thin = model.nav_forward(xi)  # WITH gradients
                            yi = y[bi]
                            # Soft token label: avg_pool gives vessel fraction per token [0,1].
                            token_label = F.avg_pool3d(
                                (yi > 0).float().unsqueeze(0).unsqueeze(0),
                                kernel_size=gs, stride=gs,
                            )
                            if token_label.shape[2:] != live_tok.shape[2:]:
                                token_label = F.interpolate(
                                    token_label, size=live_tok.shape[2:], mode="trilinear", align_corners=False)
                            nav_pred = model.nav_head(live_tok)
                            nav_loss_accum = nav_loss_accum + nav_loss_fn(
                                nav_pred, token_label,
                                nav_cl_w=cfg.nav_cl_w, nav_conn_w=cfg.nav_conn_w,
                            )
                            # Thin-vessel confidence head loss
                            if live_thin is not None and getattr(cfg, 'thin_w', 0.0) > 0:
                                yi_4d = yi.unsqueeze(0).long()  # [1,D,H,W] — thin_vessel_proximity_label needs [B,D,H,W]
                                thin_lbl = thin_vessel_proximity_label(
                                    yi_4d, global_stride=gs
                                )
                                if thin_lbl.shape[2:] != live_thin.shape[2:]:
                                    thin_lbl = F.interpolate(thin_lbl, size=live_thin.shape[2:], mode="nearest")
                                thin_loss_accum = thin_loss_accum + F.binary_cross_entropy_with_logits(
                                    live_thin, thin_lbl,
                                    pos_weight=torch.tensor(10.0, device=dev),
                                )
                        nav_loss_accum = nav_loss_accum / max(x.shape[0], 1)
                        thin_loss_accum = thin_loss_accum / max(x.shape[0], 1)
                        # Decoder uses full-vol ctx (not live patch tokens) — matches inference
                        out = model.local(x, ctx, grad_ckpt=cfg.grad_checkpoint)
                    else:
                        # ---- PHASE 2: Decoder-only (freeze_nav=True or nav_w=0) ----
                        out = model.local(x, ctx, grad_ckpt=cfg.grad_checkpoint)
                        nav_loss_accum = torch.tensor(0.0, device=dev)
                        thin_loss_accum = torch.tensor(0.0, device=dev)

                    seg_loss = _compute_deep_sup_loss(out, y, ds_w2, ds_w3, ds_w4,
                                                     soft_labels=is_soft,
                                                     skel_gt=skel_patch)

                    # Gate regularization
                    gate_loss = torch.tensor(0.0, device=dev)
                    if cfg.gate_reg_w > 0 and "attn_gate" in out:
                        y_hard = (y > 0.5).long() if is_soft else y
                        gate_loss = gate_reg_loss(out["attn_gate"], y_hard)

                    # Combine: seg + nav + gate + thin-vessel confidence head
                    thin_w = getattr(cfg, 'thin_w', 0.0)
                    loss = (seg_loss
                            + nav_w_now * nav_loss_accum
                            + cfg.gate_reg_w * gate_loss
                            + thin_w * thin_loss_accum)
                    loss_scaled = loss / float(accum)

                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                with torch.no_grad():
                    patch_losses.append(float(seg_loss.detach().item()))
                    # For dice metric, always use hard labels
                    y_hard = (y > 0.5).long() if is_soft else y
                    patch_dices.append(dice_fg_from_logits(out["logits"].detach(), y_hard.detach()))
                    # Log nav_loss in both Phase 1 (freeze_local) and joint training
                    if cfg.nav_w > 0 and (cfg.freeze_local or not cfg.freeze_nav):
                        nav_losses_epoch.append(float(nav_loss_accum.detach().item()))

                if ((k1 // mb) % accum == 0) or (k1 == K):
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        if use_scaler:
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))

                    _set_optimizer_lr(global_step)

                    if use_scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                    if ema is not None:
                        ema.update(model)

                    global_step += 1

            loss_mean = float(np.mean(patch_losses)) if patch_losses else 0.0
            dice_mean = float(np.mean(patch_dices)) if patch_dices else 0.0
            epoch_loss += loss_mean
            epoch_dice += dice_mean
            n_batches += 1

            postfix = {
                "loss": f"{loss_mean:.4f}",
                "dice": f"{dice_mean:.4f}",
                "cl_w": f"{cl_w:.2f}",
                "ce_w": f"{ce_w:.1f}",
                "cl_it": str(cl_iters),
                "cl_dn": str(cl_down),
                "lr": f"{opt.param_groups[0]['lr']:.2e}",
            }
            if nav_losses_epoch:
                postfix["nav"] = f"{nav_losses_epoch[-1]:.3f}"
            pbar.set_postfix(postfix)

            if cfg.log_mem_every > 0 and (global_step % cfg.log_mem_every == 0):
                log_gpu_mem(prefix=f" step={global_step}")

        epoch_loss /= max(1, n_batches)
        epoch_dice /= max(1, n_batches)

        # --- Validation (skip if fold_all) ---
        val_patch = {}
        val_full = {}
        do_full = False
        if not fold_all:
            val_patch = validate_patchwise(model, ema, val_loader, cfg, dev)
            do_full = (cfg.val_fullvol_every > 0) and (epoch % cfg.val_fullvol_every == 0)
            if do_full:
                val_full = validate_fullvolume(model, ema, val_items, cfg, dev)

        if "val_patch_fg_dice" in val_patch and val_patch["val_patch_fg_dice"] > best_patch:
            best_patch = float(val_patch["val_patch_fg_dice"])

        if "val_fullvol_fg_dice" in val_full and val_full["val_fullvol_fg_dice"] > best_fullvol:
            best_fullvol = float(val_full["val_fullvol_fg_dice"])
            payload_best = {
                "model": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_fullvol": best_fullvol,
                "best_patch": best_patch,
                "cfg": asdict(cfg),
            }
            if ema is not None:
                payload_best["ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
            save_checkpoint(best_ckpt, payload_best)
            print(f"[ckpt] saved BEST -> {best_ckpt} (fullvol_dice={best_fullvol:.4f})", flush=True)

        # fold_all: save best.pt periodically (every val-fullvol-every epochs)
        # since there's no validation metric, we use train_dice as pseudo-metric
        if fold_all and cfg.val_fullvol_every > 0 and (epoch % cfg.val_fullvol_every == 0):
            pseudo_dice = float(epoch_dice)
            if pseudo_dice > best_patch:
                best_patch = pseudo_dice
            payload_best = {
                "model": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_fullvol": best_fullvol,
                "best_patch": best_patch,
                "cfg": asdict(cfg),
            }
            if ema is not None:
                payload_best["ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
            save_checkpoint(best_ckpt, payload_best)
            print(f"[ckpt] saved BEST (fold_all) -> {best_ckpt} epoch={epoch} train_dice={pseudo_dice:.4f}", flush=True)

        payload_last = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_fullvol": best_fullvol,
            "best_patch": best_patch,
            "cfg": asdict(cfg),
        }
        if use_scaler:
            payload_last["scaler"] = scaler.state_dict()
        if ema is not None:
            payload_last["ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        payload_last["history"] = history
        save_checkpoint(last_ckpt, payload_last)

        epoch_elapsed = time.time() - epoch_start_time
        nav_loss_mean = float(np.mean(nav_losses_epoch)) if nav_losses_epoch else 0.0
        lr_now = float(opt.param_groups[0]["lr"])
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
        gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB

        row = {
            "epoch": epoch,
            "global_step": global_step,
            # --- training losses ---
            "train_loss": float(epoch_loss),
            "train_patch_dice": float(epoch_dice),
            "train_nav_loss": float(nav_loss_mean),
            # --- loss schedule state ---
            "cl_w": float(cl_w),
            "ce_w": float(ce_w),
            "cl_iters": int(cl_iters),
            "cl_down": int(cl_down),
            "nav_w": float(nav_w_now),
            # --- validation ---
            "val_patch_loss": float(val_patch.get("val_patch_loss", 0.0)),
            "val_patch_dice": float(val_patch.get("val_patch_fg_dice", 0.0)),
            # Navigator binary detection quality
            "nav_token_recall": float(val_patch.get("nav_token_recall", 0.0)),
            "nav_token_prec": float(val_patch.get("nav_token_prec", 0.0)),
            "nav_token_f1": float(val_patch.get("nav_token_f1", 0.0)),
            # Navigator calibration (gap = TP logit - FP logit, higher = more discriminative)
            "nav_logit_tp": float(val_patch.get("nav_logit_tp", 0.0)),
            "nav_logit_fp": float(val_patch.get("nav_logit_fp", 0.0)),
            "nav_logit_gap": float(val_patch.get("nav_logit_gap", 0.0)),
            # Cross-attention gate quality (gap = attn_vessel - attn_bg, higher = gate works)
            "attn_vessel": float(val_patch.get("attn_vessel", 0.0)),
            "attn_bg": float(val_patch.get("attn_bg", 0.0)),
            "attn_gap": float(val_patch.get("attn_gap", 0.0)),
            # --- best so far ---
            "best_patch": float(best_patch),
            "best_fullvol": float(best_fullvol),
            # --- optimizer / schedule ---
            "lr": lr_now,
            # --- timing & hardware ---
            "epoch_time_s": round(epoch_elapsed, 1),
            "gpu_mem_peak_gb": round(gpu_mem_peak, 2),
            "gpu_mem_reserved_gb": round(gpu_mem_reserved, 2),
        }
        row.update(val_full)
        history.append(row)
        history_path.write_text(json_dump_safe(history, indent=2))

        # nnU-Net-style per-epoch summary table
        fullvol_str = (
            f"  fullvol_dice : {row.get('val_fullvol_fg_dice', 0.0):.4f}\n"
            if do_full else ""
        )
        nav_diag_str = (
            f"  nav_token    : recall={row['nav_token_recall']:.3f}  prec={row['nav_token_prec']:.3f}  F1={row['nav_token_f1']:.3f}\n"
            f"  nav_calibr   : logit_TP={row['nav_logit_tp']:.2f}  logit_FP={row['nav_logit_fp']:.2f}  gap={row['nav_logit_gap']:.2f}\n"
            f"  attn_gate    : vessel={row['attn_vessel']:.3f}  bg={row['attn_bg']:.3f}  gap={row['attn_gap']:.3f}\n"
        )

        if cfg.freeze_local:
            # Phase 1: navigator-only — seg metrics are meaningless (decoder frozen)
            # Show only navigator quality metrics
            summary = (
                f"\n{'='*65}\n"
                f"  Epoch {epoch:>4d} / {cfg.epochs}  [PHASE 1 - navigator only]  ({epoch_elapsed/60:.1f} min)\n"
                f"{'='*65}\n"
                f"  nav_loss     : {nav_loss_mean:.4f}   (target: ≤ 0.15)\n"
                + nav_diag_str +
                f"  lr           : {lr_now:.2e}\n"
                f"  gpu_peak     : {gpu_mem_peak:.2f} GB (reserved {gpu_mem_reserved:.2f} GB)\n"
                f"  STOP WHEN    : prec > 0.55 AND stable for 10 epochs\n"
                f"{'='*65}"
            )
        else:
            summary = (
                f"\n{'='*65}\n"
                f"  Epoch {epoch:>4d} / {cfg.epochs}   ({epoch_elapsed/60:.1f} min)\n"
                f"{'='*65}\n"
                f"  train_loss   : {epoch_loss:.4f}\n"
                f"  train_dice   : {epoch_dice:.4f}\n"
                f"  nav_loss     : {nav_loss_mean:.4f}\n"
                f"  val_patch_dice: {row['val_patch_dice']:.4f}\n"
                + nav_diag_str
                + fullvol_str +
                f"  best_fullvol : {best_fullvol:.4f}   best_patch: {best_patch:.4f}\n"
                f"  lr           : {lr_now:.2e}   cl_w={cl_w:.2f}  ce_w={ce_w:.1f}\n"
                f"  gpu_peak     : {gpu_mem_peak:.2f} GB (reserved {gpu_mem_reserved:.2f} GB)\n"
                f"{'='*65}"
            )
        print(summary, flush=True)
        # Also write to a clean epoch log (easy to read without tqdm noise)
        with open(out_dir / "epoch_log.txt", "a") as _elf:
            _elf.write(summary + "\n")

        torch.cuda.empty_cache()

    print("[done] training complete", flush=True)

if __name__ == "__main__":
    main()
