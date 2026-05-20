from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------
# IO helpers (cache format)
# ----------------------------

def read_cases_json(cache_dir: Path) -> Dict[str, Any]:
    p = cache_dir / "cases.json"
    if not p.exists():
        raise RuntimeError(f"cases.json not found: {p}")
    js = json.loads(p.read_text())
    for k in ("uids", "images", "labels"):
        if k not in js:
            raise RuntimeError(f"cases.json must contain key '{k}'")
    return js


def write_cases_json(dst: Path, uids: List[str], images: Dict[str, str], labels: Dict[str, str]) -> None:
    out = {"uids": uids, "images": images, "labels": labels}
    (dst / "cases.json").write_text(json.dumps(out, indent=2))


def load_pt(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("image", "label", "vol", "mask", "data"):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        raise RuntimeError(f"Unsupported dict keys in {path}: {list(obj.keys())}")
    raise RuntimeError(f"Unsupported content type in {path}: {type(obj)}")


def save_pt(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)


def _standardize_pair(img: torch.Tensor, lab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standardize to:
    - img: float [1,D,H,W] in [0,1]
    - lab: uint8 [D,H,W] with values 0/1
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4 or img.shape[0] != 1:
        raise RuntimeError(f"Expected image [1,D,H,W] or [D,H,W], got {tuple(img.shape)}")
    img = img.float().clamp(0.0, 1.0)

    if lab.ndim == 4 and lab.shape[0] == 1:
        lab = lab[0]
    if lab.ndim != 3:
        raise RuntimeError(f"Expected label [D,H,W] or [1,D,H,W], got {tuple(lab.shape)}")
    lab = (lab > 0).to(torch.uint8)
    return img, lab


# ----------------------------
# Reproducibility
# ----------------------------

def seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


# ----------------------------
# Utility padding (reflect when possible)
# ----------------------------

def _safe_pad_3d(x5: torch.Tensor, pad: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """Pad 5D [B,C,D,H,W] safely.

    PyTorch reflect pad requires each spatial dim > padding.
    If that is violated, fall back to replicate padding.
    """
    # pad order for 5D: (W_left, W_right, H_left, H_right, D_left, D_right)
    wl, wr, hl, hr, dl, dr = pad
    _, _, D, H, W = x5.shape
    use_reflect = (W > max(wl, wr)) and (H > max(hl, hr)) and (D > max(dl, dr))
    mode = "reflect" if use_reflect else "replicate"
    return F.pad(x5, pad, mode=mode)


# ----------------------------
# Torch blur + sharpen
# ----------------------------

def _gaussian_1d_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = float(max(1e-6, sigma))
    radius = int(math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum().clamp_min(1e-12)
    return k  # [K]


def gaussian_blur_3d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img: [1,D,H,W] float"""
    if sigma <= 0:
        return img
    device, dtype = img.device, img.dtype
    k = _gaussian_1d_kernel(sigma, device, dtype)
    K = int(k.numel())

    x = img.unsqueeze(0)  # [B=1,C=1,D,H,W]

    # conv along W
    kw = k.view(1, 1, 1, 1, K)
    x = _safe_pad_3d(x, (K // 2, K // 2, 0, 0, 0, 0))
    x = F.conv3d(x, kw)
    # conv along H
    kh = k.view(1, 1, 1, K, 1)
    x = _safe_pad_3d(x, (0, 0, K // 2, K // 2, 0, 0))
    x = F.conv3d(x, kh)
    # conv along D
    kd = k.view(1, 1, K, 1, 1)
    x = _safe_pad_3d(x, (0, 0, 0, 0, K // 2, K // 2))
    x = F.conv3d(x, kd)

    return x[0]  # [1,D,H,W]


def unsharp_mask_3d(img: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    blur = gaussian_blur_3d(img, sigma=sigma)
    return img + float(amount) * (img - blur)


# ----------------------------
# CoLeTra (Disconnect-to-Connect) torch-only approximation
# ----------------------------

def binary_dilate_maxpool(mask: torch.Tensor, iters: int) -> torch.Tensor:
    """mask: [D,H,W] uint8/bool -> bool [D,H,W]"""
    if iters <= 0:
        return mask > 0
    x = (mask > 0).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    for _ in range(int(iters)):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x[0, 0] > 0.5


def gaussian_weight_patch(patch_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    patch_size = int(patch_size)
    if patch_size % 2 == 0:
        patch_size += 1
    r = patch_size // 2
    zz, yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        torch.arange(-r, r + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    dist2 = zz * zz + yy * yy + xx * xx
    sigma = float(max(1e-6, sigma))
    w = torch.exp(-0.5 * dist2 / (sigma * sigma))
    w = w / w.max().clamp_min(1e-12)
    return w  # [S,S,S]


def colettra_apply_torch(
    img: torch.Tensor,  # [1,D,H,W] float
    lab: torch.Tensor,  # [D,H,W] uint8
    rng: np.random.Generator,
    patch_size: int = 15,
    n_patches: int = 2,
    sigma: float = -1.0,  # -1 => auto patch_size/6
    dilate_iters: int = 3,
    bg_sigma: float = 2.0,
) -> torch.Tensor:
    """CoLeTra idea (Valverde et al. 2025):

    Create the *appearance* of disconnected tubular structures by removing/"inpainting"
    (here: blurred replacement) tubular regions, then blending local inpainted patches
    back into the original with a Gaussian kernel.

    Labels are left unchanged.
    """
    patch_size = int(patch_size)
    if patch_size % 2 == 0:
        patch_size += 1
    if sigma is None or float(sigma) <= 0:
        sigma = patch_size / 6.0
    sigma = float(sigma)

    fg = lab > 0
    coords = torch.nonzero(fg, as_tuple=False)  # [N,3] z,y,x
    if coords.numel() == 0 or int(n_patches) <= 0:
        return img

    D, H, W = lab.shape
    device = img.device
    dtype = img.dtype

    # "inpaint" tubular area: replace (dilated label) by blurred image
    dil = binary_dilate_maxpool(lab, iters=int(dilate_iters))
    blur = gaussian_blur_3d(img, sigma=float(bg_sigma)) if float(bg_sigma) > 0 else img
    x_inp = img.clone()
    x_inp[0][dil] = blur[0][dil]

    # sample centers uniformly from fg voxels
    n = int(min(int(n_patches), coords.shape[0]))
    picks = rng.choice(int(coords.shape[0]), size=n, replace=False)
    centers = coords[picks].to(torch.int64)

    w_full = gaussian_weight_patch(patch_size, sigma=sigma, device=device, dtype=torch.float32).to(dtype)
    r = patch_size // 2

    out = img.clone()
    for c in centers:
        cz, cy, cx = int(c[0]), int(c[1]), int(c[2])
        z0, z1 = max(0, cz - r), min(D, cz + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)

        wz0 = z0 - (cz - r)
        wy0 = y0 - (cy - r)
        wx0 = x0 - (cx - r)
        wz1 = wz0 + (z1 - z0)
        wy1 = wy0 + (y1 - y0)
        wx1 = wx0 + (x1 - x0)

        w = w_full[wz0:wz1, wy0:wy1, wx0:wx1]  # [dz,dy,dx]

        patch = out[0, z0:z1, y0:y1, x0:x1]
        pinp = x_inp[0, z0:z1, y0:y1, x0:x1]
        out[0, z0:z1, y0:y1, x0:x1] = w * pinp + (1.0 - w) * patch

    return out.clamp(0.0, 1.0)


# ----------------------------
# Geometric transforms
# ----------------------------

def apply_affine_3d(img: torch.Tensor, lab: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Affine with bilinear for image and nearest for label."""
    x = img.unsqueeze(0)  # [1,1,D,H,W]
    y = lab.unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
    grid = F.affine_grid(theta, size=x.shape, align_corners=True)
    x2 = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    y2 = F.grid_sample(y, grid, mode="nearest", padding_mode="border", align_corners=True)
    lab2 = (y2[0, 0] > 0.5).to(torch.uint8)
    return x2[0].to(dtype=img.dtype), lab2


def random_affine_theta(
    rng: np.random.Generator,
    rot_deg: float,
    scale_pct: float,
    shear_pct: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Returns theta [1,3,4] in normalized coordinates."""
    rx = math.radians(float(rng.uniform(-rot_deg, rot_deg)))
    ry = math.radians(float(rng.uniform(-rot_deg, rot_deg)))
    rz = math.radians(float(rng.uniform(-rot_deg, rot_deg)))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)

    smin, smax = 1.0 - float(scale_pct), 1.0 + float(scale_pct)
    sxv = float(rng.uniform(smin, smax))
    syv = float(rng.uniform(smin, smax))
    szv = float(rng.uniform(smin, smax))
    S = np.diag([sxv, syv, szv]).astype(np.float32)

    sh = float(shear_pct)
    shxy = float(rng.uniform(-sh, sh))
    shxz = float(rng.uniform(-sh, sh))
    shyx = float(rng.uniform(-sh, sh))
    shyz = float(rng.uniform(-sh, sh))
    shzx = float(rng.uniform(-sh, sh))
    shzy = float(rng.uniform(-sh, sh))
    Sh = np.array([[1, shxy, shxz], [shyx, 1, shyz], [shzx, shzy, 1]], dtype=np.float32)

    A = (R @ Sh @ S).astype(np.float32)
    theta = np.zeros((1, 3, 4), dtype=np.float32)
    theta[0, :, :3] = A
    return torch.from_numpy(theta).to(device=device, dtype=dtype)


def _make_base_grid(B: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    zz = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)  # [D,H,W,3]
    return grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B,D,H,W,3]


def mild_grid_distortion(
    img: torch.Tensor,
    lab: torch.Tensor,
    rng: np.random.Generator,
    alpha_vox: float = 2.0,
    coarse: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mild smooth displacement field (elastic-like warp)."""
    x = img.unsqueeze(0)  # [1,1,D,H,W]
    y = lab.unsqueeze(0).unsqueeze(0).float()
    _, _, D, H, W = x.shape
    device = x.device

    base = _make_base_grid(1, D, H, W, device=device, dtype=torch.float32)

    dc = max(2, D // int(coarse))
    hc = max(2, H // int(coarse))
    wc = max(2, W // int(coarse))

    disp_c = torch.from_numpy(rng.normal(0.0, 1.0, size=(1, 3, dc, hc, wc)).astype(np.float32)).to(device)
    disp = F.interpolate(disp_c, size=(D, H, W), mode="trilinear", align_corners=True)  # [1,3,D,H,W]

    # vox -> normalized coords
    sx = (2.0 / max(1, W - 1)) * float(alpha_vox)
    sy = (2.0 / max(1, H - 1)) * float(alpha_vox)
    sz = (2.0 / max(1, D - 1)) * float(alpha_vox)
    disp[:, 0] *= sx
    disp[:, 1] *= sy
    disp[:, 2] *= sz

    grid = base + disp.permute(0, 2, 3, 4, 1).contiguous()  # [1,D,H,W,3]

    x2 = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    y2 = F.grid_sample(y, grid, mode="nearest", padding_mode="border", align_corners=True)
    lab2 = (y2[0, 0] > 0.5).to(torch.uint8)
    return x2[0].to(dtype=img.dtype), lab2


def simulated_lowres(img: torch.Tensor, rng: np.random.Generator, min_scale: float = 0.5) -> torch.Tensor:
    """Downsample+upsample image only."""
    s = float(rng.uniform(min_scale, 1.0))
    if s >= 0.999:
        return img
    x = img.unsqueeze(0)  # [1,1,D,H,W]
    _, _, D, H, W = x.shape
    d2 = max(2, int(round(D * s)))
    h2 = max(2, int(round(H * s)))
    w2 = max(2, int(round(W * s)))
    x2 = F.interpolate(x, size=(d2, h2, w2), mode="trilinear", align_corners=True)
    x3 = F.interpolate(x2, size=(D, H, W), mode="trilinear", align_corners=True)
    return x3[0].clamp(0.0, 1.0)


# ----------------------------
# Intensity transforms
# ----------------------------

def intensity_augment(
    img: torch.Tensor,
    rng: np.random.Generator,
    p_noise: float,
    noise_std: float,
    p_smooth: float,
    smooth_sigma: float,
    p_shift_scale: float,
    scale_rng: Tuple[float, float],
    shift_rng: Tuple[float, float],
    p_contrast: float,
    contrast_rng: Tuple[float, float],
    p_gamma: float,
    gamma_rng: Tuple[float, float],
    p_sharpen: float,
    sharp_sigma: float,
    sharp_amount_rng: Tuple[float, float],
    p_invert: float,
) -> torch.Tensor:
    x = img

    if p_smooth > 0 and rng.random() < p_smooth:
        sigma = float(smooth_sigma) * float(rng.uniform(0.75, 1.25))
        x = gaussian_blur_3d(x, sigma=sigma)

    if p_shift_scale > 0 and rng.random() < p_shift_scale:
        sc = float(rng.uniform(scale_rng[0], scale_rng[1]))
        sh = float(rng.uniform(shift_rng[0], shift_rng[1]))
        x = x * sc + sh

    if p_contrast > 0 and rng.random() < p_contrast:
        cf = float(rng.uniform(contrast_rng[0], contrast_rng[1]))
        mean = float(x.mean().item())
        x = (x - mean) * cf + mean

    if p_gamma > 0 and rng.random() < p_gamma:
        g = float(rng.uniform(gamma_rng[0], gamma_rng[1]))
        # avoid 0^gamma issues
        x = torch.pow(x.clamp(0.01, 0.99), g)

    if p_sharpen > 0 and rng.random() < p_sharpen:
        amt = float(rng.uniform(sharp_amount_rng[0], sharp_amount_rng[1]))
        sig = float(sharp_sigma) * float(rng.uniform(0.75, 1.25))
        x = unsharp_mask_3d(x, sigma=sig, amount=amt)

    if p_noise > 0 and rng.random() < p_noise:
        n = torch.from_numpy(rng.normal(0.0, noise_std, size=tuple(x.shape)).astype(np.float32)).to(
            device=x.device, dtype=x.dtype
        )
        x = x + n

    if p_invert > 0 and rng.random() < p_invert:
        x = 1.0 - x

    return x.clamp(0.0, 1.0)


# ----------------------------
# Full pipeline per sample
# ----------------------------

def augment_one(
    img: torch.Tensor,
    lab: torch.Tensor,
    rng: np.random.Generator,
    # flips
    p_flip: float,
    # affine
    p_affine: float,
    rot_deg: float,
    scale_pct: float,
    shear_pct: float,
    # grid distortion
    p_grid: float,
    grid_alpha_vox: float,
    grid_coarse: int,
    # lowres
    p_lowres: float,
    lowres_min_scale: float,
    # colettra
    p_colettra: float,
    colettra_patch: int,
    colettra_npatch: int,
    colettra_sigma: float,
    colettra_dilate: int,
    colettra_bg_sigma: float,
    # intensity
    p_noise: float,
    noise_std: float,
    p_smooth: float,
    smooth_sigma: float,
    p_shift_scale: float,
    scale_rng: Tuple[float, float],
    shift_rng: Tuple[float, float],
    p_contrast: float,
    contrast_rng: Tuple[float, float],
    p_gamma: float,
    gamma_rng: Tuple[float, float],
    p_sharpen: float,
    sharp_sigma: float,
    sharp_amount_rng: Tuple[float, float],
    p_invert: float,
    # schedule
    elastic_every: int,
    aug_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = img, lab

    # flips (D/H/W)
    if p_flip > 0:
        if rng.random() < p_flip:
            x = torch.flip(x, dims=(1,))
            y = torch.flip(y, dims=(0,))
        if rng.random() < p_flip:
            x = torch.flip(x, dims=(2,))
            y = torch.flip(y, dims=(1,))
        if rng.random() < p_flip:
            x = torch.flip(x, dims=(3,))
            y = torch.flip(y, dims=(2,))

    # affine
    if p_affine > 0 and rng.random() < p_affine:
        theta = random_affine_theta(
            rng=rng,
            rot_deg=rot_deg,
            scale_pct=scale_pct,
            shear_pct=shear_pct,
            device=x.device,
            dtype=torch.float32,
        )
        x, y = apply_affine_3d(x, y, theta)

    # elastic/grid distortion (optional schedule)
    do_elastic = True
    if elastic_every and int(elastic_every) > 1:
        do_elastic = (aug_index % int(elastic_every) == 0)
    if do_elastic and p_grid > 0 and rng.random() < p_grid:
        x, y = mild_grid_distortion(x, y, rng=rng, alpha_vox=grid_alpha_vox, coarse=grid_coarse)

    # low-res simulation
    if p_lowres > 0 and rng.random() < p_lowres:
        x = simulated_lowres(x, rng=rng, min_scale=lowres_min_scale)

    # CoLeTra
    if p_colettra > 0 and rng.random() < p_colettra:
        x = colettra_apply_torch(
            x,
            y,
            rng=rng,
            patch_size=colettra_patch,
            n_patches=colettra_npatch,
            sigma=colettra_sigma,
            dilate_iters=colettra_dilate,
            bg_sigma=colettra_bg_sigma,
        )

    # intensity
    x = intensity_augment(
        x,
        rng=rng,
        p_noise=p_noise,
        noise_std=noise_std,
        p_smooth=p_smooth,
        smooth_sigma=smooth_sigma,
        p_shift_scale=p_shift_scale,
        scale_rng=scale_rng,
        shift_rng=shift_rng,
        p_contrast=p_contrast,
        contrast_rng=contrast_rng,
        p_gamma=p_gamma,
        gamma_rng=gamma_rng,
        p_sharpen=p_sharpen,
        sharp_sigma=sharp_sigma,
        sharp_amount_rng=sharp_amount_rng,
        p_invert=p_invert,
    )

    return x.contiguous(), y.contiguous()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        "Offline cache augmentation -> new cache with updated cases.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--src", type=Path, required=True, help="source cache dir containing cases.json + images/ + labels/")
    ap.add_argument("--dst", type=Path, required=True, help="destination cache dir")
    ap.add_argument("--k", type=int, default=3, help="number of augmented variants per original UID")
    ap.add_argument("--copy-original", action="store_true", help="also copy original samples into dst cache")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0, cuda:1, ...")
    ap.add_argument("--limit", type=int, default=-1, help="process only first N uids for quick test")
    ap.add_argument("--save-fp16", action="store_true", help="store images as float16 to save disk (labels remain uint8)")

    # ---- geometric ----
    ap.add_argument("--p-flip", type=float, default=0.5)

    # Rotation/affine (alias rot-prob)
    ap.add_argument("--p-affine", type=float, default=0.15, help="probability of applying random affine")
    ap.add_argument("--rot-prob", type=float, default=None, help="alias for --p-affine")
    ap.add_argument("--rot-deg", type=float, default=15.0)
    ap.add_argument("--scale-pct", type=float, default=0.10)
    ap.add_argument("--shear-pct", type=float, default=0.10)

    # Elastic/grid (aliases elastic-*)
    ap.add_argument("--p-grid", type=float, default=0.08, help="probability of applying mild grid/elastic distortion")
    ap.add_argument("--elastic-prob", type=float, default=None, help="alias for --p-grid")
    ap.add_argument("--grid-alpha-vox", type=float, default=2.5, help="displacement magnitude in voxels (approx)")
    ap.add_argument("--elastic-alpha", type=float, default=None, help="alias for --grid-alpha-vox")
    ap.add_argument("--grid-coarse", type=int, default=6)
    ap.add_argument("--elastic-every", type=int, default=2, help="apply elastic only to every N-th augmented variant")

    ap.add_argument("--p-lowres", type=float, default=0.15)
    ap.add_argument("--lowres-min-scale", type=float, default=0.55)

    # ---- CoLeTra ----
    ap.add_argument("--p-colettra", type=float, default=0.35)
    ap.add_argument("--colettra-patch", type=int, default=15)
    ap.add_argument("--colettra-npatch", type=int, default=2)
    ap.add_argument("--colettra-sigma", type=float, default=-1.0, help="-1 => auto (patch/6)")
    ap.add_argument("--colettra-dilate", type=int, default=3)
    ap.add_argument("--colettra-bg-sigma", type=float, default=2.0)

    # ---- intensity ----
    ap.add_argument("--p-noise", type=float, default=0.25)
    ap.add_argument("--noise-std", type=float, default=0.012)

    ap.add_argument("--p-smooth", type=float, default=0.20)
    ap.add_argument("--smooth-sigma", type=float, default=0.8)

    ap.add_argument("--p-shift-scale", type=float, default=0.35)
    ap.add_argument("--scale-min", type=float, default=0.90)
    ap.add_argument("--scale-max", type=float, default=1.10)
    ap.add_argument("--shift-min", type=float, default=-0.05)
    ap.add_argument("--shift-max", type=float, default=0.05)

    ap.add_argument("--p-contrast", type=float, default=0.20)
    ap.add_argument("--contrast-min", type=float, default=0.85)
    ap.add_argument("--contrast-max", type=float, default=1.20)

    ap.add_argument("--p-gamma", type=float, default=0.20)
    ap.add_argument("--gamma-min", type=float, default=0.8)
    ap.add_argument("--gamma-max", type=float, default=1.2)

    ap.add_argument("--p-sharpen", type=float, default=0.15)
    ap.add_argument("--sharp-sigma", type=float, default=0.8)
    ap.add_argument("--sharp-amt-min", type=float, default=0.3)
    ap.add_argument("--sharp-amt-max", type=float, default=0.8)

    ap.add_argument("--p-invert", type=float, default=0.05)

    args = ap.parse_args()

    # Apply alias overrides (if provided)
    if args.rot_prob is not None:
        args.p_affine = float(args.rot_prob)
    if args.elastic_prob is not None:
        args.p_grid = float(args.elastic_prob)
    if args.elastic_alpha is not None:
        args.grid_alpha_vox = float(args.elastic_alpha)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng_master = seed_everything(int(args.seed))

    js = read_cases_json(src)
    uids = list(js["uids"])
    if args.limit and int(args.limit) > 0:
        uids = uids[: int(args.limit)]

    out_uids: List[str] = []
    out_images: Dict[str, str] = {}
    out_labels: Dict[str, str] = {}

    def dst_img_path(uid: str) -> Path:
        return dst / "images" / f"{uid}.pt"

    def dst_lab_path(uid: str) -> Path:
        return dst / "labels" / f"{uid}.pt"

    # Copy selected meta files if present
    for meta_name in ("meta.json", "dataset.json", "dataset_fingerprint.json"):
        sp = src / meta_name
        if sp.exists():
            (dst / meta_name).write_text(sp.read_text())

    print(f"[augment] src={src}")
    print(f"[augment] dst={dst}")
    print(f"[augment] uids={len(uids)}  k={args.k}  copy_original={bool(args.copy_original)}  device={device}")

    torch.set_grad_enabled(False)

    for i, uid in enumerate(uids, start=1):
        ip = Path(js["images"].get(uid, ""))
        lp = Path(js["labels"].get(uid, ""))
        if not ip.exists():
            ip = src / "images" / f"{uid}.pt"
        if not lp.exists():
            lp = src / "labels" / f"{uid}.pt"
        if not ip.exists() or not lp.exists():
            raise RuntimeError(f"Missing files for uid={uid}: image={ip} label={lp}")

        img = load_pt(ip)
        lab = load_pt(lp)
        img, lab = _standardize_pair(img, lab)

        # move to device (one case at a time)
        img = img.to(device=device, non_blocking=False)
        lab = lab.to(device=device, non_blocking=False)

        # ---- copy original ----
        if args.copy_original:
            ouid = uid
            x0 = img
            if args.save_fp16:
                x0 = x0.half()
            save_pt(dst_img_path(ouid), x0.detach().cpu())
            save_pt(dst_lab_path(ouid), lab.detach().cpu())
            out_uids.append(ouid)
            out_images[ouid] = str(dst_img_path(ouid))
            out_labels[ouid] = str(dst_lab_path(ouid))

        # ---- K variants ----
        for k in range(1, int(args.k) + 1):
            auid = f"{uid}_aug{k:02d}"

            # deterministic per-sample seed (stable across runs)
            sub_seed = int(args.seed) + i * 10007 + k * 917
            sub_rng = np.random.default_rng(sub_seed)

            x, y = augment_one(
                img=img,
                lab=lab,
                rng=sub_rng,
                p_flip=float(args.p_flip),
                p_affine=float(args.p_affine),
                rot_deg=float(args.rot_deg),
                scale_pct=float(args.scale_pct),
                shear_pct=float(args.shear_pct),
                p_grid=float(args.p_grid),
                grid_alpha_vox=float(args.grid_alpha_vox),
                grid_coarse=int(args.grid_coarse),
                p_lowres=float(args.p_lowres),
                lowres_min_scale=float(args.lowres_min_scale),
                p_colettra=float(args.p_colettra),
                colettra_patch=int(args.colettra_patch),
                colettra_npatch=int(args.colettra_npatch),
                colettra_sigma=float(args.colettra_sigma),
                colettra_dilate=int(args.colettra_dilate),
                colettra_bg_sigma=float(args.colettra_bg_sigma),
                p_noise=float(args.p_noise),
                noise_std=float(args.noise_std),
                p_smooth=float(args.p_smooth),
                smooth_sigma=float(args.smooth_sigma),
                p_shift_scale=float(args.p_shift_scale),
                scale_rng=(float(args.scale_min), float(args.scale_max)),
                shift_rng=(float(args.shift_min), float(args.shift_max)),
                p_contrast=float(args.p_contrast),
                contrast_rng=(float(args.contrast_min), float(args.contrast_max)),
                p_gamma=float(args.p_gamma),
                gamma_rng=(float(args.gamma_min), float(args.gamma_max)),
                p_sharpen=float(args.p_sharpen),
                sharp_sigma=float(args.sharp_sigma),
                sharp_amount_rng=(float(args.sharp_amt_min), float(args.sharp_amt_max)),
                p_invert=float(args.p_invert),
                elastic_every=int(args.elastic_every),
                aug_index=k,
            )

            if args.save_fp16:
                x = x.half()

            save_pt(dst_img_path(auid), x.detach().cpu())
            save_pt(dst_lab_path(auid), y.detach().cpu())
            out_uids.append(auid)
            out_images[auid] = str(dst_img_path(auid))
            out_labels[auid] = str(dst_lab_path(auid))

        # free per-case tensors
        del img, lab
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        if (i % 10 == 0) or (i == len(uids)):
            print(
                f"  [{i}/{len(uids)}] uid={uid} -> wrote {('orig + ' if args.copy_original else '')}{args.k} aug variants",
                flush=True,
            )

    write_cases_json(dst, out_uids, out_images, out_labels)
    print("\n✅ Done.")
    print(f"✅ New cache: {dst}")
    print(f"✅ cases.json entries: {len(out_uids)}")


if __name__ == "__main__":
    main()
