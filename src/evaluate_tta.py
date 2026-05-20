#!/usr/bin/env python3
"""
Enhanced evaluation with TTA, Gaussian weighting, and post-processing.
Designed to push fullvol dice from ~82% to >90%.

Usage:
    python src/evaluate_tta.py \
        --checkpoint logs/navbrush_run_20260205_131932/best.pt \
        --cache-dir data/cache_augmented \
        --tta-flips \
        --gaussian-weight \
        --overlap 0.75 \
        --postprocess
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import from train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    NavBrushModel,
    read_cases_json,
    list_cases_from_cache,
    split_train_val_baseuids,
    _resolve_cached_path,
    _autocast_dtype,
    seed_everything,
    configure_cuda_for_a100,
    dice_fg_from_logits,
    EMA,
)


# -----------------------------------------------------------------------------
# Gaussian weighting for sliding window
# -----------------------------------------------------------------------------
def gaussian_weight_3d(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a 3D Gaussian importance map for patch weighting.
    Center of patch gets weight ~1.0, edges get lower weights.
    This reduces boundary artifacts in sliding window inference.
    """
    d, h, w = patch_size
    tmp = np.zeros((d, h, w), dtype=np.float64)

    center_d, center_h, center_w = d // 2, h // 2, w // 2
    sigma_d = d * sigma_scale
    sigma_h = h * sigma_scale
    sigma_w = w * sigma_scale

    for z in range(d):
        for y in range(h):
            for x in range(w):
                tmp[z, y, x] = np.exp(
                    -0.5 * (
                        ((z - center_d) / sigma_d) ** 2 +
                        ((y - center_h) / sigma_h) ** 2 +
                        ((x - center_w) / sigma_w) ** 2
                    )
                )

    # Normalize so max = 1.0
    tmp = tmp / tmp.max()
    # Clip minimum to avoid zero weights at corners
    tmp = np.clip(tmp, 1e-4, 1.0)

    return torch.from_numpy(tmp).float().to(device)


def gaussian_weight_3d_fast(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Vectorized version of Gaussian weight computation."""
    d, h, w = patch_size
    sigma_d = d * sigma_scale
    sigma_h = h * sigma_scale
    sigma_w = w * sigma_scale

    zz = torch.arange(d, dtype=torch.float64) - d // 2
    yy = torch.arange(h, dtype=torch.float64) - h // 2
    xx = torch.arange(w, dtype=torch.float64) - w // 2

    gz = torch.exp(-0.5 * (zz / sigma_d) ** 2)
    gy = torch.exp(-0.5 * (yy / sigma_h) ** 2)
    gx = torch.exp(-0.5 * (xx / sigma_w) ** 2)

    # Outer product for 3D
    g = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    g = g / g.max()
    g = g.clamp(min=1e-4)

    return g.float().to(device)


# -----------------------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------------------
def connected_components_3d(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Keep only connected components larger than min_size voxels.
    Uses scipy if available, otherwise returns input unchanged.
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("[warn] scipy not installed, skipping connected component filtering")
        return mask

    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask

    # Find component sizes
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # Keep only large components
    keep = np.zeros_like(mask, dtype=np.uint8)
    for i, size in enumerate(sizes, 1):
        if size >= min_size:
            keep[labeled == i] = 1

    return keep


def morphological_closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply morphological closing to fill small holes in vessels."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask

    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    closed = ndimage.binary_closing(mask, structure=struct, iterations=iterations)
    return closed.astype(np.uint8)


def postprocess_prediction(
    pred: np.ndarray,
    threshold: float = 0.5,
    min_component_size: int = 30,
    closing_iters: int = 1,
) -> np.ndarray:
    """Full post-processing pipeline."""
    # 1. Threshold
    binary = (pred > threshold).astype(np.uint8)

    # 2. Morphological closing (fill small gaps in vessels)
    if closing_iters > 0:
        binary = morphological_closing(binary, iterations=closing_iters)

    # 3. Remove small disconnected components
    if min_component_size > 0:
        binary = connected_components_3d(binary, min_size=min_component_size)

    return binary


# -----------------------------------------------------------------------------
# TTA flipping
# -----------------------------------------------------------------------------
FLIP_AXES = [
    (),       # no flip (original)
    (2,),     # flip D
    (3,),     # flip H
    (4,),     # flip W
    (2, 3),   # flip D+H
    (2, 4),   # flip D+W
    (3, 4),   # flip H+W
    (2, 3, 4),  # flip all
]


def flip_volume(vol: torch.Tensor, axes: Tuple[int, ...]) -> torch.Tensor:
    """Flip a 5D tensor [B,C,D,H,W] along specified spatial axes."""
    if not axes:
        return vol
    return torch.flip(vol, dims=list(axes))


# -----------------------------------------------------------------------------
# Volume padding for proper alignment
# -----------------------------------------------------------------------------
def pad_volume_for_inference(
    img: torch.Tensor,  # [1,1,D,H,W]
    patch_size: Tuple[int, int, int],
    global_stride: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Pad volume so that:
    1. Each dimension >= patch_size (so at least one full patch fits)
    2. Each dimension is divisible by global_stride (so navigator tokens align)
    Returns padded volume and original (D,H,W) for later cropping.
    """
    _, _, D, H, W = img.shape
    orig_shape = (D, H, W)
    pd, ph, pw = patch_size
    gs = max(1, int(global_stride))

    # Ensure at least patch_size in each dim, then round up to global_stride
    def _pad_dim(dim_size, patch_dim):
        target = max(dim_size, patch_dim)
        # Round up to nearest multiple of global_stride
        remainder = target % gs
        if remainder != 0:
            target += gs - remainder
        return target

    td = _pad_dim(D, pd)
    th = _pad_dim(H, ph)
    tw = _pad_dim(W, pw)

    if td == D and th == H and tw == W:
        return img, orig_shape

    # Symmetric padding
    pad_d = td - D
    pad_h = th - H
    pad_w = tw - W
    pad = (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2,
        pad_d // 2, pad_d - pad_d // 2,
    )
    img_padded = F.pad(img, pad, mode="constant", value=0.0)
    return img_padded, orig_shape


def unpad_volume(
    vol: torch.Tensor,  # [1,C,Dp,Hp,Wp] or [1,Dp,Hp,Wp]
    orig_shape: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Remove padding to restore original spatial dimensions."""
    D, H, W = orig_shape
    Dp, Hp, Wp = padded_shape

    d0 = (Dp - D) // 2
    h0 = (Hp - H) // 2
    w0 = (Wp - W) // 2

    if vol.ndim == 5:
        return vol[:, :, d0:d0+D, h0:h0+H, w0:w0+W]
    elif vol.ndim == 4:
        return vol[:, d0:d0+D, h0:h0+H, w0:w0+W]
    return vol


def _aligned_starts(dim_size: int, patch_dim: int, overlap: float, global_stride: int) -> List[int]:
    """Generate patch start coordinates aligned to global_stride."""
    gs = max(1, int(global_stride))
    stride = max(gs, int(patch_dim * (1.0 - overlap)))
    # Align stride to global_stride
    stride = (stride // gs) * gs
    if stride < gs:
        stride = gs

    starts = []
    pos = 0
    while pos + patch_dim <= dim_size:
        starts.append(pos)
        pos += stride

    # Add the last aligned position that covers the end
    last = dim_size - patch_dim
    last = (last // gs) * gs  # align
    last = max(0, last)
    if last not in starts:
        starts.append(last)

    return sorted(set(starts))


# -----------------------------------------------------------------------------
# Chunked navigator for large volumes
# -----------------------------------------------------------------------------
def _chunk_starts_1d(dim_size: int, chunk_size: int, stride: int, gs: int) -> List[int]:
    """Generate start positions for 1D chunking, aligned to global_stride."""
    starts = []
    pos = 0
    while pos + chunk_size <= dim_size:
        starts.append(pos)
        pos += stride
    # Ensure the end is covered
    last = max(0, dim_size - chunk_size)
    last = (last // gs) * gs
    if last not in starts:
        starts.append(last)
    return sorted(set(starts))


def chunked_nav_forward(
    model: NavBrushModel,
    img: torch.Tensor,   # [1,1,D,H,W] padded volume
    chunk_vox: int = 160,  # chunk size in voxels per dim (match training patch)
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Run navigator on volume, chunking in all 3 spatial dimensions.

    The Mamba SSM was trained on sequences from ~160³ patches (10x10x10 = 1000 tokens).
    For larger volumes, the token count explodes (e.g. 455x556x556 → 29x35x35 = 35K tokens),
    causing Mamba to produce poor context features.

    This function chunks the volume into overlapping 3D blocks matching the training
    patch size (160³ → 1000 tokens each), processes each independently through the
    navigator, and stitches the resulting token grids with overlap-averaging.

    Returns: context tokens [1,C,Dt,Ht,Wt]
    """
    _, _, D, H, W = img.shape
    gs = model.global_stride  # 16

    # If volume fits in a single chunk, run normally
    if D <= chunk_vox and H <= chunk_vox and W <= chunk_vox:
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            return model.nav_forward(img).float()

    # Chunk size per dimension: use training patch size, aligned to gs
    cd = min(D, max(gs * 4, (chunk_vox // gs) * gs))
    ch = min(H, max(gs * 4, (chunk_vox // gs) * gs))
    cw = min(W, max(gs * 4, (chunk_vox // gs) * gs))

    # 25% overlap in each dimension
    ov_d = max(gs, (cd // 4 // gs) * gs)
    ov_h = max(gs, (ch // 4 // gs) * gs)
    ov_w = max(gs, (cw // 4 // gs) * gs)

    stride_d = cd - ov_d
    stride_h = ch - ov_h
    stride_w = cw - ov_w

    # Full token grid dimensions
    Dt = D // gs
    Ht = H // gs
    Wt = W // gs

    # Get token channel dimension from model
    C = model.local.ctx_proj.in_channels
    device = img.device

    full_tokens = torch.zeros((1, C, Dt, Ht, Wt), device=device, dtype=torch.float32)
    token_count = torch.zeros((1, 1, Dt, Ht, Wt), device=device, dtype=torch.float32)

    # Generate chunk start positions
    d_starts = _chunk_starts_1d(D, cd, stride_d, gs)
    h_starts = _chunk_starts_1d(H, ch, stride_h, gs)
    w_starts = _chunk_starts_1d(W, cw, stride_w, gs)

    total_chunks = len(d_starts) * len(h_starts) * len(w_starts)
    n = 0

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                chunk = img[:, :, d0:d0+cd, h0:h0+ch, w0:w0+cw]

                with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
                    chunk_tokens = model.nav_forward(chunk).float()  # [1,C,td,th,tw]

                td, th, tw = chunk_tokens.shape[2:]

                # Map to positions in full token grid
                td0 = d0 // gs
                th0 = h0 // gs
                tw0 = w0 // gs
                td1 = min(td0 + td, Dt)
                th1 = min(th0 + th, Ht)
                tw1 = min(tw0 + tw, Wt)

                full_tokens[:, :, td0:td1, th0:th1, tw0:tw1] += \
                    chunk_tokens[:, :, :td1-td0, :th1-th0, :tw1-tw0]
                token_count[:, :, td0:td1, th0:th1, tw0:tw1] += 1.0

                del chunk, chunk_tokens
                n += 1
                if n % 20 == 0 or n == total_chunks:
                    torch.cuda.empty_cache()
                    print(f"  [chunked_nav] {n}/{total_chunks} chunks processed", end="\r")

    # Average overlapping regions
    full_tokens = full_tokens / token_count.clamp_min(1.0)
    torch.cuda.empty_cache()

    print(f"  [chunked_nav] {total_chunks} chunks, chunk={cd}x{ch}x{cw}, "
          f"token grid {Dt}x{Ht}x{Wt}")
    return full_tokens


# -----------------------------------------------------------------------------
# Main inference with TTA
# -----------------------------------------------------------------------------
@torch.inference_mode()
def infer_fullvolume_tta(
    model: NavBrushModel,
    img_gpu: torch.Tensor,  # [1,1,D,H,W]
    patch_size: Tuple[int, int, int],
    overlap: float = 0.75,
    gaussian_w: Optional[torch.Tensor] = None,
    tta_flips: bool = True,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    nav_chunk_vox: int = 160,  # chunk navigator to this size per dim (match training patch)
) -> torch.Tensor:
    """
    Full-volume inference with:
    - Chunked 3D navigator matching training patch size (fixes Mamba token mismatch)
    - Proper padding & alignment to global_stride
    - Sliding window with Gaussian weighting
    - Test-Time Augmentation (8 flip combinations)

    Returns: probability map [1,D_orig,H_orig,W_orig] (float32, 0-1)
    """
    device = img_gpu.device
    _, _, D_orig, H_orig, W_orig = img_gpu.shape
    pd, ph, pw = patch_size
    gs = model.global_stride  # typically 16

    flip_list = FLIP_AXES if tta_flips else [()]

    # Accumulator on ORIGINAL (unpadded) size
    prob_accum = torch.zeros((1, 1, D_orig, H_orig, W_orig), device=device, dtype=torch.float32)
    n_tta = 0

    for flip_axes in flip_list:
        # Flip input (on original size)
        img_flipped = flip_volume(img_gpu, flip_axes)

        # Pad for proper alignment AFTER flipping
        img_padded, orig_shape = pad_volume_for_inference(img_flipped, patch_size, gs)
        _, _, Dp, Hp, Wp = img_padded.shape

        # Navigator forward pass - chunked to match training patch size
        tokens = chunked_nav_forward(
            model, img_padded,
            chunk_vox=nav_chunk_vox,
            amp=amp, amp_dtype=amp_dtype,
        )

        # Generate aligned start coordinates
        zs = _aligned_starts(Dp, pd, overlap, gs)
        ys = _aligned_starts(Hp, ph, overlap, gs)
        xs = _aligned_starts(Wp, pw, overlap, gs)

        out_logits = torch.zeros((1, 2, Dp, Hp, Wp), device=device, dtype=torch.float32)
        out_weight = torch.zeros((1, 1, Dp, Hp, Wp), device=device, dtype=torch.float32)

        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    # All patches are guaranteed to be exactly patch_size (due to padding)
                    patch = img_padded[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw]
                    st = torch.tensor([[z0, y0, x0]], device=device, dtype=torch.long)
                    ctx = model.crop_ctx_tokens_for_patches(tokens, st, patch_size)
                    # Add absolute position encoding for region awareness
                    ctx = model.add_position_encoding(ctx, st)

                    with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
                        logits = model.local(patch, ctx, grad_ckpt=False)["logits"].float()

                    if gaussian_w is not None:
                        w = gaussian_w.unsqueeze(0).unsqueeze(0)  # [1,1,pd,ph,pw]
                        out_logits[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw] += logits * w
                        out_weight[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw] += w
                    else:
                        out_logits[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw] += logits
                        out_weight[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw] += 1.0

        # Average
        out_logits = out_logits / out_weight.clamp_min(1e-6)

        # Convert to probabilities
        probs = torch.softmax(out_logits, dim=1)[:, 1:2]  # [1,1,Dp,Hp,Wp]

        # Remove padding to get back to original size
        probs = unpad_volume(probs, orig_shape, (Dp, Hp, Wp))

        # Un-flip the probabilities
        probs = flip_volume(probs, flip_axes)

        prob_accum += probs
        n_tta += 1

        # Free padded volume memory
        del img_padded, tokens, out_logits, out_weight
        torch.cuda.empty_cache()

    # Average across all TTA augmentations
    prob_accum = prob_accum / n_tta

    return prob_accum.squeeze(1)  # [1,D_orig,H_orig,W_orig]


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------
def evaluate(
    model: NavBrushModel,
    val_items: List[Dict[str, str]],
    cache_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    overlap: float = 0.75,
    tta_flips: bool = True,
    gaussian_weight: bool = True,
    postprocess: bool = True,
    threshold: float = 0.5,
    min_component_size: int = 30,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_cases: int = 0,
    max_voxels: int = 50_000_000,
) -> Dict[str, Any]:
    """Evaluate model on validation set with all enhancements."""
    model.eval()

    # Create Gaussian weight map
    gw = None
    if gaussian_weight:
        gw = gaussian_weight_3d_fast(patch_size, sigma_scale=0.125, device=device)
        print(f"[eval] Gaussian weight map created: {gw.shape}")

    results = []
    dices_raw = []
    dices_postproc = []

    items = val_items[:max_cases] if max_cases > 0 else val_items

    for i, it in enumerate(items):
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], cache_dir, "image")
        lp = _resolve_cached_path(uid, it["label_pt"], cache_dir, "label")

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
        if vox > max_voxels:
            print(f"[eval] skip {uid}: vox={vox} > max={max_voxels}")
            continue

        print(f"[eval] [{i+1}/{len(items)}] {uid} shape={D}x{H}x{W} ({vox/1e6:.1f}M voxels)")
        t0 = time.time()

        img_gpu = img.unsqueeze(0).to(device, non_blocking=True)  # [1,1,D,H,W]
        lab_gpu = lab.unsqueeze(0).to(device, non_blocking=True)  # [1,D,H,W]

        # Run inference with TTA
        prob_map = infer_fullvolume_tta(
            model, img_gpu, patch_size,
            overlap=overlap,
            gaussian_w=gw,
            tta_flips=tta_flips,
            amp=amp,
            amp_dtype=amp_dtype,
        )

        # Compute raw dice (threshold = 0.5)
        pred_raw = (prob_map > threshold).long()
        raw_pred_for_dice = torch.zeros((1, 2, D, H, W), device=device)
        raw_pred_for_dice[:, 0] = 1.0 - prob_map
        raw_pred_for_dice[:, 1] = prob_map
        dice_raw = dice_fg_from_logits(raw_pred_for_dice, lab_gpu)
        dices_raw.append(dice_raw)

        # Post-processing
        if postprocess:
            pred_np = prob_map.squeeze().cpu().numpy()
            pred_pp = postprocess_prediction(
                pred_np,
                threshold=threshold,
                min_component_size=min_component_size,
                closing_iters=1,
            )
            pred_pp_t = torch.from_numpy(pred_pp).long().unsqueeze(0).to(device)
            tgt = lab_gpu.float()
            pred_f = pred_pp_t.float()
            eps = 1e-5
            inter = (pred_f * tgt).sum()
            den = pred_f.sum() + tgt.sum()
            dice_pp = float(((2 * inter + eps) / (den + eps)).item())
            dices_postproc.append(dice_pp)
        else:
            dice_pp = dice_raw
            dices_postproc.append(dice_pp)

        elapsed = time.time() - t0
        print(f"  -> dice_raw={dice_raw:.4f}  dice_postproc={dice_pp:.4f}  time={elapsed:.1f}s")

        results.append({
            "uid": uid,
            "shape": f"{D}x{H}x{W}",
            "dice_raw": dice_raw,
            "dice_postproc": dice_pp,
            "time_sec": elapsed,
        })

        # Free GPU memory
        del img_gpu, lab_gpu, prob_map
        torch.cuda.empty_cache()

    # Summary
    mean_raw = float(np.mean(dices_raw)) if dices_raw else 0.0
    mean_pp = float(np.mean(dices_postproc)) if dices_postproc else 0.0

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} cases evaluated")
    print(f"  Mean Dice (raw):           {mean_raw:.4f} ({mean_raw*100:.2f}%)")
    print(f"  Mean Dice (postprocessed):  {mean_pp:.4f} ({mean_pp*100:.2f}%)")
    print(f"  TTA: {'8-flip' if tta_flips else 'none'}")
    print(f"  Gaussian weight: {gaussian_weight}")
    print(f"  Overlap: {overlap}")
    print(f"  Threshold: {threshold}")
    print(f"{'='*60}")

    return {
        "mean_dice_raw": mean_raw,
        "mean_dice_postproc": mean_pp,
        "per_case": results,
    }


# -----------------------------------------------------------------------------
# Threshold search
# -----------------------------------------------------------------------------
@torch.inference_mode()
def search_best_threshold(
    model: NavBrushModel,
    val_items: List[Dict[str, str]],
    cache_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    overlap: float = 0.75,
    tta_flips: bool = True,
    gaussian_weight: bool = True,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_cases: int = 4,
    max_voxels: int = 50_000_000,
) -> float:
    """Search for the optimal threshold on the validation set."""
    model.eval()
    gw = gaussian_weight_3d_fast(patch_size, sigma_scale=0.125, device=device) if gaussian_weight else None

    # Collect probability maps
    all_probs = []
    all_labels = []

    items = val_items[:max_cases] if max_cases > 0 else val_items

    for it in items:
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], cache_dir, "image")
        lp = _resolve_cached_path(uid, it["label_pt"], cache_dir, "label")

        img = torch.load(ip, map_location="cpu").float()
        lab = torch.load(lp, map_location="cpu")
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.clamp(0.0, 1.0)
        if lab.ndim == 4 and lab.shape[0] == 1:
            lab = lab[0]
        lab = (lab > 0).numpy().astype(np.uint8)

        D, H, W = lab.shape
        if int(D * H * W) > max_voxels:
            continue

        img_gpu = img.unsqueeze(0).to(device)
        prob = infer_fullvolume_tta(
            model, img_gpu, patch_size, overlap=overlap,
            gaussian_w=gw, tta_flips=tta_flips, amp=amp, amp_dtype=amp_dtype,
        )
        all_probs.append(prob.squeeze().cpu().numpy())
        all_labels.append(lab)
        del img_gpu, prob
        torch.cuda.empty_cache()

    # Search thresholds
    best_t, best_dice = 0.5, 0.0
    print("\n[threshold search]")
    for t in np.arange(0.30, 0.70, 0.02):
        dices = []
        for p, l in zip(all_probs, all_labels):
            pred = (p > t).astype(np.float32)
            tgt = l.astype(np.float32)
            inter = (pred * tgt).sum()
            den = pred.sum() + tgt.sum()
            d = float((2 * inter + 1e-5) / (den + 1e-5))
            dices.append(d)
        mean_d = np.mean(dices)
        marker = " <-- BEST" if mean_d > best_dice else ""
        print(f"  threshold={t:.2f}  dice={mean_d:.4f}{marker}")
        if mean_d > best_dice:
            best_dice = mean_d
            best_t = t

    print(f"\n  Best threshold: {best_t:.2f} (dice={best_dice:.4f})")
    return best_t


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> NavBrushModel:
    """Load model from checkpoint, auto-detecting architecture parameters."""
    print(f"[load] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract model state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("cfg", ckpt.get("config", {}))
    else:
        state = ckpt
        config = {}

    # Detect model parameters from state dict keys
    # Detect base_ch from stem layer
    stem_key = "local.stem.conv.weight"
    if stem_key in state:
        base_ch = state[stem_key].shape[0]
    else:
        base_ch = config.get("base_ch", 28)

    tok_dim = config.get("tok_dim", 128)
    nav_dim = config.get("nav_dim", 48)
    nav_down = config.get("nav_down", 4)
    nav_token_stride = config.get("nav_token_stride", 4)
    nav_mamba_layers = config.get("nav_mamba_layers", 4)
    nav_mamba_axes = config.get("nav_mamba_axes", ("dhw", "hwd", "wdh"))
    if isinstance(nav_mamba_axes, list):
        nav_mamba_axes = tuple(nav_mamba_axes)
    snake_k = config.get("snake_k", 5)
    snake_k_u4 = config.get("snake_k_u4", 3)
    gn_groups = config.get("gn_groups", 16)
    dropout = config.get("dropout", 0.0)
    mamba_dropout = config.get("mamba_dropout", 0.0)
    ctx_inject = config.get("ctx_inject", "bottleneck,u4,u3,u2,u1")
    nav_multiscale = config.get("nav_multiscale", False)

    # Detect pretrained navigator from config or state dict keys
    pretrained_nav_used = config.get("pretrained_nav") is not None
    # Also detect by checking for VSSBlock3D keys in state dict
    if not pretrained_nav_used:
        pretrained_nav_used = any("vss_blocks" in k for k in state.keys())

    print(f"[load] base_ch={base_ch} tok_dim={tok_dim} nav_dim={nav_dim} pretrained_nav={pretrained_nav_used}")

    # For inference, pass pretrained_nav="dummy" to create correct architecture
    # (no actual weight loading from VMamba checkpoint - we load from our own checkpoint)
    pretrained_nav_arg = "__inference_mode__" if pretrained_nav_used else None

    model = NavBrushModel(
        in_ch=1,
        base_ch=base_ch,
        tok_dim=tok_dim,
        nav_dim=nav_dim,
        nav_down=nav_down,
        nav_token_stride=nav_token_stride,
        nav_mamba_layers=nav_mamba_layers,
        nav_mamba_axes=nav_mamba_axes,
        snake_k=snake_k,
        snake_k_u4=snake_k_u4,
        gn_groups=gn_groups,
        dropout=dropout,
        mamba_dropout=mamba_dropout,
        ctx_inject=ctx_inject,
        safe_cudnn_u1=True,
        ctx_halo_tokens=0,
        ctx_halo_kernel=3,
        nav_multiscale=nav_multiscale,
        pretrained_nav=pretrained_nav_arg,
    )

    # Check if EMA weights are available (preferred - they give better performance)
    ema_state = None
    if isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        print(f"[load] Found EMA weights ({len(ema_state)} params) - using EMA for inference")

    # Load raw model weights first
    model_sd = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
        else:
            skipped.append(k)

    model.load_state_dict(model_sd, strict=False)
    print(f"[load] Loaded {loaded} model parameters, skipped {len(skipped)}")

    # Apply EMA weights on top (overwrite with EMA shadow params)
    if ema_state is not None:
        ema_applied = 0
        for name, param in model.named_parameters():
            if name in ema_state and param.shape == ema_state[name].shape:
                param.data.copy_(ema_state[name])
                ema_applied += 1
        print(f"[load] Applied {ema_applied} EMA shadow parameters")

    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate with TTA + post-processing")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to best.pt checkpoint")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Path to data/cache_augmented")
    ap.add_argument("--patch-size", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--overlap", type=float, default=0.75,
                    help="Sliding window overlap (default: 0.75)")
    ap.add_argument("--tta-flips", action="store_true", default=True,
                    help="Enable 8-flip TTA (default: True)")
    ap.add_argument("--no-tta-flips", action="store_false", dest="tta_flips")
    ap.add_argument("--gaussian-weight", action="store_true", default=True,
                    help="Use Gaussian weighting (default: True)")
    ap.add_argument("--no-gaussian-weight", action="store_false", dest="gaussian_weight")
    ap.add_argument("--postprocess", action="store_true", default=True,
                    help="Apply post-processing (default: True)")
    ap.add_argument("--no-postprocess", action="store_false", dest="postprocess")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--search-threshold", action="store_true",
                    help="Search for best threshold on val set")
    ap.add_argument("--min-component-size", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--max-cases", type=int, default=0,
                    help="Max cases to evaluate (0 = all)")
    ap.add_argument("--max-voxels", type=int, default=200_000_000,
                    help="Max voxels per case (default: 200M, set 0 for unlimited)")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None,
                    help="Save results JSON to this path")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda:0")
    seed_everything(args.seed)
    configure_cuda_for_a100(safe_cudnn=True, allow_tf32=True, enable_cudnn=False)

    amp_dtype = _autocast_dtype(args.amp_dtype)
    patch_size = tuple(args.patch_size)

    # Load data
    items = list_cases_from_cache(args.cache_dir)
    _, val_items = split_train_val_baseuids(items, args.val_ratio, args.seed)
    print(f"[data] total={len(items)} val={len(val_items)}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)

    # Threshold search (optional)
    threshold = args.threshold
    if args.search_threshold:
        threshold = search_best_threshold(
            model, val_items, args.cache_dir, patch_size, device,
            overlap=args.overlap, tta_flips=args.tta_flips,
            gaussian_weight=args.gaussian_weight,
            amp=True, amp_dtype=amp_dtype,
            max_cases=min(4, len(val_items)),
            max_voxels=args.max_voxels,
        )

    # Full evaluation
    results = evaluate(
        model, val_items, args.cache_dir, patch_size, device,
        overlap=args.overlap,
        tta_flips=args.tta_flips,
        gaussian_weight=args.gaussian_weight,
        postprocess=args.postprocess,
        threshold=threshold,
        min_component_size=args.min_component_size,
        amp=True,
        amp_dtype=amp_dtype,
        max_cases=args.max_cases,
        max_voxels=args.max_voxels,
    )

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] Results written to {args.output}")


if __name__ == "__main__":
    main()
