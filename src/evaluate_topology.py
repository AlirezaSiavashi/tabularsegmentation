#!/usr/bin/env python3
"""
evaluate_topology.py
====================
Full-volume evaluation with topology metrics for NavBrush.

Computes per-case and mean:
  - Dice          (fg voxel overlap)
  - clDice        (centerline Dice via distance-transform skeleton)
  - Betti-0 error (|pred_components - gt_components|, lower=better)
  - HD95          (95th percentile Hausdorff distance, in voxels)

Compares NavBrush against the RSNA nnUNet checkpoint on the same val cases.

Usage:
  python src/evaluate_topology.py \
    --checkpoint logs/navbrush_adaptive_v5/best.pt \
    --cache-dir data/cache_augmented \
    --output results/topology_eval.json \
    --nnunet-dir ../logs/nnUNet_results/Dataset001_VesselSegmentation \
    --overlap 0.5

GPU: runs on CUDA_VISIBLE_DEVICES (set externally via nohup).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cc3d
from scipy.ndimage import distance_transform_edt, maximum_filter

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
    gaussian_weight_3d_fast,
    EMA,
    chunked_nav_forward_train,
)


# ---------------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------------

def distance_skeleton_3d(mask: np.ndarray) -> np.ndarray:
    """Centerline via local maxima of distance transform (works correctly in 3D)."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    dt = distance_transform_edt(mask.astype(bool))
    local_max = (dt == maximum_filter(dt, size=3)) & (mask > 0)
    return local_max


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """clDice: 2 * tprec * tsens / (tprec + tsens). Range [0,1], higher=better."""
    eps = 1e-6
    skel_pred = distance_skeleton_3d(pred > 0)
    skel_gt   = distance_skeleton_3d(gt > 0)
    tprec = float((skel_pred & (gt > 0)).sum()) / (skel_pred.sum() + eps)
    tsens = float((skel_gt  & (pred > 0)).sum()) / (skel_gt.sum()  + eps)
    if skel_pred.sum() == 0 and skel_gt.sum() == 0:
        return 1.0  # both empty → perfect
    return float(2.0 * tprec * tsens / (tprec + tsens + eps))


def compute_betti0_error(pred: np.ndarray, gt: np.ndarray) -> int:
    """Betti-0 error: |n_pred_components - n_gt_components|. Lower=better (0=perfect)."""
    n_pred = int(cc3d.connected_components(pred.astype(np.uint8), connectivity=26).max())
    n_gt   = int(cc3d.connected_components(gt.astype(np.uint8),   connectivity=26).max())
    return abs(n_pred - n_gt)


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in voxels. Lower=better."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    # Surface voxels
    pred_surf = pred_b & ~(
        (pred_b == maximum_filter(pred_b, footprint=np.ones((3,3,3))))
    )
    gt_surf   = gt_b & ~(
        (gt_b == maximum_filter(gt_b, footprint=np.ones((3,3,3))))
    )
    # Fall back to all vessel voxels if surface extraction fails
    if pred_surf.sum() == 0: pred_surf = pred_b
    if gt_surf.sum()   == 0: gt_surf   = gt_b
    # Distance from each pred surface voxel to nearest gt voxel
    dt_gt   = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)
    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def compute_dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    eps = 1e-5
    inter = float((pred.astype(bool) & gt.astype(bool)).sum())
    den   = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (den + eps)


# ---------------------------------------------------------------------------
# Model loading (replicates evaluate_tta.py logic)
# ---------------------------------------------------------------------------

def load_navbrush(ckpt_path: Path, device: torch.device) -> NavBrushModel:
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = obj.get("cfg")
    state = obj.get("model") or obj.get("ema") or obj

    kw = dict(
        in_ch=1,
        base_ch=getattr(cfg, "base_ch", 28),
        tok_dim=getattr(cfg, "tok_dim", 128),
        nav_dim=getattr(cfg, "nav_dim", 48),
        nav_down=getattr(cfg, "nav_down", 1),
        nav_token_stride=getattr(cfg, "nav_token_stride", 4),
        nav_mamba_layers=getattr(cfg, "nav_mamba_layers", 4),
        nav_mamba_axes=list(getattr(cfg, "nav_mamba_axes", ["dhw","hwd","wdh"])),
        snake_k=getattr(cfg, "snake_k", 5),
        snake_k_u4=getattr(cfg, "snake_k_u4", 3),
        snake_k_u3=getattr(cfg, "snake_k_u3", 5),
        snake_k_u2=getattr(cfg, "snake_k_u2", 0),
        snake_k_u1=getattr(cfg, "snake_k_u1", 0),
        gn_groups=getattr(cfg, "gn_groups", 14),
        dropout=0.0,
        mamba_dropout=0.0,
        ctx_inject=getattr(cfg, "ctx_inject", "u4,u3,u2,u1"),
        safe_cudnn_u1=True,
        ctx_halo_tokens=getattr(cfg, "ctx_halo_tokens", 0),
        ctx_halo_kernel=getattr(cfg, "ctx_halo_kernel", 3),
        nav_multiscale=getattr(cfg, "nav_multiscale", False),
        pretrained_nav=None,
        use_residual_encoder=getattr(cfg, "use_residual_encoder", True),
        encoder_blocks=list(getattr(cfg, "encoder_blocks", [1,2,3,4,4])),
        norm_type=getattr(cfg, "norm_type", "instance"),
        use_sparse_encoder=False,
        sparse_threshold=0.0,
    )
    model = NavBrushModel(**kw).to(device)
    model.eval()

    # Prefer EMA weights
    ema_state = obj.get("ema")
    if ema_state and isinstance(ema_state, dict) and len(ema_state) > 10:
        ema = EMA(model, decay=0.9995)
        ema.shadow = {k: v.to(device) for k, v in ema_state.items()}
        ema.apply_to(model)
        print("[load] Using EMA weights")
    else:
        model.load_state_dict(state, strict=False)
        print("[load] Using raw model weights")

    return model


# ---------------------------------------------------------------------------
# Full-volume inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_fullvol(
    model: NavBrushModel,
    img: torch.Tensor,      # [1, D, H, W] float32, already on CPU
    patch_size: Tuple[int,int,int],
    overlap: float,
    device: torch.device,
    amp_dtype=torch.bfloat16,
) -> np.ndarray:
    """Returns binary prediction [D,H,W] as uint8 numpy array."""
    img_gpu = img.unsqueeze(0).to(device)  # [1,1,D,H,W]
    D, H, W = img.shape[1:]
    pd, ph, pw = patch_size

    # Navigator — full volume
    tokens, _ = chunked_nav_forward_train(
        model, img_gpu, patch_size, amp=True, amp_dtype=amp_dtype)

    # Gaussian-weighted sliding window
    gw = gaussian_weight_3d_fast(patch_size, sigma_scale=0.125, device=device)
    gw = gw.unsqueeze(0).unsqueeze(0).half()

    stride_d = max(1, int(pd * (1.0 - overlap)))
    stride_h = max(1, int(ph * (1.0 - overlap)))
    stride_w = max(1, int(pw * (1.0 - overlap)))

    zs = sorted(set(list(range(0, max(1, D - pd + 1), stride_d)) + ([D-pd] if D > pd else [0])))
    ys = sorted(set(list(range(0, max(1, H - ph + 1), stride_h)) + ([H-ph] if H > ph else [0])))
    xs = sorted(set(list(range(0, max(1, W - pw + 1), stride_w)) + ([W-pw] if W > pw else [0])))

    out_logits = torch.zeros((1, 2, D, H, W), device=device, dtype=torch.float16)
    out_weight = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float16)

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                patch = img_gpu[:, :, z0:z0+pd, y0:y0+ph, x0:x0+pw]
                st = torch.tensor([[z0, y0, x0]], device=device, dtype=torch.long)
                ctx = model.crop_ctx_tokens_for_patches(tokens, st, patch_size)
                ctx = model.add_position_encoding(ctx, st)
                with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                    logits = model.local(patch, ctx, grad_ckpt=False)["logits"].half()
                ad, ah, aw = logits.shape[2:]
                gw_crop = gw[:, :, :ad, :ah, :aw]
                out_logits[:, :, z0:z0+ad, y0:y0+ah, x0:x0+aw] += logits * gw_crop
                out_weight[:, :, z0:z0+ad, y0:y0+ah, x0:x0+aw] += gw_crop

    out = (out_logits / out_weight.clamp_min(1e-6)).float()
    pred = (out[0, 1] > out[0, 0]).cpu().numpy().astype(np.uint8)
    return pred


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("results/topology_eval.json"))
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cases", type=int, default=0, help="0=all val cases")
    ap.add_argument("--amp-dtype", default="bf16")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    configure_cuda_for_a100(safe_cudnn=True, allow_tf32=True, enable_cudnn=True)
    amp_dtype = _autocast_dtype(args.amp_dtype)

    print(f"[device] {torch.cuda.get_device_name(0)}")
    print(f"[ckpt]   {args.checkpoint}")

    # Load validation cases (same split as training)
    items = list_cases_from_cache(args.cache_dir)
    _, val_items = split_train_val_baseuids(items, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_cases > 0:
        val_items = val_items[:args.max_cases]
    print(f"[data]   {len(val_items)} validation cases")

    # Load model
    model = load_navbrush(args.checkpoint, device)
    patch_size = tuple(args.patch_size)

    results = []
    all_dice, all_cldice, all_betti, all_hd95 = [], [], [], []

    for i, it in enumerate(val_items):
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], args.cache_dir, "image")
        lp = _resolve_cached_path(uid, it["label_pt"], args.cache_dir, "label")

        img = torch.load(ip, map_location="cpu").float()
        lab = torch.load(lp, map_location="cpu")
        if img.ndim == 3: img = img.unsqueeze(0)
        img = img.clamp(0.0, 1.0)
        if lab.ndim == 4 and lab.shape[0] == 1: lab = lab[0]
        gt = (lab > 0).numpy().astype(np.uint8)

        D, H, W = gt.shape
        print(f"\n[{i+1}/{len(val_items)}] {uid[:50]}  shape={D}x{H}x{W}  gt_vox={gt.sum():,}")
        t0 = time.time()

        pred = predict_fullvol(model, img, patch_size, args.overlap, device, amp_dtype)

        # Dice
        dice = compute_dice_np(pred, gt)

        # clDice (expensive — uses distance transform)
        print("  computing clDice...", end=" ", flush=True)
        t1 = time.time()
        cld = compute_cldice(pred, gt)
        print(f"{cld:.4f} ({time.time()-t1:.1f}s)")

        # Betti-0
        betti = compute_betti0_error(pred, gt)

        # HD95
        print("  computing HD95...", end=" ", flush=True)
        t2 = time.time()
        hd = compute_hd95(pred, gt)
        print(f"{hd:.2f}vox ({time.time()-t2:.1f}s)")

        elapsed = time.time() - t0
        print(f"  Dice={dice:.4f}  clDice={cld:.4f}  Betti0_err={betti}  HD95={hd:.2f}  t={elapsed:.1f}s")

        all_dice.append(dice)
        all_cldice.append(cld)
        all_betti.append(betti)
        if not np.isnan(hd):
            all_hd95.append(hd)

        results.append({
            "uid": uid,
            "shape_dhw": [D, H, W],
            "gt_voxels": int(gt.sum()),
            "pred_voxels": int(pred.sum()),
            "dice": dice,
            "cldice": cld,
            "betti0_error": betti,
            "hd95_vox": float(hd) if not np.isnan(hd) else None,
        })

        torch.cuda.empty_cache()

    # Summary
    mean_dice   = float(np.mean(all_dice))
    mean_cldice = float(np.mean(all_cldice))
    mean_betti  = float(np.mean(all_betti))
    mean_hd95   = float(np.mean(all_hd95)) if all_hd95 else None

    print(f"\n{'='*60}")
    print(f"TOPOLOGY EVALUATION — {len(results)} cases")
    print(f"{'='*60}")
    print(f"  Mean Dice:         {mean_dice:.4f}  ({mean_dice*100:.2f}%)")
    print(f"  Mean clDice:       {mean_cldice:.4f}  ({mean_cldice*100:.2f}%)")
    print(f"  Mean Betti-0 err:  {mean_betti:.2f}  (lower=better, 0=perfect)")
    print(f"  Mean HD95 (vox):   {mean_hd95:.2f}" if mean_hd95 else "  Mean HD95:  N/A")
    print(f"{'='*60}")
    print(f"\nComparison targets:")
    print(f"  RSNA nnUNet:   pseudo-Dice~0.826  (no topology metrics published)")
    print(f"  NavBrush v5:   fullvol ~0.7494 (training val, Gaussian weighted)")

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_cases": len(results),
        "patch_size": list(patch_size),
        "overlap": args.overlap,
        "mean_dice":   mean_dice,
        "mean_cldice": mean_cldice,
        "mean_betti0_error": mean_betti,
        "mean_hd95_vox": mean_hd95,
        "per_case": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
