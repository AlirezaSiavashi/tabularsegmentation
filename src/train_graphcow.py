#!/usr/bin/env python3
"""
train_graphcow.py
=================

Phase 1 training (voxel_only) for GraphCoW-Net on TopCoW 2024 13-class.

This script only runs Phase 1. Later phases share the same model but load
this phase's best checkpoint and change:
  * which loss terms are active (via GraphCoWLoss.phase = ...)
  * which parts of the model are trained (frozen/unfrozen).
A separate script per phase keeps the LR schedule / freeze logic readable.

Run:
  /scratch/.../aneur/bin/python mamba_snake/src/train_graphcow.py \\
      --cases mamba_snake/data/cache_topcow/cases.json \\
      --variant vits16 --unfreeze-top-k 2 --batch-size 2 --patch 192 192 192 \\
      --epochs 120 --fold 0 --out logs/graphcow_phase1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import TopCoWDataset, collate_patches  # noqa: E402
from models.graphcow_net import GraphCoWNet  # noqa: E402
from models.losses.graphcow_losses import (  # noqa: E402
    GraphCoWLoss, GraphCoWLossWeights,
)


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def _gaussian_weight(patch: Tuple[int, int, int], sigma_scale: float = 0.125,
                    device: str = "cuda") -> torch.Tensor:
    """Standard nnU-Net Gaussian patch weight. Centers get weight 1; edges
    fade so overlapping tiles blend smoothly."""
    d, h, w = patch
    sigma = [s * sigma_scale for s in patch]
    g = torch.zeros((d, h, w), device=device)
    coords = [torch.arange(n, device=device) - (n - 1) / 2 for n in patch]
    zz, yy, xx = torch.meshgrid(*coords, indexing="ij")
    g = torch.exp(-(zz * zz / (2 * sigma[0] ** 2) +
                    yy * yy / (2 * sigma[1] ** 2) +
                    xx * xx / (2 * sigma[2] ** 2)))
    return (g / g.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding_window_infer(
    model: torch.nn.Module,
    image: torch.Tensor,                 # [1, 1, D, H, W]
    num_classes: int,
    patch: Tuple[int, int, int] = (192, 192, 192),
    overlap: float = 0.5,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Full-volume sliding-window inference. Returns softmax probs
    [1, num_classes, D, H, W] on CPU so large volumes don't OOM."""
    model.eval()
    D, H, W = image.shape[-3:]
    pD, pH, pW = patch

    # Pad so each dim is >= patch_size AND multiple of 32 (encoder requirement).
    def _pad(n, p):
        n2 = max(n, p)
        return ((n2 + 31) // 32) * 32
    tgt = (_pad(D, pD), _pad(H, pH), _pad(W, pW))
    pad = (0, tgt[2] - W, 0, tgt[1] - H, 0, tgt[0] - D)
    img = F.pad(image, pad, mode="constant", value=0).to(device)

    stride = tuple(int(round(p * (1 - overlap))) for p in patch)

    starts = [list(range(0, max(1, tgt[i] - patch[i] + 1), stride[i]))
              for i in range(3)]
    for i in range(3):
        if starts[i][-1] + patch[i] < tgt[i]:
            starts[i].append(tgt[i] - patch[i])

    probs_acc = torch.zeros((1, num_classes) + tgt, dtype=torch.float32, device=device)
    weight_acc = torch.zeros((1, 1) + tgt, dtype=torch.float32, device=device)
    gw = _gaussian_weight(patch, device=device).unsqueeze(0).unsqueeze(0)

    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                tile = img[:, :, z:z + pD, y:y + pH, x:x + pW]
                with torch.amp.autocast("cuda", enabled=amp_dtype == torch.bfloat16,
                                        dtype=torch.bfloat16):
                    out = model(tile, mode="voxel_only", render_tube=False)
                p = F.softmax(out["logits"].float(), dim=1)
                probs_acc[:, :, z:z + pD, y:y + pH, x:x + pW] += p * gw
                weight_acc[:, :, z:z + pD, y:y + pH, x:x + pW] += gw

    probs_acc = probs_acc / weight_acc.clamp_min(1e-6)
    # Crop back to original size.
    probs_acc = probs_acc[:, :, :D, :H, :W]
    return probs_acc.cpu()


def per_class_dice(pred: torch.Tensor, target: torch.Tensor,
                   num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """pred: [D,H,W] long, target: [D,H,W] long."""
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        if denom == 0:
            dices.append(float("nan"))   # class not present
        else:
            dices.append(2 * inter / (denom + eps))
    return torch.tensor(dices, dtype=torch.float32)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    patch: Tuple[int, int, int],
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    dice_acc: List[torch.Tensor] = []
    t0 = time.time()
    for i, batch in enumerate(val_loader):
        if max_cases is not None and i >= max_cases:
            break
        img = batch["image"].to(torch.float32)  # [1, 1, D, H, W]
        if img.dim() == 4:
            img = img.unsqueeze(0)
        lbl = batch["label"]                    # [1, D, H, W]  OR  [D, H, W]
        if lbl.dim() == 4:
            lbl = lbl.squeeze(0)
        probs = sliding_window_infer(model, img, num_classes=num_classes,
                                     patch=patch, device=device, amp_dtype=amp_dtype)
        pred = probs.argmax(dim=1).squeeze(0).long()   # [D, H, W]
        d = per_class_dice(pred, lbl.long(), num_classes)
        dice_acc.append(d)
        uid = batch["uid"][0] if isinstance(batch["uid"], list) else batch["uid"]
        present_mean = float(d[1:][~torch.isnan(d[1:])].mean().item())
        print(f"    [val {i + 1}] {uid}: mean_fg_dice={present_mean:.4f}")

    D = torch.stack(dice_acc, dim=0)  # [N, C]
    mean_per_class = torch.stack([
        D[:, c][~torch.isnan(D[:, c])].mean() if (~torch.isnan(D[:, c])).any()
        else torch.tensor(float("nan"))
        for c in range(num_classes)
    ])
    fg = mean_per_class[1:]
    mean_fg = float(fg[~torch.isnan(fg)].mean().item())
    out = {"val_mean_fg_dice": mean_fg, "val_time_s": time.time() - t0}
    for c in range(num_classes):
        v = mean_per_class[c].item()
        out[f"val_dice/class{c:02d}"] = v
    return out


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float,
                      min_lr_frac: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_lr_frac + 0.5 * (1 - min_lr_frac) * (1 + math.cos(math.pi * t)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--variant", default="vits16")
    ap.add_argument("--unfreeze-top-k", type=int, default=2)
    ap.add_argument("--modalities", nargs="+", default=["mr"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patch", type=int, nargs=3, default=[192, 192, 192])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--samples-per-epoch", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=14)
    ap.add_argument("--lr-vit", type=float, default=1e-5)
    ap.add_argument("--lr-encoder-adapter", type=float, default=5e-5)
    ap.add_argument("--lr-decoder", type=float, default=1e-4)
    ap.add_argument("--lr-graph", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-max-cases", type=int, default=None)
    ap.add_argument("--cld-target-classes", type=str, default="",
                    help="Comma-separated classes to apply clDice to. "
                         "Empty = all foreground. e.g. '8,9,10,13'.")
    ap.add_argument("--cld-iters", type=int, default=5,
                    help="clDice skeletonization iterations. 10 is classic; "
                         "5 is sufficient at 0.5mm iso and halves activation memory.")
    ap.add_argument("--cb-iters", type=int, default=4,
                    help="cbDice skeletonization iterations.")
    ap.add_argument("--conn-weight", type=float, default=0.0,
                    help="Per-class connectivity loss weight (Phase 1b).")
    ap.add_argument("--conn-target-classes", type=str, default="",
                    help="Comma-separated class indices the connectivity "
                         "loss applies to. Empty = all foreground classes.")
    ap.add_argument("--conn-dilate-kernel", type=int, default=5,
                    help="Tube dilation kernel (voxels) for connectivity.")
    ap.add_argument("--conn-skel-iters", type=int, default=5)
    ap.add_argument("--class-quota", type=str, default="",
                    help="Comma-separated class indices that must be "
                         "guaranteed-sampled. e.g. '8,9,10,13'.")
    ap.add_argument("--class-quota-prob", type=float, default=0.35)
    ap.add_argument("--elastic-prob", type=float, default=0.0)
    ap.add_argument("--class-weight-boost", type=str, default="",
                    help="Comma-separated 'cls:val' pairs to set CE class "
                         "weights, e.g. '8:2.0,9:2.0,10:2.0,13:3.0'.")
    ap.add_argument("--resume-weights-only", action="store_true",
                    help="Load model weights but restart optimiser, LR "
                         "schedule, and epoch counter from zero.")
    ap.add_argument("--full-vol-encoder", action="store_true",
                    help="Feed full brain volume to DINOv3 tri-planar (global "
                         "topology context) while keeping patch-based stem+decoder. "
                         "Key architectural improvement for Pcom/Acom/3rd-A2.")
    ap.add_argument("--attn-weight", type=float, default=0.0,
                    help="ViT attention supervision loss weight. Supervises the "
                         "CLS attention of the last 2 unfrozen ViT blocks to be "
                         "high on thin-vessel patches (classes 8,9,10,13). "
                         "Recommended: 0.3. Only active when > 0.")
    ap.add_argument("--attn-target-classes", type=str, default="8,9,10,13",
                    help="Comma-separated classes for attention supervision.")
    ap.add_argument("--cld-weight", type=float, default=0.5,
                    help="clDice loss weight. Set to 0 to disable (saves ~10 GB "
                         "at 192^3 bs=1 because S(p) keeps a long backward graph).")
    ap.add_argument("--cb-weight", type=float, default=0.5,
                    help="cbDice loss weight. Set to 0 to disable.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="HuggingFace cache dir for DINOv3 weights")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Parse class-quota + connectivity target-classes.
    def _parse_int_csv(s: str) -> Tuple[int, ...]:
        s = (s or "").strip()
        if not s:
            return ()
        return tuple(int(x) for x in s.split(",") if x.strip())

    class_quota = _parse_int_csv(args.class_quota)
    conn_target_classes = _parse_int_csv(args.conn_target_classes) or None
    cld_target_classes = _parse_int_csv(args.cld_target_classes) or None

    # Dataloaders.
    train_ds = TopCoWDataset(
        cases_json=args.cases, fold=args.fold, split="train",
        modalities=tuple(args.modalities), patch_size=tuple(args.patch),
        samples_per_epoch=args.samples_per_epoch,
        class_quota=class_quota,
        class_quota_prob=args.class_quota_prob,
        elastic_prob=args.elastic_prob,
        return_full_vol=args.full_vol_encoder,
    )
    val_ds = TopCoWDataset(
        cases_json=args.cases, fold=args.fold, split="val",
        modalities=tuple(args.modalities), patch_size=tuple(args.patch),
        samples_per_epoch=0,  # unused for val
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_patches,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=None,
        pin_memory=False,
    )
    print(f"[data] train_cases={len(train_ds.cases)} val_cases={len(val_ds.cases)} "
          f"samples_per_epoch={args.samples_per_epoch} batch={args.batch_size} "
          f"x accum={args.grad_accum}")

    # Model.
    device = "cuda"
    net = GraphCoWNet(
        num_classes=args.num_classes,
        variant=args.variant,
        unfreeze_top_k=args.unfreeze_top_k,
        cache_dir=args.cache_dir,
    ).to(device)
    net.train()

    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[model] params total={n_total / 1e6:.2f}M trainable={n_train / 1e6:.2f}M "
          f"ViT unfrozen blocks={getattr(net.encoder.triplanar.vit, 'num_unfrozen_blocks', 0)}")

    # Optimizer.
    groups = net.trainable_parameter_groups(
        lr_encoder_adapter=args.lr_encoder_adapter,
        lr_decoder=args.lr_decoder,
        lr_graph=args.lr_graph,
        lr_vit=args.lr_vit,
        weight_decay=args.weight_decay,
    )
    # Remember each group's base LR for the cosine schedule.
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups)

    # Loss.
    class_weight = torch.ones(args.num_classes)
    # Default: slight boost on small classes (Acom/Pcom are tiny).
    for c in (8, 9, 10):
        class_weight[c] = 1.5
    # CLI override: --class-weight-boost "8:2.0,9:2.0,10:2.0,13:3.0"
    if args.class_weight_boost.strip():
        for pair in args.class_weight_boost.split(","):
            k, v = pair.split(":")
            class_weight[int(k)] = float(v)
    print(f"[loss] class_weight={class_weight.tolist()}")

    loss_weights = GraphCoWLossWeights()
    loss_weights.cld = args.cld_weight
    loss_weights.cb = args.cb_weight
    loss_weights.conn = args.conn_weight
    criterion = GraphCoWLoss(
        num_classes=args.num_classes,
        class_weight=class_weight.to(device),
        weights=loss_weights,
        cld_iters=args.cld_iters,
        cb_iters=args.cb_iters,
        conn_target_classes=conn_target_classes,
        conn_dilate_kernel=args.conn_dilate_kernel,
        conn_skel_iters=args.conn_skel_iters,
        cld_target_classes=cld_target_classes,
    ).to(device)

    # Schedule in units of optimizer steps, not mini-batches.
    steps_per_epoch = max(1, args.samples_per_epoch //
                          (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"[sched] steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"warmup_steps={warmup_steps}")

    start_epoch = 0
    best_dice = -1.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(ckpt["model"], strict=False)
        print(f"[resume] loaded {args.resume}; missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        if args.resume_weights_only:
            # Warm-restart: keep weights, restart optimizer + schedule + best.
            start_epoch = 0
            best_dice = -1.0
            print("[resume] weights-only: optimiser & schedule reset, "
                  "best_dice reset.")
        else:
            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_dice = float(ckpt.get("best_dice", -1.0))
            print(f"[resume] continuing at epoch {start_epoch} best={best_dice:.4f}")

    # Training loop.
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        net.train()
        t_ep = time.time()
        running = {"loss": 0.0, "dice_ce": 0.0, "cb": 0.0, "cld": 0.0,
                   "conn": 0.0, "attn": 0.0, "aux_voxel": 0.0}
        count = 0
        opt.zero_grad(set_to_none=True)

        for step_in_epoch in range(steps_per_epoch):
            # Pull grad_accum mini-batches per optimizer step.
            total_loss_accum = 0.0
            logs_accum: Dict[str, float] = {}
            for micro in range(args.grad_accum):
                try:
                    batch = next(train_iter)
                except (StopIteration, NameError):
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                img = batch["image"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)

                # Full volume for global-context encoder (batch size=1 so
                # index 0; variable spatial size, kept on CPU until needed).
                full_vol = None
                if args.full_vol_encoder and "full_vol" in batch:
                    # batch["full_vol"] is a list of [1,D,H,W] tensors.
                    # We process one at a time (batch=1 in training).
                    full_vol = batch["full_vol"][0].unsqueeze(0).to(device, non_blocking=True)

                attn_mask = lbl if args.attn_weight > 0 else None
                attn_classes = tuple(_parse_int_csv(args.attn_target_classes)) or (8, 9, 10, 13)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = net(img, mode="voxel_only", render_tube=False,
                              attn_supervision_mask=attn_mask,
                              attn_target_classes=attn_classes,
                              full_vol=full_vol)
                    loss, logs = criterion(
                        out, targets={"mask": lbl}, phase="voxel_warmup",
                    )
                    if args.attn_weight > 0 and "attn_loss" in out:
                        attn_l = out["attn_loss"]
                        loss = loss + args.attn_weight * attn_l
                        logs["attn"] = attn_l.detach()
                    loss = loss / args.grad_accum
                loss.backward()
                total_loss_accum += loss.item()
                for k, v in logs.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        logs_accum[k] = logs_accum.get(k, 0.0) + float(v.item())

            # LR step.
            lr_scales = {}
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = cosine_with_warmup(
                    global_step, total_steps, warmup_steps, base_lrs[gi],
                )
                lr_scales[f"lr/g{gi}"] = g["lr"]

            # Grad clip at 1.0, standard for transformers.
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], max_norm=1.0,
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            # Running log.
            running["loss"] += total_loss_accum
            for k in ("dice_ce/dice", "dice_ce/ce", "cb", "cldice", "conn",
                      "attn", "aux_voxel"):
                v = logs_accum.get(k)
                if v is not None:
                    key = ("dice_ce" if k.startswith("dice_ce") else
                           "cld" if k == "cldice" else k.replace("/", "_"))
                    running[key] = running.get(key, 0.0) + v / args.grad_accum
            count += 1

            if (step_in_epoch + 1) % 10 == 0:
                elapsed = time.time() - t_ep
                avg = {k: v / count for k, v in running.items()}
                print(
                    f"  ep{epoch:03d} step{step_in_epoch + 1:03d}/{steps_per_epoch}  "
                    f"loss={avg['loss']:.4f}  dice_ce={avg.get('dice_ce', 0):.3f}  "
                    f"cb={avg.get('cb', 0):.3f}  cld={avg.get('cld', 0):.3f}  "
                    f"conn={avg.get('conn', 0):.3f}  "
                    f"attn={avg.get('attn', 0):.3f}  "
                    f"aux={avg.get('aux_voxel', 0):.3f}  "
                    f"lr_vit={opt.param_groups[2]['lr']:.2e}  "
                    f"lr_dec={opt.param_groups[1]['lr']:.2e}  "
                    f"({elapsed:.0f}s)"
                )

        epoch_time = time.time() - t_ep
        avg = {k: v / max(count, 1) for k, v in running.items()}
        print(f"[ep {epoch:03d}] train_loss={avg['loss']:.4f}  "
              f"dice_ce={avg.get('dice_ce', 0):.4f}  time={epoch_time:.0f}s")

        # --- validate ---
        do_val = ((epoch + 1) % args.val_every == 0) or (epoch == args.epochs - 1)
        val_stats = {}
        if do_val:
            val_stats = validate(
                net, val_loader, num_classes=args.num_classes,
                patch=tuple(args.patch), device=device,
                max_cases=args.val_max_cases,
            )
            print(f"[ep {epoch:03d}] VAL mean_fg_dice={val_stats['val_mean_fg_dice']:.4f}  "
                  f"time={val_stats['val_time_s']:.0f}s")

        # --- checkpoint ---
        ckpt = {
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": vars(args),
            "train_avg": avg,
            "val": val_stats,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if do_val and val_stats.get("val_mean_fg_dice", -1.0) > best_dice:
            best_dice = val_stats["val_mean_fg_dice"]
            ckpt["best_dice"] = best_dice
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[ep {epoch:03d}] new best mean_fg_dice={best_dice:.4f} -> best.pt")

        # --- append training log ---
        log_row = {"epoch": epoch, "train": avg, "val": val_stats,
                   "time_s": epoch_time}
        with (out_dir / "train_log.jsonl").open("a") as f:
            f.write(json.dumps(log_row) + "\n")

    print(f"\nDone. best_val_mean_fg_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
