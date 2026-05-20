#!/usr/bin/env python3
"""
preprocess_topcow.py
====================

Cache the TopCoW 2024 dataset (13-class Circle-of-Willis labels) at a common
0.5 mm^3 isotropic spacing as .pt tensors, ready for GraphCoW-Net training.

Why this is a separate step from the training loader:
  * Resampling 200+ volumes per epoch is wasted CPU; do it once.
  * The encoder ASSUMES isotropic voxels (tri-planar views must share
    physical vessel thickness across axes).
  * Intensity normalization differs per modality (CT uses HU window + scale;
    MR uses robust-percentile clip). Fixed at cache time so the loader is
    modality-agnostic.

Output layout (at --out):
  cache_topcow/
    images/<uid>.pt          -> torch.float16 [1, D, H, W]
    labels/<uid>.pt          -> torch.uint8   [D, H, W]  values 0..13
    meta.json                -> target spacing, norm params, class names
    cases.json               -> per-case metadata + 5-fold split indices

Run (serial, ~3-5 min per case):
  /scratch/.../aneur/bin/python mamba_snake/src/preprocess_topcow.py \\
      --src /scratch/.../TopCoW2024_Data_Release \\
      --out mamba_snake/data/cache_topcow \\
      --target-spacing 0.5 --modality mr

Parallel:
  --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch


COW_CLASSES = {
    0: "background",
    1: "BA",
    2: "R-PCA",
    3: "L-PCA",
    4: "R-ICA",
    5: "R-MCA",
    6: "L-ICA",
    7: "L-MCA",
    8: "R-Pcom",
    9: "L-Pcom",
    10: "Acom",
    11: "R-ACA",
    12: "L-ACA",
    13: "3rd-A2",
}


def resample_to_spacing(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float],
    is_label: bool,
) -> sitk.Image:
    """Resample a SimpleITK image to the target physical spacing.

    target_spacing is in (x, y, z) order (SimpleITK convention).
    Labels use nearest-neighbor; images use trilinear.
    """
    src_spacing = img.GetSpacing()
    src_size = img.GetSize()
    new_size = [
        int(round(src_size[i] * src_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)


def normalize_mr(arr: np.ndarray, clip_pcts=(0.5, 99.5)) -> np.ndarray:
    """Robust percentile clip to [0, 1] for MR. No negative values expected."""
    lo, hi = np.percentile(arr, clip_pcts)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return arr.astype(np.float32)


def normalize_ct(arr: np.ndarray, window=(-100.0, 700.0)) -> np.ndarray:
    """HU window to [0, 1] for CT. Standard head/neck CTA window."""
    lo, hi = window
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def process_one(args_tuple) -> Dict:
    (src_img, src_lbl, out_img, out_lbl, target_spacing, modality, uid) = args_tuple
    img = sitk.ReadImage(str(src_img))
    lbl = sitk.ReadImage(str(src_lbl))

    img_r = resample_to_spacing(img, target_spacing, is_label=False)
    lbl_r = resample_to_spacing(lbl, target_spacing, is_label=True)

    # Arrays in SITK are (z, y, x); we want (D=z, H=y, W=x) for torch.
    img_arr = sitk.GetArrayFromImage(img_r)  # (D, H, W) float32
    lbl_arr = sitk.GetArrayFromImage(lbl_r).astype(np.uint8)  # (D, H, W) uint8

    if modality == "mr":
        img_norm = normalize_mr(img_arr)
    else:
        img_norm = normalize_ct(img_arr)

    # Per-class foreground voxel count for sampling DB.
    classes, counts = np.unique(lbl_arr, return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(classes, counts) if c != 0}

    # Save. Image as fp16 to halve disk; label as uint8.
    img_t = torch.from_numpy(img_norm[None].astype(np.float16))   # [1, D, H, W]
    lbl_t = torch.from_numpy(lbl_arr)                             # [D, H, W]
    torch.save(img_t, out_img)
    torch.save(lbl_t, out_lbl)

    return {
        "uid": uid,
        "modality": modality,
        "shape_dhw": list(img_arr.shape),
        "spacing_dhw": [float(target_spacing[2]),
                        float(target_spacing[1]),
                        float(target_spacing[0])],
        "class_counts": class_counts,
        "n_classes_present": len(class_counts),
        "img_path": str(out_img),
        "lbl_path": str(out_lbl),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Path to TopCoW2024_Data_Release/ (must contain "
                         "imagesTr/ and cow_seg_labelsTr/)")
    ap.add_argument("--out", required=True,
                    help="Output cache dir (will create images/ and labels/)")
    ap.add_argument("--target-spacing", type=float, default=0.5,
                    help="Isotropic target spacing in mm")
    ap.add_argument("--modality", choices=("mr", "ct", "both"),
                    default="mr")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip actual processing; just print the plan")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    imgs_dir = src / "imagesTr"
    lbls_dir = src / "cow_seg_labelsTr"

    tasks = []
    modalities = ["mr", "ct"] if args.modality == "both" else [args.modality]
    for mod in modalities:
        for img_path in sorted(imgs_dir.glob(f"topcow_{mod}_*_0000.nii.gz")):
            stem = img_path.name.replace("_0000.nii.gz", "")  # topcow_mr_001
            lbl_path = lbls_dir / f"{stem}.nii.gz"
            if not lbl_path.exists():
                print(f"  SKIP {stem}: no label at {lbl_path}")
                continue
            out_img = out / "images" / f"{stem}.pt"
            out_lbl = out / "labels" / f"{stem}.pt"
            target = (args.target_spacing,) * 3
            tasks.append((img_path, lbl_path, out_img, out_lbl, target, mod, stem))

    print(f"Planned tasks: {len(tasks)}  modalities={modalities}  "
          f"target_spacing={args.target_spacing}mm")

    if args.dry_run:
        for t in tasks[:5]:
            print(" ", t[6], t[0].name, "->", t[2].name)
        print(f"  ... ({len(tasks)} total)")
        return

    results: List[Dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                results.append(r)
                print(f"[{i + 1}/{len(tasks)}] {r['uid']}  "
                      f"shape={r['shape_dhw']}  classes={r['n_classes_present']}")
    else:
        for i, t in enumerate(tasks):
            r = process_one(t)
            results.append(r)
            print(f"[{i + 1}/{len(tasks)}] {r['uid']}  "
                  f"shape={r['shape_dhw']}  classes={r['n_classes_present']}")

    results.sort(key=lambda r: r["uid"])

    # Stratified 5-fold split by modality.
    rng = random.Random(args.seed)
    for r in results:
        r["_stratum"] = r["modality"]

    strata: Dict[str, List[int]] = {}
    for i, r in enumerate(results):
        strata.setdefault(r["_stratum"], []).append(i)

    fold_assign = [0] * len(results)
    for stratum, idxs in strata.items():
        rng.shuffle(idxs)
        for pos, idx in enumerate(idxs):
            fold_assign[idx] = pos % args.folds

    for i, r in enumerate(results):
        r["fold"] = fold_assign[i]
        del r["_stratum"]

    meta = {
        "target_spacing_dhw": [args.target_spacing] * 3,
        "normalization": {
            "mr": {"method": "robust01", "clip_pcts": [0.5, 99.5]},
            "ct": {"method": "hu_window", "window": [-100.0, 700.0]},
        },
        "classes": COW_CLASSES,
        "num_classes": 14,
        "n_folds": args.folds,
        "seed": args.seed,
        "modalities": modalities,
        "num_cases": len(results),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "cases.json").write_text(json.dumps(results, indent=2))

    print(f"\nDone. Cached {len(results)} cases to {out}")
    print(f"  meta: {out / 'meta.json'}")
    print(f"  cases: {out / 'cases.json'}")


if __name__ == "__main__":
    main()
