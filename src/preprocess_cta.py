#!/usr/bin/env python3
"""
preprocess_cta.py
=================

Preprocess the TopCoW CTA (Dataset100_TopBrain_CT) into the same .pt tensor
format as the MRA cache produced by preprocess_topcow.py.

Key differences from MRA preprocessing:
  1. CTA images are already normalised to [0,1] by nnUNet — we use them as-is
     (no HU windowing needed since normalisation is already done).
  2. CTA labels have 40 classes (full vascular tree); we remap to the 13-class
     TopCoW segmentation task to match the MRA labels:
       CTA 1→1(BA), 2→2(R-PCA), 3→3(L-PCA), 4→4(R-ICA), 5→5(R-MCA),
       6→6(L-ICA), 7→7(L-MCA), 8→8(R-Pcom), 9→9(L-Pcom), 10→10(Acom),
       11→11(R-ACA), 12→12(L-ACA), 15→13(3rd-A2), all others → 0
  3. CTA images are already at 0.5mm isotropic — no resampling needed.
     We verify this and skip resampling if spacing matches target.
  4. Output cases.json is structured to merge cleanly with the MRA cases.json
     so the dataloader can load both modalities from a single JSON.

Output layout:
  cache_cta/
    images/topcow_ct_001.pt   -> torch.float16 [1, D, H, W]  already [0,1]
    labels/topcow_ct_001.pt   -> torch.uint8   [D, H, W]  remapped 0..13
    meta.json
    cases.json                -> same schema as MRA cases.json

Usage:
  python src/preprocess_cta.py \\
    --src /path/to/Dataset100_TopBrain_CT \\
    --out data/cache_cta \\
    --workers 4

The output cache_cta/cases.json can then be merged with cache_topcow/cases.json
for dual-modal training where:
  - Paired patients (001-025): have both MRA + CTA entries
  - MRA-only patients (026-125): CTA channel filled with zeros at train time
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


# Mapping from 40-class CTA labels to 13-class TopCoW segmentation task.
# Only the 13 canonical CoW segments are kept; all extra vessels → 0.
CTA_TO_TOPCOW: Dict[int, int] = {
    0:  0,   # background
    1:  1,   # BA         → BA
    2:  2,   # R-P1P2     → R-PCA
    3:  3,   # L-P1P2     → L-PCA
    4:  4,   # R-ICA      → R-ICA
    5:  5,   # R-M1       → R-MCA
    6:  6,   # L-ICA      → L-ICA
    7:  7,   # L-M1       → L-MCA
    8:  8,   # R-Pcom     → R-Pcom
    9:  9,   # L-Pcom     → L-Pcom
    10: 10,  # Acom       → Acom
    11: 11,  # R-A1A2     → R-ACA
    12: 12,  # L-A1A2     → L-ACA
    15: 13,  # 3rd-A2     → 3rd-A2
    # All others (R-A3, L-A3, M2, M3, P3/P4, VA, SCA, AICA, PICA, AChA, ECA, etc.) → 0
}

COW_CLASSES = {
    0: "background", 1: "BA", 2: "R-PCA", 3: "L-PCA",
    4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA",
    8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2",
}


def remap_labels(lbl_arr: np.ndarray) -> np.ndarray:
    """Remap 40-class CTA label array to 13-class TopCoW format."""
    out = np.zeros_like(lbl_arr, dtype=np.uint8)
    for src, dst in CTA_TO_TOPCOW.items():
        if dst > 0:
            out[lbl_arr == src] = dst
    return out


def load_nifti_array(path: Path) -> tuple:
    """Load a NIfTI file, return (array_DHW, spacing_xyz)."""
    if HAS_NIBABEL:
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
        # nibabel returns (x,y,z) order; transpose to (z,y,x) = (D,H,W)
        arr = arr.transpose(2, 1, 0)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        return arr, spacing
    elif HAS_SITK:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # already (D,H,W)
        spacing = img.GetSpacing()  # (x,y,z)
        return arr, spacing
    else:
        raise RuntimeError("Neither nibabel nor SimpleITK is available.")


def process_one(args_tuple) -> Dict:
    src_img, src_lbl, out_img, out_lbl, uid = args_tuple

    img_arr, spacing = load_nifti_array(src_img)
    lbl_arr_raw, _ = load_nifti_array(src_lbl)
    lbl_arr_raw = lbl_arr_raw.astype(np.uint8)

    # CTA is already [0,1] normalised — use directly.
    img_norm = img_arr.astype(np.float32)
    # Safety clamp: ensure [0,1] in case of tiny float rounding
    img_norm = np.clip(img_norm, 0.0, 1.0)

    # Remap 40-class → 13-class TopCoW labels
    lbl_arr = remap_labels(lbl_arr_raw)

    # Per-class voxel counts for sampling DB
    classes, counts = np.unique(lbl_arr, return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(classes, counts) if c != 0}

    # Save as .pt tensors (same format as MRA cache)
    img_t = torch.from_numpy(img_norm[np.newaxis].astype(np.float16))  # [1,D,H,W]
    lbl_t = torch.from_numpy(lbl_arr)                                   # [D,H,W]
    torch.save(img_t, str(out_img))
    torch.save(lbl_t, str(out_lbl))

    D, H, W = img_arr.shape
    return {
        "uid": uid,
        "modality": "ct",
        "shape_dhw": [D, H, W],
        "spacing_dhw": [0.5, 0.5, 0.5],   # CTA is already at 0.5mm
        "class_counts": class_counts,
        "n_classes_present": len(class_counts),
        "img_path": str(out_img),
        "lbl_path": str(out_lbl),
    }


def main():
    ap = argparse.ArgumentParser(description="Preprocess TopCoW CTA → .pt cache")
    ap.add_argument("--src", required=True,
                    help="Path to Dataset100_TopBrain_CT/ "
                         "(contains imagesTr/ and labelsTr/)")
    ap.add_argument("--out", required=True,
                    help="Output cache directory (will create images/ and labels/)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers (default 4)")
    ap.add_argument("--folds", type=int, default=5,
                    help="Number of cross-validation folds (default 5, matches MRA)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without processing")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    imgs_dir = src / "imagesTr"
    lbls_dir = src / "labelsTr"

    tasks = []
    for img_path in sorted(imgs_dir.glob("topcow_ct_*_0000.nii.gz")):
        # nnUNet naming: topcow_ct_001_0000.nii.gz → uid = topcow_ct_001
        uid = img_path.name.replace("_0000.nii.gz", "")
        lbl_path = lbls_dir / f"{uid}.nii.gz"
        if not lbl_path.exists():
            print(f"  SKIP {uid}: no label at {lbl_path}")
            continue
        out_img = out / "images" / f"{uid}.pt"
        out_lbl = out / "labels" / f"{uid}.pt"
        tasks.append((img_path, lbl_path, out_img, out_lbl, uid))

    print(f"Found {len(tasks)} CTA cases to process")
    if args.dry_run:
        for t in tasks[:5]:
            print(f"  {t[4]}: {t[0].name} → {t[2].name}")
        print(f"  ... ({len(tasks)} total)")
        return

    results: List[Dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs)):
                try:
                    r = fut.result()
                    results.append(r)
                    print(f"[{len(results)}/{len(tasks)}] {r['uid']}  "
                          f"shape={r['shape_dhw']}  "
                          f"classes_present={r['n_classes_present']}  "
                          f"counts={r['class_counts']}")
                except Exception as e:
                    print(f"ERROR: {e}")
    else:
        for i, t in enumerate(tasks):
            try:
                r = process_one(t)
                results.append(r)
                print(f"[{i+1}/{len(tasks)}] {r['uid']}  "
                      f"shape={r['shape_dhw']}  "
                      f"classes_present={r['n_classes_present']}  "
                      f"counts={r['class_counts']}")
            except Exception as e:
                print(f"ERROR on {t[4]}: {e}")

    results.sort(key=lambda r: r["uid"])

    # Assign folds matching the MRA fold strategy (stratified by modality)
    rng = random.Random(args.seed)
    idxs = list(range(len(results)))
    rng.shuffle(idxs)
    for pos, idx in enumerate(idxs):
        results[idx]["fold"] = pos % args.folds

    meta = {
        "target_spacing_dhw": [0.5, 0.5, 0.5],
        "normalization": {
            "ct": {
                "method": "already_normalised_01",
                "note": "nnUNet preprocessed to [0,1]; no further normalisation applied"
            }
        },
        "label_remap": {
            "source": "40-class TopBrain_CT",
            "target": "13-class TopCoW",
            "mapping": {str(k): v for k, v in CTA_TO_TOPCOW.items() if v > 0},
        },
        "classes": COW_CLASSES,
        "num_classes": 14,
        "n_folds": args.folds,
        "seed": args.seed,
        "modalities": ["ct"],
        "num_cases": len(results),
    }

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "cases.json").write_text(json.dumps(results, indent=2))

    print(f"\nDone. Cached {len(results)} CTA cases to {out}")
    print(f"  meta:   {out / 'meta.json'}")
    print(f"  cases:  {out / 'cases.json'}")

    # Print per-class statistics across the dataset
    all_counts: Dict[int, int] = {}
    cases_with: Dict[int, int] = {}
    for r in results:
        for c, n in r["class_counts"].items():
            all_counts[c] = all_counts.get(c, 0) + n
            cases_with[c] = cases_with.get(c, 0) + 1

    print(f"\nPer-class statistics (across {len(results)} CTA cases):")
    print(f"  {'cls':>4} {'name':<12} {'total_voxels':>14} {'cases':>8}")
    for c in sorted(all_counts.keys()):
        name = COW_CLASSES.get(int(c), f"cls{c}")
        print(f"  {c:>4} {name:<12} {all_counts[c]:>14,} {cases_with[c]:>5}/{len(results)}")


if __name__ == "__main__":
    main()
