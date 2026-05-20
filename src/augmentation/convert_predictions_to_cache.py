"""
Convert nnU-Net predictions (seg.npz + roi_data.npz) to mamba_snake cache format.

Reads each prediction subfolder:
  - roi_data.npz  -> images/{uid}.pt  (float32, [1,D,H,W], normalized 0-1)
  - seg.npz       -> labels/{uid}.pt  (uint8, [D,H,W], binary 0/1)

Outputs a new cache dir with cases.json, images/, labels/
ready to be passed to augment_cache_vessels.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


def convert(pred_dir: Path, out_dir: Path, max_cases: int = -1) -> None:
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    cases = sorted([p for p in pred_dir.iterdir() if p.is_dir()])
    if max_cases > 0:
        cases = cases[:max_cases]

    uids: List[str] = []
    images: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    skipped = 0

    for case_dir in tqdm(cases, desc="converting"):
        uid = case_dir.name
        seg_path = case_dir / "seg.npz"
        roi_path = case_dir / "roi_data.npz"

        if not seg_path.exists() or not roi_path.exists():
            skipped += 1
            continue

        # --- image: roi_data.npz -> float32 [1,D,H,W] normalized to [0,1] ---
        roi = np.load(roi_path)["roi"]          # [1,D,H,W] float16
        img = roi.astype(np.float32)
        # normalize to [0,1] robustly (1st-99th percentile)
        lo = float(np.percentile(img, 1.0))
        hi = float(np.percentile(img, 99.0))
        if hi > lo + 1e-6:
            img = np.clip(img, lo, hi)
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)
        img_t = torch.from_numpy(img)           # [1,D,H,W] float32

        # --- label: seg.npz -> uint8 [D,H,W] binary ---
        seg = np.load(seg_path)["segmentation"] # [D,H,W] uint8, multi-class
        lbl = (seg > 0).astype(np.uint8)        # binarize: any vessel = 1
        lbl_t = torch.from_numpy(lbl)           # [D,H,W] uint8

        img_out = out_img / f"{uid}.pt"
        lbl_out = out_lbl / f"{uid}.pt"
        torch.save(img_t, img_out)
        torch.save(lbl_t, lbl_out)

        uids.append(uid)
        images[uid] = str(img_out)
        labels[uid] = str(lbl_out)

    # write cases.json
    (out_dir / "cases.json").write_text(json.dumps(
        {"uids": uids, "images": images, "labels": labels}, indent=2
    ))

    print(f"\n✅ Converted {len(uids)} cases, skipped {skipped}")
    print(f"✅ Output: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="predictions_v4_margin15_30/ directory")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="output cache dir (images/ labels/ cases.json)")
    ap.add_argument("--max-cases", type=int, default=-1,
                    help="limit for quick test")
    args = ap.parse_args()
    convert(args.pred_dir, args.out_dir, args.max_cases)


if __name__ == "__main__":
    main()
