#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sampling_db.py

Builds fast sampling indices per case:
  - fg_coords: all vessel voxels (label>0)
  - bg_coords: random background voxels (label==0) (optional)
  - hardneg_coords: "anatomical hard negatives" = label==0 AND intensity above a quantile threshold

This is a ONE-TIME preprocessing step that removes heavy per-iteration CPU work and
enables active background sampling.

Expected cache layout:
  cache_dir/cases.json with:
    {"uids":[...], "images":{uid:path}, "labels":{uid:path}}
  and/or fallback:
    cache_dir/images/{uid}.pt, cache_dir/labels/{uid}.pt

Each saved file:
  out_dir/{uid}.npz with arrays: fg, hardneg, shape
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def _resolve_cached_path(uid: str, p: str, cache_dir: Path, kind: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)
    if kind == "image":
        alt = cache_dir / "images" / f"{uid}.pt"
    else:
        alt = cache_dir / "labels" / f"{uid}.pt"
    return str(alt)


def list_cases_from_cache(cache_dir: Path):
    cases_json = cache_dir / "cases.json"
    if not cases_json.exists():
        raise RuntimeError(f"cases.json not found in cache-dir: {cases_json}")
    js = json.loads(cases_json.read_text())
    uids = js["uids"]
    images = js["images"]
    labels = js["labels"]
    items = []
    for uid in uids:
        items.append({"uid": uid, "image_pt": images[uid], "label_pt": labels[uid]})
    if not items:
        raise RuntimeError("No items in cases.json")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--intensity-q", type=float, default=0.90,
                    help="Quantile for hard negative sampling from intensity (label==0)")
    ap.add_argument("--max-hardneg", type=int, default=400000,
                    help="Cap hardneg coords per case to limit disk size")
    ap.add_argument("--max-fg", type=int, default=400000,
                    help="Cap fg coords per case to limit disk size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = list_cases_from_cache(args.cache_dir)
    summary = {"cache_dir": str(args.cache_dir), "out_dir": str(args.out_dir),
               "intensity_q": float(args.intensity_q), "cases": []}

    for it in items:
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], args.cache_dir, "image")
        lp = _resolve_cached_path(uid, it["label_pt"], args.cache_dir, "label")

        img = torch.load(ip, map_location="cpu")  # [1,D,H,W], float
        lab = torch.load(lp, map_location="cpu")  # [D,H,W], uint8/int
        if img.ndim != 4 or img.shape[0] != 1:
            raise RuntimeError(f"{uid}: image must be [1,D,H,W], got {tuple(img.shape)}")
        if lab.ndim != 3:
            raise RuntimeError(f"{uid}: label must be [D,H,W], got {tuple(lab.shape)}")

        img_np = img[0].float().numpy()
        lab_np = (lab > 0).numpy().astype(np.uint8)

        fg = np.argwhere(lab_np > 0).astype(np.int32)  # [N,3] z,y,x
        if fg.shape[0] > args.max_fg:
            sel = rng.choice(fg.shape[0], size=args.max_fg, replace=False)
            fg = fg[sel]

        # hard negatives: label==0 and intensity > quantile
        bg_mask = (lab_np == 0)
        if bg_mask.any():
            thr = float(np.quantile(img_np[bg_mask], args.intensity_q))
            hard = np.argwhere((bg_mask) & (img_np >= thr)).astype(np.int32)
        else:
            thr = 1.0
            hard = np.zeros((0, 3), dtype=np.int32)

        if hard.shape[0] > args.max_hardneg:
            sel = rng.choice(hard.shape[0], size=args.max_hardneg, replace=False)
            hard = hard[sel]

        out = args.out_dir / f"{uid}.npz"
        np.savez_compressed(out, fg=fg, hardneg=hard, shape=np.array(lab_np.shape, dtype=np.int32))

        summary["cases"].append({
            "uid": uid,
            "shape": list(lab_np.shape),
            "fg_n": int(fg.shape[0]),
            "hardneg_n": int(hard.shape[0]),
            "hardneg_thr_q": float(args.intensity_q),
            "hardneg_thr_val": float(thr),
        })

        print(json.dumps({"uid": uid, "shape": list(lab_np.shape),
                          "fg_n": int(fg.shape[0]), "hardneg_n": int(hard.shape[0])}), flush=True)

    (args.out_dir / "sampling_db_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Wrote {len(summary['cases'])} npz files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
