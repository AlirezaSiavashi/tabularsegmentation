
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_cached_path(uid: str, p: str, cache_dir: Path, kind: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)
    if kind == "image":
        alt = cache_dir / "images" / f"{uid}.pt"
    else:
        alt = cache_dir / "labels" / f"{uid}.pt"
    return str(alt)


def _load_cases(cache_dir: Path) -> List[Dict[str, str]]:
    cases_json = cache_dir / "cases.json"
    if not cases_json.exists():
        raise RuntimeError(f"cases.json not found: {cases_json}")
    js = json.loads(cases_json.read_text())
    uids = js.get("uids", [])
    images = js.get("images", {})
    labels = js.get("labels", {})
    items = []
    for uid in uids:
        if uid not in images or uid not in labels:
            continue
        items.append({"uid": uid, "image_pt": images[uid], "label_pt": labels[uid]})
    if not items:
        raise RuntimeError("No valid items in cases.json")
    return items


def _cap_coords(coords: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0:
        return coords[:0]
    if coords.shape[0] > max_n:
        sel = rng.choice(coords.shape[0], size=max_n, replace=False)
        return coords[sel]
    return coords


def _ring_around_fg(fg_mask: torch.Tensor, ring_width: int) -> torch.Tensor:
    """
    fg_mask: [1,1,D,H,W] bool
    returns: ring mask [D,H,W] bool where ring = dilate(fg, ring_width) & ~fg
    """
    if ring_width <= 0:
        return torch.zeros_like(fg_mask[0, 0], dtype=torch.bool)

    x = fg_mask.float()
    # iterative dilation with 3x3x3 max-pool
    for _ in range(ring_width):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    dil = x > 0.5
    ring = dil & (~fg_mask)
    return ring[0, 0]


def _sample_from_mask(mask: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords.astype(np.int32, copy=False)
    return _cap_coords(coords, max_n, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--intensity-q", type=float, default=0.90,
                    help="Quantile on background intensity for hard negatives")
    ap.add_argument("--ring-width", type=int, default=3,
                    help="Boundary ring width in voxels around vessel mask")

    ap.add_argument("--max-fg", type=int, default=400000)
    ap.add_argument("--max-hardneg", type=int, default=400000)
    ap.add_argument("--max-bg-easy", type=int, default=400000)
    ap.add_argument("--max-bg-boundary", type=int, default=400000)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (0.0 < args.intensity_q < 1.0):
        raise ValueError("--intensity-q must be in (0,1)")

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = _load_cases(args.cache_dir)

    summary: Dict[str, Any] = {
        "cache_dir": str(args.cache_dir),
        "out_dir": str(args.out_dir),
        "intensity_q": float(args.intensity_q),
        "ring_width": int(args.ring_width),
        "cases": [],
    }

    for it in items:
        uid = it["uid"]
        ip = _resolve_cached_path(uid, it["image_pt"], args.cache_dir, "image")
        lp = _resolve_cached_path(uid, it["label_pt"], args.cache_dir, "label")

        img = torch.load(ip, map_location="cpu")  # [C,D,H,W] or [1,D,H,W]
        lab = torch.load(lp, map_location="cpu")  # [D,H,W]

        if img.ndim != 4:
            raise RuntimeError(f"{uid}: image must be [C,D,H,W], got {tuple(img.shape)}")
        if lab.ndim != 3:
            raise RuntimeError(f"{uid}: label must be [D,H,W], got {tuple(lab.shape)}")
        if img.shape[1:] != lab.shape:
            raise RuntimeError(f"{uid}: shape mismatch img={tuple(img.shape)} lab={tuple(lab.shape)}")

        # use channel 0 for sampling intensity; if you cache multi-channel, this remains stable
        img0 = img[0].float()  # [D,H,W]
        lab_bin = (lab > 0)

        # fg coords
        fg = torch.nonzero(lab_bin, as_tuple=False).numpy().astype(np.int32, copy=False)
        fg = _cap_coords(fg, args.max_fg, rng)

        # background masks
        bg_mask = (~lab_bin).numpy()

        # boundary ring background
        fg_mask_5d = lab_bin[None, None, ...]  # [1,1,D,H,W]
        ring = _ring_around_fg(fg_mask_5d, args.ring_width).numpy()
        bg_boundary_mask = ring & bg_mask

        # hard negatives on background (exclude boundary ring to avoid over-focusing on edges)
        bg_for_thr = bg_mask & (~bg_boundary_mask)
        if bg_for_thr.any():
            thr = float(np.quantile(img0.numpy()[bg_for_thr], args.intensity_q))
            hardneg_mask = bg_mask & (img0.numpy() >= thr)
        elif bg_mask.any():
            # fallback: if everything is boundary, compute on all bg
            thr = float(np.quantile(img0.numpy()[bg_mask], args.intensity_q))
            hardneg_mask = bg_mask & (img0.numpy() >= thr)
        else:
            thr = float("nan")
            hardneg_mask = np.zeros_like(bg_mask, dtype=bool)

        # easy bg: bg excluding ring and hardneg
        bg_easy_mask = bg_mask & (~bg_boundary_mask) & (~hardneg_mask)

        bg_boundary = _sample_from_mask(bg_boundary_mask, args.max_bg_boundary, rng)
        hardneg = _sample_from_mask(hardneg_mask, args.max_hardneg, rng)
        bg_easy = _sample_from_mask(bg_easy_mask, args.max_bg_easy, rng)

        out = args.out_dir / f"{uid}.npz"
        np.savez_compressed(
            out,
            fg=fg,
            bg_easy=bg_easy,
            bg_boundary=bg_boundary,
            hardneg=hardneg,
            shape=np.array(lab.shape, dtype=np.int32),
        )

        rec = {
            "uid": uid,
            "shape": list(lab.shape),
            "fg_n": int(fg.shape[0]),
            "bg_easy_n": int(bg_easy.shape[0]),
            "bg_boundary_n": int(bg_boundary.shape[0]),
            "hardneg_n": int(hardneg.shape[0]),
            "hardneg_thr_q": float(args.intensity_q),
            "hardneg_thr_val": float(thr) if np.isfinite(thr) else None,
        }
        summary["cases"].append(rec)
        print(json.dumps(rec), flush=True)

    (args.out_dir / "sampling_db_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Wrote {len(summary['cases'])} .npz files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
