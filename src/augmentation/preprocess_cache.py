from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _strip_singleton(arr: np.ndarray) -> np.ndarray:
    # Handle (X,Y,Z,1) or (1,X,Y,Z)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    return arr


def _nib_load_canonical(path: Path) -> nib.Nifti1Image:
    nii = nib.load(str(path))
    # force consistent orientation (RAS)
    return nib.as_closest_canonical(nii)


def _to_dhw(arr_xyz: np.ndarray) -> np.ndarray:
    # (X,Y,Z) -> (Z,Y,X)
    return np.transpose(arr_xyz, (2, 1, 0)).copy()


def _spacing_dhw_from_nifti(nii: nib.Nifti1Image) -> Tuple[float, float, float]:
    # NIfTI zooms in (X,Y,Z). After transpose to (Z,Y,X) => (Z,Y,X)
    sx, sy, sz = map(float, nii.header.get_zooms()[:3])
    return (sz, sy, sx)


def _mask_for_stats(img: np.ndarray) -> np.ndarray:
    # Robust mask for z-score stats: keep voxels above very low percentile
    # (works for both HU and normalized MR-like)
    lo = np.percentile(img, 1.0)
    return img > lo


def _normalize(
    img: np.ndarray,
    mode: str,
    clip_pcts: Tuple[float, float],
    window: Optional[Tuple[float, float]],
    eps: float = 1e-6,
) -> np.ndarray:
    img = img.astype(np.float32, copy=False)

    if mode == "robust01":
        lo, hi = np.percentile(img, clip_pcts)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + eps:
            return np.zeros_like(img, dtype=np.float32)
        img = np.clip(img, lo, hi)
        img = img - lo
        img = img / max(eps, (hi - lo))
        return img.astype(np.float32)

    if mode == "zscore":
        m = _mask_for_stats(img)
        if not m.any():
            return np.zeros_like(img, dtype=np.float32)
        mu = float(img[m].mean())
        sd = float(img[m].std())
        sd = max(sd, eps)
        img = (img - mu) / sd
        # soft clip extreme outliers for stability
        img = np.clip(img, -8.0, 8.0)
        return img.astype(np.float32)

    if mode == "window":
        if window is None:
            raise ValueError("window mode requires --window-min and --window-max")
        wmin, wmax = window
        if wmax <= wmin + eps:
            raise ValueError("Invalid window: max must be > min")
        img = np.clip(img, wmin, wmax)
        img = (img - wmin) / (wmax - wmin)
        return img.astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode}")


def _resample_dhw_torch(
    img: torch.Tensor,            # [C,D,H,W]
    lab: torch.Tensor,            # [D,H,W]
    spacing_dhw: Tuple[float, float, float],
    target_spacing_dhw: Tuple[float, float, float],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float, float]]:
    dz, dy, dx = spacing_dhw
    tdz, tdy, tdx = target_spacing_dhw

    sd = dz / tdz
    sh = dy / tdy
    sw = dx / tdx

    C, D, H, W = img.shape
    newD = max(1, int(round(D * sd)))
    newH = max(1, int(round(H * sh)))
    newW = max(1, int(round(W * sw)))

    img5 = img.unsqueeze(0)  # [1,C,D,H,W]
    lab5 = lab.unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]

    img_rs = F.interpolate(img5, size=(newD, newH, newW), mode="trilinear", align_corners=False)
    lab_rs = F.interpolate(lab5, size=(newD, newH, newW), mode="nearest")

    img_out = img_rs.squeeze(0).contiguous()                       # [C,D,H,W]
    lab_out = lab_rs.squeeze(0).squeeze(0).to(torch.uint8).contiguous()  # [D,H,W]
    return img_out, lab_out, target_spacing_dhw


def list_cases(dataset_dir: Path) -> List[Dict[str, str]]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise RuntimeError(f"Missing imagesTr/labelsTr in {dataset_dir}")

    items: List[Dict[str, str]] = []
    for img_path in sorted(images_dir.glob("*_0000.nii.gz")):
        uid = img_path.name.replace("_0000.nii.gz", "")
        lab_path = labels_dir / f"{uid}.nii.gz"
        if lab_path.exists():
            items.append({"uid": uid, "image": str(img_path), "label": str(lab_path)})
    if not items:
        raise RuntimeError(f"No training pairs found under {images_dir} and {labels_dir}")
    return items


def _parse_multi_windows(vals: List[float]) -> List[Tuple[float, float]]:
    if len(vals) % 2 != 0:
        raise ValueError("--multi-window must be pairs: min1 max1 min2 max2 ...")
    out = []
    for i in range(0, len(vals), 2):
        wmin, wmax = float(vals[i]), float(vals[i + 1])
        if wmax <= wmin:
            raise ValueError(f"Invalid window pair: {wmin} {wmax}")
        out.append((wmin, wmax))
    return out


@torch.no_grad()
def process_one(
    item: Dict[str, str],
    out_img_dir: Path,
    out_lab_dir: Path,
    dtype: str,
    target_spacing: Optional[Tuple[float, float, float]],
    norm: str,
    clip_pcts: Tuple[float, float],
    window: Optional[Tuple[float, float]],
    multi_windows: Optional[List[Tuple[float, float]]],
) -> Dict[str, Any]:
    uid = item["uid"]
    img_path = Path(item["image"])
    lab_path = Path(item["label"])

    img_nii = _nib_load_canonical(img_path)
    lab_nii = _nib_load_canonical(lab_path)

    img = np.asarray(img_nii.dataobj).astype(np.float32)
    lab = np.asarray(lab_nii.dataobj).astype(np.int64)

    img = _strip_singleton(img)
    lab = _strip_singleton(lab)

    if img.ndim != 3 or lab.ndim != 3:
        raise RuntimeError(f"{uid}: expected 3D volumes, got img={img.shape}, lab={lab.shape}")
    if img.shape != lab.shape:
        raise RuntimeError(f"{uid}: image/label shape mismatch img={img.shape} lab={lab.shape}")

    spacing_dhw = _spacing_dhw_from_nifti(img_nii)

    img = _to_dhw(img)
    lab = _to_dhw(lab).astype(np.uint8, copy=False)

    # normalize (single or multi-channel)
    if multi_windows is not None and len(multi_windows) > 0:
        chans = []
        for w in multi_windows:
            chans.append(_normalize(img, mode="window", clip_pcts=clip_pcts, window=w))
        img_c = np.stack(chans, axis=0)  # [C,D,H,W]
    else:
        img_n = _normalize(img, mode=norm, clip_pcts=clip_pcts, window=window)
        img_c = img_n[None, ...]  # [1,D,H,W]

    img_t = torch.from_numpy(img_c).contiguous()     # [C,D,H,W]
    lab_t = torch.from_numpy(lab).contiguous()       # [D,H,W]

    if target_spacing is not None:
        img_t, lab_t, spacing_dhw = _resample_dhw_torch(img_t, lab_t, spacing_dhw, target_spacing)

    # choose dtype
    if dtype == "fp16":
        img_t = img_t.to(torch.float16)
    else:
        img_t = img_t.to(torch.float32)
    lab_t = lab_t.to(torch.uint8)

    fg = int((lab_t > 0).sum().item())
    vox = int(lab_t.numel())
    fg_ratio = float(fg) / float(max(1, vox))

    out_img = out_img_dir / f"{uid}.pt"
    out_lab = out_lab_dir / f"{uid}.pt"
    torch.save(img_t, out_img)
    torch.save(lab_t, out_lab)

    return {
        "uid": uid,
        "image_pt": str(out_img),
        "label_pt": str(out_lab),
        "shape_dhw": list(lab_t.shape),
        "spacing_dhw": [float(x) for x in spacing_dhw],
        "fg_voxels": fg,
        "fg_ratio": fg_ratio,
        "dtype": dtype,
        "norm": norm if multi_windows is None else "multi-window",
        "multi_windows": multi_windows,
        "clip_pcts": list(clip_pcts),
        "window": list(window) if window is not None else None,
        "channels": int(img_t.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True, help="nnUNet_raw-like dataset directory")
    ap.add_argument("--out-dir", type=Path, required=True, help="cache output directory")

    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32"],
                    help="cache image dtype (fp32 recommended for vessels)")
    ap.add_argument("--target-spacing", type=float, nargs=3, default=None, metavar=("DZ", "DY", "DX"),
                    help="optional resample spacing in DHW (mm): dz dy dx")
    ap.add_argument("--max-cases", type=int, default=None)

    ap.add_argument("--norm", type=str, default="robust01", choices=["robust01", "zscore", "window"],
                    help="normalization mode")
    ap.add_argument("--clip-pcts", type=float, nargs=2, default=(0.1, 99.9),
                    metavar=("LO", "HI"), help="percentiles for robust01 clipping")
    ap.add_argument("--window-min", type=float, default=None)
    ap.add_argument("--window-max", type=float, default=None)

    ap.add_argument("--multi-window", type=float, nargs="*", default=None,
                    help="optional multi-window channels: min1 max1 min2 max2 ... (uses window norm)")

    args = ap.parse_args()

    items = list_cases(args.dataset_dir)
    if args.max_cases is not None:
        items = items[: args.max_cases]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = args.out_dir / "images"
    out_lab_dir = args.out_dir / "labels"
    out_img_dir.mkdir(exist_ok=True)
    out_lab_dir.mkdir(exist_ok=True)

    target_spacing = tuple(args.target_spacing) if args.target_spacing is not None else None
    window = None
    if args.window_min is not None or args.window_max is not None:
        if args.window_min is None or args.window_max is None:
            raise ValueError("Provide both --window-min and --window-max")
        window = (float(args.window_min), float(args.window_max))

    multi_windows = None
    if args.multi_window is not None and len(args.multi_window) > 0:
        multi_windows = _parse_multi_windows(args.multi_window)

    meta: Dict[str, Any] = {
        "source_dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "dtype": args.dtype,
        "target_spacing_dhw": list(target_spacing) if target_spacing is not None else None,
        "norm": args.norm if multi_windows is None else "multi-window",
        "clip_pcts": list(map(float, args.clip_pcts)),
        "window": list(window) if window is not None else None,
        "multi_windows": multi_windows,
        "num_cases": len(items),
        "cases": [],
    }

    results = []
    for it in tqdm(items, desc="Caching"):
        r = process_one(
            it, out_img_dir, out_lab_dir,
            dtype=args.dtype,
            target_spacing=target_spacing,
            norm=args.norm,
            clip_pcts=(float(args.clip_pcts[0]), float(args.clip_pcts[1])),
            window=window,
            multi_windows=multi_windows,
        )
        results.append(r)

    meta["cases"] = results

    cases_json = {
        "uids": [r["uid"] for r in results],
        "images": {r["uid"]: r["image_pt"] for r in results},
        "labels": {r["uid"]: r["label_pt"] for r in results},
    }

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (args.out_dir / "cases.json").write_text(json.dumps(cases_json, indent=2))

    fg_ratios = [r["fg_ratio"] for r in results]
    print("Done caching.")
    print(f"  cases: {len(results)}")
    print(f"  fg_ratio: mean={float(np.mean(fg_ratios)):.6f}  median={float(np.median(fg_ratios)):.6f}")
    print(f"  wrote: {args.out_dir/'cases.json'} and {args.out_dir/'meta.json'}")


if __name__ == "__main__":
    main()
