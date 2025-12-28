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


def _maybe_squeeze_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    return arr


def _normalize01(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # robust normalization into [0,1]
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)
    img = img - img.min()
    den = img.max()
    if den < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img / den).astype(np.float32)


def _nib_load_canonical(path: Path) -> nib.Nifti1Image:
    nii = nib.load(str(path))
    # make orientation consistent (RAS)
    nii = nib.as_closest_canonical(nii)
    return nii


def _to_dhw(arr_xyz: np.ndarray) -> np.ndarray:
    """
    nibabel canonical gives array in (X, Y, Z).
    We convert to (D, H, W) = (Z, Y, X) to match your training code.
    """
    return np.transpose(arr_xyz, (2, 1, 0)).copy()


def _spacing_dhw_from_nifti(nii: nib.Nifti1Image) -> Tuple[float, float, float]:
    """
    NIfTI pixdim is for (X,Y,Z). After transpose to (Z,Y,X), spacing becomes (Z,Y,X).
    """
    hdr = nii.header
    sp_xyz = hdr.get_zooms()[:3]  # (sx, sy, sz) in mm
    sx, sy, sz = float(sp_xyz[0]), float(sp_xyz[1]), float(sp_xyz[2])
    return (sz, sy, sx)  # (dz, dy, dx)


def _resample_dhw_torch(
    img: torch.Tensor,
    lab: torch.Tensor,
    spacing_dhw: Tuple[float, float, float],
    target_spacing_dhw: Tuple[float, float, float],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float, float]]:
    """
    img: [1,D,H,W] float
    lab: [D,H,W] uint8/long
    """
    dz, dy, dx = spacing_dhw
    tdz, tdy, tdx = target_spacing_dhw

    # scale factors: new_size = old_size * (old_spacing / target_spacing)
    sd = dz / tdz
    sh = dy / tdy
    sw = dx / tdx

    _, D, H, W = img.shape
    newD = max(1, int(round(D * sd)))
    newH = max(1, int(round(H * sh)))
    newW = max(1, int(round(W * sw)))

    img4 = img.unsqueeze(0)  # [B=1,C=1,D,H,W]
    lab4 = lab.unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]

    img_rs = F.interpolate(img4, size=(newD, newH, newW), mode="trilinear", align_corners=False)
    lab_rs = F.interpolate(lab4, size=(newD, newH, newW), mode="nearest")

    img_out = img_rs.squeeze(0)          # [1,D,H,W]
    lab_out = lab_rs.squeeze(0).squeeze(0).to(torch.uint8)  # [D,H,W]

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


@torch.no_grad()
def process_one(
    item: Dict[str, str],
    out_img_dir: Path,
    out_lab_dir: Path,
    dtype: str,
    target_spacing: Optional[Tuple[float, float, float]],
) -> Dict[str, Any]:
    uid = item["uid"]
    img_path = Path(item["image"])
    lab_path = Path(item["label"])

    img_nii = _nib_load_canonical(img_path)
    lab_nii = _nib_load_canonical(lab_path)

    img = np.asarray(img_nii.dataobj).astype(np.float32)
    lab = np.asarray(lab_nii.dataobj).astype(np.int64)

    img = _maybe_squeeze_4d(img)
    lab = _maybe_squeeze_4d(lab)

    if img.ndim != 3 or lab.ndim != 3:
        raise RuntimeError(f"Expected 3D volumes, got img={img.shape}, lab={lab.shape} for uid={uid}")

    spacing_dhw = _spacing_dhw_from_nifti(img_nii)

    img = _to_dhw(img)
    lab = _to_dhw(lab).astype(np.uint8)

    img = _normalize01(img)

    img_t = torch.from_numpy(img)[None, ...]  # [1,D,H,W]
    lab_t = torch.from_numpy(lab)            # [D,H,W]

    # optional resample
    if target_spacing is not None:
        img_t, lab_t, spacing_dhw = _resample_dhw_torch(img_t, lab_t, spacing_dhw, target_spacing)

    # choose dtype
    if dtype == "fp16":
        img_t = img_t.to(torch.float16)
    else:
        img_t = img_t.to(torch.float32)

    lab_t = lab_t.to(torch.uint8)

    # stats
    fg = int((lab_t > 0).sum().item())
    vox = int(lab_t.numel())
    fg_ratio = float(fg) / float(max(1, vox))

    out_img = out_img_dir / f"{uid}.pt"
    out_lab = out_lab_dir / f"{uid}.pt"
    torch.save(img_t.contiguous(), out_img)
    torch.save(lab_t.contiguous(), out_lab)

    return {
        "uid": uid,
        "image_pt": str(out_img),
        "label_pt": str(out_lab),
        "shape_dhw": list(lab_t.shape),
        "spacing_dhw": [float(x) for x in spacing_dhw],
        "fg_voxels": fg,
        "fg_ratio": fg_ratio,
        "dtype": dtype,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True, help="nnUNet_raw-like dataset directory")
    ap.add_argument("--out-dir", type=Path, required=True, help="cache output directory")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="cache image dtype")
    ap.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("DZ", "DY", "DX"),
        help="optional resample spacing in DHW (mm): dz dy dx",
    )
    ap.add_argument("--max-cases", type=int, default=None)
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

    meta: Dict[str, Any] = {
        "source_dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "dtype": args.dtype,
        "target_spacing_dhw": list(target_spacing) if target_spacing is not None else None,
        "num_cases": len(items),
        "cases": [],
    }

    results = []
    for it in tqdm(items, desc="Caching"):
        r = process_one(it, out_img_dir, out_lab_dir, args.dtype, target_spacing)
        results.append(r)

    meta["cases"] = results

    # lightweight index file for training
    cases_json = {
        "uids": [r["uid"] for r in results],
        "images": {r["uid"]: r["image_pt"] for r in results},
        "labels": {r["uid"]: r["label_pt"] for r in results},
    }

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (args.out_dir / "cases.json").write_text(json.dumps(cases_json, indent=2))

    # quick summary
    fg_ratios = [r["fg_ratio"] for r in results]
    print("Done caching.")
    print(f"  cases: {len(results)}")
    print(f"  fg_ratio: mean={float(np.mean(fg_ratios)):.6f}  median={float(np.median(fg_ratios)):.6f}")
    print(f"  wrote: {args.out_dir/'cases.json'} and {args.out_dir/'meta.json'}")


if __name__ == "__main__":
    main()