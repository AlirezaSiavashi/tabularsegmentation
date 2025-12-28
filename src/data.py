from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

@dataclass
class CaseItem:
    img: Path
    lbl: Path
    uid: str

def _load_nii(path: Path) -> np.ndarray:
    x = nib.load(str(path))
    arr = np.asanyarray(x.dataobj)
    # expected already RAS + normalized by your prep script, but we keep it generic
    return arr.astype(np.float32)

def _load_lbl(path: Path) -> np.ndarray:
    x = nib.load(str(path))
    arr = np.asanyarray(x.dataobj).astype(np.int64)
    return arr

def _random_crop_3d(img: np.ndarray, lbl: np.ndarray, patch: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
    D,H,W = img.shape[-3:]
    pd,ph,pw = patch
    if D <= pd or H <= ph or W <= pw:
        # pad if needed
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        img = np.pad(img, ((0,pad_d),(0,pad_h),(0,pad_w)), mode="constant")
        lbl = np.pad(lbl, ((0,pad_d),(0,pad_h),(0,pad_w)), mode="constant")
        D,H,W = img.shape[-3:]

    sd = np.random.randint(0, D - pd + 1)
    sh = np.random.randint(0, H - ph + 1)
    sw = np.random.randint(0, W - pw + 1)
    img_c = img[sd:sd+pd, sh:sh+ph, sw:sw+pw]
    lbl_c = lbl[sd:sd+pd, sh:sh+ph, sw:sw+pw]
    return img_c, lbl_c

def _random_flip(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # flips over D/H/W randomly
    for axis in [0,1,2]:
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=axis).copy()
            lbl = np.flip(lbl, axis=axis).copy()
    return img, lbl

def _intensity_jitter(img: np.ndarray) -> np.ndarray:
    # simple gamma + scale/shift
    if np.random.rand() < 0.5:
        g = np.random.uniform(0.8, 1.2)
        img = np.power(np.clip(img, 0, 1), g)
    if np.random.rand() < 0.5:
        a = np.random.uniform(0.9, 1.1)
        b = np.random.uniform(-0.05, 0.05)
        img = np.clip(a * img + b, 0, 1)
    if np.random.rand() < 0.3:
        n = np.random.uniform(0.0, 0.03)
        img = np.clip(img + np.random.normal(0, n, size=img.shape).astype(np.float32), 0, 1)
    return img

class NnUNetRaw3DDataset(Dataset):
    """
    Reads nnUNet_raw/DatasetXXX_*/imagesTr/{uid}_0000.nii.gz
                       labelsTr/{uid}.nii.gz
    Returns tensors:
      image: (1,D,H,W) float32
      label: (1,D,H,W) int64
    """
    def __init__(
        self,
        nnunet_raw_dataset_dir: Path,
        split_uids: List[str],
        patch: Tuple[int,int,int],
        training: bool = True,
    ):
        self.root = Path(nnunet_raw_dataset_dir)
        self.imagesTr = self.root / "imagesTr"
        self.labelsTr = self.root / "labelsTr"
        self.patch = patch
        self.training = training

        self.items: List[CaseItem] = []
        for uid in split_uids:
            img = self.imagesTr / f"{uid}_0000.nii.gz"
            lbl = self.labelsTr / f"{uid}.nii.gz"
            if img.exists() and lbl.exists():
                self.items.append(CaseItem(img=img, lbl=lbl, uid=uid))

        if not self.items:
            raise RuntimeError(f"No cases found in {self.root} for the given split.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        img = _load_nii(it.img)  # (D,H,W)
        lbl = _load_lbl(it.lbl)  # (D,H,W)

        # enforce shapes
        if img.ndim != 3 or lbl.ndim != 3:
            raise ValueError(f"Expected 3D volumes, got img {img.shape}, lbl {lbl.shape}")

        if self.training:
            img, lbl = _random_crop_3d(img, lbl, self.patch)
            img, lbl = _random_flip(img, lbl)
            img = _intensity_jitter(img)
        else:
            # validation: center crop/pad to patch (for quick eval) OR keep full volume in val loader
            img, lbl = _random_crop_3d(img, lbl, self.patch)

        img_t = torch.from_numpy(img[None, ...].astype(np.float32))  # (1,D,H,W)
        lbl_t = torch.from_numpy(lbl[None, ...].astype(np.int64))    # (1,D,H,W)
        return {"image": img_t, "label": lbl_t, "uid": it.uid}

def make_splits(all_uids: List[str], fold: int = 0, num_folds: int = 5, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    uids = list(all_uids)
    rng.shuffle(uids)
    folds = np.array_split(uids, num_folds)
    val = list(folds[fold])
    train = [u for u in uids if u not in set(val)]
    return train, val

def discover_uids(nnunet_raw_dataset_dir: Path) -> List[str]:
    imagesTr = Path(nnunet_raw_dataset_dir) / "imagesTr"
    uids = []
    for p in sorted(imagesTr.glob("*_0000.nii.gz")):
        uid = p.name.replace("_0000.nii.gz", "")
        uids.append(uid)
    return uids
