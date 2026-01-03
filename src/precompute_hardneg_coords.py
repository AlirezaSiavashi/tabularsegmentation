#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def resolve(uid, p, cache_dir, kind):
    pp = Path(p)
    if pp.exists(): return pp
    alt = cache_dir / ("images" if kind=="image" else "labels") / f"{uid}.pt"
    return alt

def main(cache_dir: str, out_dir: str, radius: int = 7, n_coords: int = 20000, seed: int = 42):
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    js = json.loads((cache_dir/"cases.json").read_text())
    uids = js["uids"]; labels = js["labels"]

    rng = np.random.default_rng(seed)

    for uid in uids:
        lp = resolve(uid, labels[uid], cache_dir, "label")
        lab = torch.load(lp, map_location="cpu").to(torch.uint8)  # [D,H,W]
        fg = (lab > 0)
        idx = fg.nonzero(as_tuple=False)
        if idx.numel() == 0:
            torch.save(torch.empty((0,3), dtype=torch.int16), out_dir/f"{uid}.pt")
            continue

        D,H,W = lab.shape
        coords = []

        # sample around fg points with random offsets; keep only BG
        for _ in range(n_coords * 3):  # oversample then filter
            p = idx[int(rng.integers(0, idx.shape[0]))]
            cz, cy, cx = int(p[0]), int(p[1]), int(p[2])

            dz, dy, dx = rng.normal(size=3)
            norm = (dz*dz + dy*dy + dx*dx) ** 0.5 + 1e-8
            dz, dy, dx = dz/norm, dy/norm, dx/norm
            dist = int(rng.integers(1, radius+1))

            nz = int(np.clip(cz + dz*dist, 0, D-1))
            ny = int(np.clip(cy + dy*dist, 0, H-1))
            nx = int(np.clip(cx + dx*dist, 0, W-1))

            if lab[nz,ny,nx].item() == 0:
                coords.append((nz,ny,nx))
                if len(coords) >= n_coords:
                    break

        arr = torch.tensor(coords, dtype=torch.int16) if coords else torch.empty((0,3), dtype=torch.int16)
        torch.save(arr, out_dir/f"{uid}.pt")

        print(f"{uid}: saved {arr.shape[0]} hardneg coords -> {out_dir/f'{uid}.pt'}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.cache_dir, args.out_dir, radius=args.radius, n_coords=args.n, seed=args.seed)
