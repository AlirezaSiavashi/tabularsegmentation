#!/bin/bash
#SBATCH --job-name=adap_nav_v2
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=192:00:00
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/rsna2025_1st_place/logs/slurm_adaptive_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/rsna2025_1st_place/logs/slurm_adaptive_%j.log

set -euo pipefail

PYTHON=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/aneur/bin/python
WORKDIR=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/rsna2025_1st_place/mamba_snake

echo "=== Adaptive Nav v2 (patch=160, gate-reg=0.05): $(date) ==="
echo "=== Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES ==="

$PYTHON -c "
from mamba_ssm import Mamba
import torch
print('mamba_ssm OK | torch', torch.__version__, '| cuda', torch.version.cuda)
"

cd $WORKDIR

# Sanity: verify best.pt exists and has fullvol=0.6939
$PYTHON -c "
import torch, json, sys
ckpt = torch.load('logs/navbrush_adaptive_joint/best.pt', map_location='cpu', weights_only=False)
fv = ckpt.get('best_fullvol', -1)
pd = ckpt.get('best_patch', -1)
print(f'Resuming from: fullvol={fv:.4f}  patch={pd:.4f}')
assert fv > 0.68, f'Expected fullvol>0.68 but got {fv}'
print('Checkpoint OK')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================================
# Adaptive Navigator v2 — fixes for plateau:
#
# FIX 1: patch-size 128->160 (+56% context per patch)
#   - 15mm Pcom = 30 voxels at 0.5mm: fits entirely in 160^3 without cut
#   - 128^3 often cuts Pcom at boundary → fragmented gradient signal
#   - Memory estimate: ~28 GB peak (fits comfortably in 40 GB)
#
# FIX 2: gate-reg-w 0.0->0.05 (restores FiLM gate selectivity)
#   - Current run: attn_gate vessel=1.000, bg=0.997, gap=0.003 (BROKEN)
#   - Target:      attn_gate vessel≈1.000, bg≈0.04, gap≈0.96
#   - Gate-reg penalizes gate>0 on background voxels
#   - Without this, QueryGuidedFiLM3D saturates → navigator context lost
#
# Init: navbrush_adaptive_joint/best.pt (fullvol=0.6939, new best)
# Architecture: IDENTICAL to job 1420 (all v9 params match)
# =========================================================================

$PYTHON src/train.py \
  --cache-dir         data/cache_augmented \
  --out-dir           logs/navbrush_adaptive_v2 \
  --init-ckpt         logs/navbrush_adaptive_joint/best.pt \
  --init-mode         match \
  --patch-size        160 160 160 \
  --patches-per-volume 2 \
  --base-ch           28 \
  --tok-dim           128 \
  --nav-dim           48 \
  --nav-down          1 \
  --nav-token-stride  4 \
  --nav-mamba-layers  4 \
  --nav-mamba-axes    "dhw,hwd,wdh" \
  --snake-k           5 \
  --snake-k-u4        3 \
  --snake-k-u3        5 \
  --snake-k-u2        0 \
  --snake-k-u1        0 \
  --gn-groups         14 \
  --dropout           0.15 \
  --mamba-dropout     0.05 \
  --ctx-inject        u4,u3,u2,u1 \
  --ctx-halo-tokens   0 \
  --ctx-halo-kernel   3 \
  --norm-type         instance \
  --residual-encoder \
  --encoder-blocks    1 2 3 4 4 \
  --epochs            200 \
  --lr                5e-5 \
  --weight-decay      1e-4 \
  --warmup-epochs     3 \
  --amp-dtype         bf16 \
  --grad-checkpoint \
  --num-workers       2 \
  --nav-w             0.15 \
  --nav-cl-w          0.10 \
  --nav-conn-w        0.10 \
  --nav-ce-fg-w       30.0 \
  --thin-w            0.3 \
  --gate-reg-w        0.05 \
  --skeleton-recall \
  --strong-aug \
  --val-fullvol-every 10 \
  --val-fullvol-max-cases 8

echo "=== Done: $(date) ==="
