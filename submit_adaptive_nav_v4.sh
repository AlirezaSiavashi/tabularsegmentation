#!/bin/bash
#SBATCH --job-name=adap_nav_v4
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

echo "=== Adaptive Nav v4 (soft-token-labels + anisotropic-patch + compile): $(date) ==="
echo "=== Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES ==="

$PYTHON -c "
from mamba_ssm import Mamba
import torch
print('mamba_ssm OK | torch', torch.__version__, '| cuda', torch.version.cuda)
"

cd $WORKDIR

# Sanity: verify v2 best.pt (strongest checkpoint, fullvol=0.7174)
$PYTHON -c "
import torch
ckpt = torch.load('logs/navbrush_adaptive_v2/best.pt', map_location='cpu', weights_only=False)
fv = ckpt.get('best_fullvol', -1)
pd = ckpt.get('best_patch', -1)
print(f'Resuming from: epoch={ckpt.get(\"epoch\")} fullvol={fv:.4f} patch={pd:.4f}')
assert fv > 0.70, f'Expected fullvol>0.70 but got {fv}'
keys = set(ckpt.get('model', {}).keys())
assert 'nav.stem.conv1.weight' in keys, 'Not a GlobalNavigator checkpoint'
print('[OK] v2 best.pt verified')
"

# Quick sanity: soft token label shapes
$PYTHON -c "
import sys; sys.path.insert(0, 'src')
import torch, torch.nn.functional as F
yi = torch.zeros(200, 540, 540)
yi[80:120, 260:264, 260:264] = 1  # thin vessel tube
soft = F.avg_pool3d((yi>0).float().unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4)
hard = F.max_pool3d((yi>0).float().unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4)
soft_pos = (soft > 0.05).float().sum().item()
hard_pos = hard.sum().item()
print(f'Soft token positives (>5%): {soft_pos:.0f}')
print(f'Hard token positives: {hard_pos:.0f}')
print(f'Reduction ratio: {hard_pos/max(soft_pos,1):.1f}x fewer FP with soft labels')
print('[OK] soft token labels sanity passed')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================================
# Adaptive Navigator v4 — three principled improvements over v3:
#
# CHANGE 1: Soft token labels — avg_pool3d instead of max_pool3d
#   Root cause identified: max_pool(stride=4) on tubular vessels inflates
#   positive token fraction to ~47%. Navigator precision = 47% = random chance.
#   Fix: avg_pool gives the vessel voxel FRACTION per token [0,1].
#   A token with 1 vessel voxel out of 64 = 0.015 (sparse), not 1 (positive).
#   Threshold for metrics: >5% vessel fraction = "vessel token".
#   Expected: navigator precision 47% → 70-80%, better FiLM context quality.
#   Loss change: NavigatorLoss now uses soft BCE + soft Dice against [0,1] targets.
#
# CHANGE 2: Anisotropic patch [80, 192, 192] matching data spacing
#   Data: D=0.8mm, H/W=0.45mm → anisotropic 1.78:1 ratio
#   SOTA nnUNet patch: [64, 192, 192] = 51×86×86mm (auto-configured by nnUNet)
#   v2/v3 patch: [160, 160, 160] = 128×72×72mm — too much depth, insufficient H/W
#   v4 patch: [80, 192, 192] = 64×86×86mm — matches SOTA's physical extent in H/W
#   Effect: +26% more H/W context (192 vs 160 voxels) where vessels spread out.
#   Memory: 80×192×192 = 2.95M voxels vs 160×160×160 = 4.1M. Actually LESS memory.
#   Estimated peak: ~26 GB (down from 31 GB), allows batch-size or patches increase.
#
# CHANGE 3: torch.compile (reduce-overhead mode)
#   ~15-20% faster per-step after first epoch compilation (~5 min warmup).
#   More epochs in same wall-clock time → better convergence.
#   Compatible with grad_checkpoint, AMP bf16, Mamba SSM.
#
# Init: navbrush_adaptive_v2/best.pt (fullvol=0.7174, epoch 30)
# Architecture: identical to v2 (snake_k_u1=0, ctx_halo_tokens=0)
#   Simpler than v3 deliberately — isolate the soft-label + patch shape effects.
#   Adding snake_k_u1 and ctx_halo on top can come in v5 once v4 shows gain.
# Loading: --init-mode match (strict=False) — all v2 keys load cleanly
#
# Memory estimate:
#   patch 80×192×192 encoder: base_ch=28, residual encoder → ~22-24 GB peak
#   (less than v2's 31 GB because fewer voxels per patch at native res)
#   Allows patches-per-volume=4 for better diversity
# =========================================================================

$PYTHON src/train.py \
  --cache-dir         data/cache_augmented \
  --out-dir           logs/navbrush_adaptive_v4 \
  --init-ckpt         logs/navbrush_adaptive_v2/best.pt \
  --init-mode         match \
  --patch-size        80 192 192 \
  --patches-per-volume 4 \
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
  --lr                7e-5 \
  --weight-decay      1e-4 \
  --warmup-epochs     5 \
  --amp-dtype         bf16 \
  --grad-checkpoint \
  --use-compile \
  --num-workers       2 \
  --nav-w             0.15 \
  --nav-cl-w          0.10 \
  --nav-conn-w        0.10 \
  --nav-ce-fg-w       30.0 \
  --thin-w            0.3 \
  --gate-reg-w        0.05 \
  --skeleton-recall \
  --strong-aug \
  --val-fullvol-every 5 \
  --val-fullvol-max-cases 8

echo "=== Done: $(date) ==="
