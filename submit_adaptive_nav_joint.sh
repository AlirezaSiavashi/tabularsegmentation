#!/bin/bash
#SBATCH --job-name=adaptive_nav
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

echo "=== Adaptive Navigator (Option C): $(date) ==="
echo "=== Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES ==="

$PYTHON -c "
from mamba_ssm import Mamba
import torch
print('mamba_ssm OK | torch', torch.__version__, '| cuda', torch.version.cuda)
"

cd $WORKDIR

# Full sanity suite before training starts
$PYTHON -c "
import sys; sys.path.insert(0,'src')
import torch
from train import (NavBrushModel, ThinVesselConfidenceHead, AdaptiveFinePass,
                   thin_vessel_proximity_label)

# 1. thin_vessel_proximity_label receives [B,D,H,W] — verify fix
yi = torch.zeros(128,128,128, dtype=torch.long)
yi[40:90, 60:62, 60:62] = 1   # thin 2-voxel vessel
yi_4d = yi.unsqueeze(0).long()  # [1,D,H,W]
lbl = thin_vessel_proximity_label(yi_4d, global_stride=4)
assert lbl.shape == (1,1,32,32,32), f'wrong shape {lbl.shape}'
assert lbl.sum() > 0, 'label is all zero — thin_vessel_proximity_label broken'
print(f'[1] thin_vessel_proximity_label OK: {lbl.shape} sum={lbl.sum().item():.0f}')

# 2. v9 checkpoint loads with match mode
import torch
ckpt = torch.load('logs/navbrush_v9_stride4/best.pt', map_location='cpu', weights_only=False)
keys = set(ckpt.get('model', ckpt).keys())
assert 'nav.stem.conv1.weight' in keys, 'not a GlobalNavigator checkpoint'
assert 'thin_head.head.weight' not in keys, 'thin_head should be missing (new)'
print('[2] v9 checkpoint is GlobalNavigator — match loading will add thin_head/adaptive_fine as new')

# 3. ThinVesselConfidenceHead zero-init = safe warm start
head = ThinVesselConfidenceHead(tok_dim=128)
assert head.head.weight.abs().max() == 0, 'head weight not zero-init'
print('[3] ThinVesselConfidenceHead zero-init OK')

# 4. AdaptiveFinePass zero-init proj = identity at epoch 0
from train import AdaptiveFinePass
fine = AdaptiveFinePass(tok_dim=128, fine_mamba_layers=1, sub_size=8, gn_groups=14)
assert fine.proj[0].weight.abs().max() == 0, 'proj not zero-init'
print('[4] AdaptiveFinePass zero-init proj OK')

print('ALL SANITY CHECKS PASSED')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================================
# OPTION C: v9 exact architecture + new thin_head + adaptive_fine
#
# Architecture exactly matches navbrush_v9_stride4/best.pt:
#   - GlobalNavigator (nav.stem + nav.token + nav.mamba) NOT pretrained VMamba
#   - use_residual_encoder=True, encoder_blocks=1,2,3,4,4
#   - gn_groups=14 (matches v9 snake conv channel dims)
#   - norm_type=instance (v9 used instance norm)
#   - ctx_inject=u4,u3,u2,u1  ctx_halo_tokens=0 (v9 settings)
#   - snake_k=5, snake_k_u3=5, snake_k_u4=3, snake_k_u1/u2=0 (v9 settings)
#   - patch_size=128 (v9 used 128, not 160)
#
# New components (initialised from zero → safe warm start):
#   - thin_head: 1×1×1 conv, predicts per-token thin-vessel probability
#   - adaptive_fine: BiMambaSSM on flagged sub-volumes at same resolution
#
# Loading: --init-mode match  →  load_state_dict_match_shapes (strict=False)
#   - v9 weights load for all matching keys
#   - thin_head.* and adaptive_fine.* stay at zero-init (new params)
#   - Training starts from fullvol=0.6691 immediately
#
# thin_w=0.3: BCE loss on thinness head, pos_weight=10 for rare thin vessels
# Soft label fix: yi_4d = yi.unsqueeze(0).long() → [B,D,H,W] (not [1,1,D,H,W])
# =========================================================================

$PYTHON src/train.py \
  --cache-dir         data/cache_augmented \
  --out-dir           logs/navbrush_adaptive_joint \
  --init-ckpt         logs/navbrush_v9_stride4/best.pt \
  --init-mode         match \
  --patch-size        128 128 128 \
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
  --lr                1e-4 \
  --weight-decay      1e-4 \
  --warmup-epochs     10 \
  --amp-dtype         bf16 \
  --grad-checkpoint \
  --num-workers       2 \
  --nav-w             0.15 \
  --nav-cl-w          0.10 \
  --nav-conn-w        0.10 \
  --nav-ce-fg-w       30.0 \
  --thin-w            0.3 \
  --gate-reg-w        0.0 \
  --skeleton-recall \
  --strong-aug \
  --val-fullvol-every 10 \
  --val-fullvol-max-cases 8

echo "=== Done: $(date) ==="
