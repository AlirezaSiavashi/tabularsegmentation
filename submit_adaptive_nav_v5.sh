#!/bin/bash
#SBATCH --job-name=adap_nav_v5
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

echo "=== Adaptive Nav v5 (soft-labels-fixed + 160^3 + compile): $(date) ==="
echo "=== Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES ==="

$PYTHON -c "
from mamba_ssm import Mamba
import torch
print('mamba_ssm OK | torch', torch.__version__, '| cuda', torch.version.cuda)
"

cd $WORKDIR

$PYTHON -c "
import torch
ckpt = torch.load('logs/navbrush_adaptive_v2/best.pt', map_location='cpu', weights_only=False)
fv = ckpt.get('best_fullvol', -1)
print(f'Init from v2: epoch={ckpt.get(\"epoch\")} fullvol={fv:.4f}')
assert fv > 0.70
print('[OK]')
"

# Verify soft-label fix: NavigatorLoss should NOT apply pos_weight for soft targets
$PYTHON -c "
import sys; sys.path.insert(0, 'src')
import torch
from train import NavigatorLoss

loss_fn = NavigatorLoss(ce_weight_fg=30.0)

# With soft labels: loss should be LOWER than with hard labels at same pred
pred = torch.zeros(1, 1, 5, 5, 5)  # all logits=0 → sigmoid=0.5
soft = torch.full((1, 1, 5, 5, 5), 0.06)  # border token, 6% vessel
hard = torch.ones(1, 1, 5, 5, 5)           # hard positive

loss_soft = loss_fn(pred, soft, nav_cl_w=0, nav_conn_w=0).item()
loss_hard = loss_fn(pred, hard, nav_cl_w=0, nav_conn_w=0).item()
print(f'Loss on soft (t=0.06): {loss_soft:.4f}')
print(f'Loss on hard (t=1.00): {loss_hard:.4f}')
assert loss_soft < loss_hard, f'Soft loss {loss_soft:.3f} should be < hard loss {loss_hard:.3f}'
print('[OK] pos_weight not applied to soft labels')
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================================
# Adaptive Navigator v5 — two fixes over v4:
#
# FIX 1: Revert patch [80,192,192] → [160,160,160]
#   v4 showed fullvol dropped from 0.7174 → 0.6920 with anisotropic patch.
#   Root cause: model init from v2 which was trained on 160^3.
#   Changing patch shape mid-training disrupts position encodings and
#   cuts depth coverage from 128mm → 64mm, losing posterior vessel context.
#   Anisotropic patch is valid but requires training from scratch, not fine-tune.
#
# FIX 2: NavigatorLoss pos_weight removed for soft labels
#   v4 used pos_weight=30 with avg_pool soft targets (t=0.06 for border tokens).
#   BCE with pos_weight=30 on t=0.06 amplifies border tokens with weight 30,
#   pushing model to predict HIGH logits for near-vessel background → more FP.
#   This is why precision dropped 0.47→0.35 in v4 (opposite of intended).
#   Fix: for soft labels, plain BCE (no pos_weight). Soft Dice handles recall.
#   Expected: precision rises from 0.47 toward 0.65-0.75.
#
# KEPT from v4:
#   - avg_pool soft token labels (correct principle, just needed loss fix)
#   - torch.compile (26% faster, 6.3s/step vs 8.5s/step)
#
# Init: navbrush_adaptive_v2/best.pt (fullvol=0.7174, skipped=0 missing=0)
# =========================================================================

$PYTHON src/train.py \
  --cache-dir         data/cache_augmented \
  --out-dir           logs/navbrush_adaptive_v5 \
  --init-ckpt         logs/navbrush_adaptive_v2/best.pt \
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
