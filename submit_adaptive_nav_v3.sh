#!/bin/bash
#SBATCH --job-name=adap_nav_v3
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

echo "=== Adaptive Nav v3 (snake-u1, ctx-halo-2, lr-restart): $(date) ==="
echo "=== Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES ==="

$PYTHON -c "
from mamba_ssm import Mamba
import torch
print('mamba_ssm OK | torch', torch.__version__, '| cuda', torch.version.cuda)
"

cd $WORKDIR

# Sanity: verify v2 best.pt exists and has fullvol>=0.71
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================================
# Adaptive Navigator v3 — architectural improvements over v2:
#
# CHANGE 1: snake-k-u1 0 -> 3  (was disabled at u1)
#   - u1 operates at native 0.5mm resolution: finest scale in the decoder
#   - Pcom diameter ~1-2mm = 2-4 voxels: snake conv at u1 can trace these
#   - snake_k=3 means 3-step chain (K=3 offsets), minimal memory overhead
#   - New params random-init, loaded via --init-mode match (strict=False)
#   - base_ch=28 → u1 has 28 channels: tiny snake overhead (~0.5 GB extra)
#
# CHANGE 2: ctx-halo-tokens 0 -> 2  (was disabled)
#   - TokenHaloMixer: depthwise Conv3d(128,128,k=3) groups=128
#   - Zero-initialized → starts as identity, learned residual
#   - Allows each token to absorb info from ±2 neighboring tokens
#   - Smooths context boundaries between adjacent patches at decode time
#   - ~200K new params, zero-init → safe warm start
#
# CHANGE 3: lr 5e-5 -> 7e-5, warmup-epochs 3 -> 8
#   - v2 was at 1.3e-5 by epoch 49 (decayed too far)
#   - Reset + slight bump to allow new snake-u1 and halo params to learn
#   - warmup=8 gives new components 8 epochs to stabilize before full LR
#
# CHANGE 4: patches-per-volume 2 -> 3
#   - More spatial diversity per epoch, especially for thin-vessel regions
#   - At 160^3 patches: 3 patches ≈ 30.7 GB (within 40 GB budget with bf16)
#
# Architecture: IDENTICAL to v2 except u1 snake and ctx halo
# Init: navbrush_adaptive_v2/best.pt (fullvol=0.7174, epoch 30)
# Loading: --init-mode match (strict=False)
#   - All existing keys load from v2
#   - local.u1.snake.* (new snake at u1): random kaiming-init
#   - local.ctx_halo_mixer.dw.weight (new halo mixer): zero-init (safe)
#
# Memory estimate:
#   v2 baseline:         31.0 GB (epoch 49)
#   + snake at u1:       +0.5 GB (28ch * K=3 * 160^3 activations in bf16)
#   + halo mixer:        +0.1 GB (128ch depthwise, token-resolution 40^3)
#   + extra patch (x3):  +4.5 GB (160^3 patch, 28ch, bf16)
#   Total estimate:      ~36 GB — safe within 40 GB
# =========================================================================

$PYTHON src/train.py \
  --cache-dir         data/cache_augmented \
  --out-dir           logs/navbrush_adaptive_v3 \
  --init-ckpt         logs/navbrush_adaptive_v2/best.pt \
  --init-mode         match \
  --patch-size        160 160 160 \
  --patches-per-volume 3 \
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
  --snake-k-u1        3 \
  --gn-groups         14 \
  --dropout           0.15 \
  --mamba-dropout     0.05 \
  --ctx-inject        u4,u3,u2,u1 \
  --ctx-halo-tokens   2 \
  --ctx-halo-kernel   3 \
  --norm-type         instance \
  --residual-encoder \
  --encoder-blocks    1 2 3 4 4 \
  --epochs            200 \
  --lr                7e-5 \
  --weight-decay      1e-4 \
  --warmup-epochs     8 \
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
  --val-fullvol-every 5 \
  --val-fullvol-max-cases 8

echo "=== Done: $(date) ==="
