#!/usr/bin/env bash
set -euo pipefail

# ---------
# Paths
# ---------
SRC="${SRC:-/workspace/mamba_snake/data/cache_vesselroi_pt}"
DST="${DST:-/workspace/mamba_snake/data/cache_vesselroi_pt_aug_full}"

# ---------
# Aug count + seed
# ---------
K="${K:-3}"
SEED="${SEED:-123}"

# ---------
# GPU selection
#   GPU 1 is full -> we force visibility to GPU 0 only.
#   IMPORTANT: after remapping, the visible GPU is indexed as cuda:0.
# ---------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"

# ---------
# CoLeTra (optional)
# ---------
P_COLETTRA="${P_COLETTRA:-0.35}"
COLETTRA_PATCH="${COLETTRA_PATCH:-15}"
COLETTRA_NPATCH="${COLETTRA_NPATCH:-2}"
COLETTRA_SIGMA="${COLETTRA_SIGMA:--1}"
COLETTRA_DILATE="${COLETTRA_DILATE:-3}"
COLETTRA_BG_SIGMA="${COLETTRA_BG_SIGMA:-2.0}"

# ---------
# Geometric
# ---------
P_FLIP="${P_FLIP:-0.5}"

P_AFFINE="${P_AFFINE:-0.35}"
ROT_DEG="${ROT_DEG:-10.0}"
SCALE_PCT="${SCALE_PCT:-0.10}"
SHEAR_PCT="${SHEAR_PCT:-0.10}"

P_GRID="${P_GRID:-0.15}"
GRID_ALPHA_VOX="${GRID_ALPHA_VOX:-1.5}"
GRID_COARSE="${GRID_COARSE:-6}"

P_LOWRES="${P_LOWRES:-0.15}"
LOWRES_MIN_SCALE="${LOWRES_MIN_SCALE:-0.55}"

# ---------
# Intensity
# ---------
P_NOISE="${P_NOISE:-0.25}"
NOISE_STD="${NOISE_STD:-0.01}"

P_SMOOTH="${P_SMOOTH:-0.20}"
SMOOTH_SIGMA="${SMOOTH_SIGMA:-0.8}"

P_SHIFT_SCALE="${P_SHIFT_SCALE:-0.30}"
SCALE_MIN="${SCALE_MIN:-0.90}"
SCALE_MAX="${SCALE_MAX:-1.10}"
SHIFT_MIN="${SHIFT_MIN:--0.05}"
SHIFT_MAX="${SHIFT_MAX:-0.05}"

P_CONTRAST="${P_CONTRAST:-0.20}"
CONTRAST_MIN="${CONTRAST_MIN:-0.85}"
CONTRAST_MAX="${CONTRAST_MAX:-1.20}"

P_SHARPEN="${P_SHARPEN:-0.15}"
SHARP_SIGMA="${SHARP_SIGMA:-0.8}"
SHARP_AMT_MIN="${SHARP_AMT_MIN:-0.3}"
SHARP_AMT_MAX="${SHARP_AMT_MAX:-0.8}"

P_INVERT="${P_INVERT:-0.05}"

python /workspace/mamba_snake/src/augmentation_phase.py \
  --src "$SRC" --dst "$DST" \
  --k "$K" --copy-original \
  --seed "$SEED" \
  --device "$DEVICE" \
  --p-flip "$P_FLIP" \
  --p-affine "$P_AFFINE" --rot-deg "$ROT_DEG" --scale-pct "$SCALE_PCT" --shear-pct "$SHEAR_PCT" \
  --p-grid "$P_GRID" --grid-alpha-vox "$GRID_ALPHA_VOX" --grid-coarse "$GRID_COARSE" \
  --p-lowres "$P_LOWRES" --lowres-min-scale "$LOWRES_MIN_SCALE" \
  --p-colettra "$P_COLETTRA" \
  --colettra-patch "$COLETTRA_PATCH" --colettra-npatch "$COLETTRA_NPATCH" --colettra-sigma "$COLETTRA_SIGMA" \
  --colettra-dilate "$COLETTRA_DILATE" --colettra-bg-sigma "$COLETTRA_BG_SIGMA" \
  --p-noise "$P_NOISE" --noise-std "$NOISE_STD" \
  --p-smooth "$P_SMOOTH" --smooth-sigma "$SMOOTH_SIGMA" \
  --p-shift-scale "$P_SHIFT_SCALE" --scale-min "$SCALE_MIN" --scale-max "$SCALE_MAX" --shift-min "$SHIFT_MIN" --shift-max "$SHIFT_MAX" \
  --p-contrast "$P_CONTRAST" --contrast-min "$CONTRAST_MIN" --contrast-max "$CONTRAST_MAX" \
  --p-sharpen "$P_SHARPEN" --sharp-sigma "$SHARP_SIGMA" --sharp-amt-min "$SHARP_AMT_MIN" --sharp-amt-max "$SHARP_AMT_MAX" \
  --p-invert "$P_INVERT"
