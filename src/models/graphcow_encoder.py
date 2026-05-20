"""
GraphCoW-Net — Encoder

Frozen DINOv3 ViT (2D) + Tri-planar 2D->3D Adapter + Multi-scale Feature Pyramid.

Design choices (patch 192^3, bf16):
  * DINOv3 ViT-S/16 frozen: 21M params, embed_dim=384. Small variant chosen because
    we run it 3x per patch (axial, sagittal, coronal) on many slices -- ViT-B would
    triple memory. Features transfer well per MedDINOv3 (arxiv 2509.02379).
   (I need more explanation for that)Tri-planar decomposition: a 3D volume is 3 stacks of 2D slices. Each stack is
    batched through the frozen ViT. The 3 feature volumes are fused via trainable
    3D convs. Bridges 2D pretraining to 3D without retraining the ViT.
  * Multi-scale token aggregation (from MedDINOv3): features are taken at 4 ViT
    depths (3, 6, 9, 12) and upsampled to 3D pyramid levels F1..F5 at strides
    {2, 4, 8, 16, 32}. F1 comes from a shallow 3D stem (not ViT) because patch=16
    is too coarse for stride-2. (I should check how we choose these value)
  * Everything downstream of the ViT is trainable. ViT uses forward-no-grad.

Inputs:
  x:   [B, 1, D, H, W]    CT/MRA volume patch, usually 192^3
  We require D, H, W divisible by 32 (for stride-32 feature map).
  We require isotropic voxel spacing (dataloader resamples to e.g. 0.5 mm^3).
  The tri-planar ViT pass does NOT stretch slices -- it reflection-pads to a
  multiple of the ViT patch size (16), preserving aspect ratio so that a
  vessel of diameter d voxels looks identical in axial/sagittal/coronal views.

Outputs:
  Dict with keys F1..F5, each [B, C_l, d_l, h_l, w_l]:
    F1: stride 2,  C=32   (from stem, pre-ViT)
    F2: stride 4,  C=64
    F3: stride 8,  C=128
    F4: stride 16, C=256  (ViT native stride)
    F5: stride 32, C=384

Memory budget (bs=1, bf16, 192^3, no grad-ckpt on ViT since frozen):
    ViT ViT-S forward:    ~1.4 GB  (3 planes x ~12 slabs of 8 slices)
    Tri-planar fusion:    ~2.5 GB
    Pyramid features:     ~3.0 GB
    Stem + adapter grads: ~2.0 GB
    Total encoder:        ~9 GB  -> leaves ~30 GB for decoder + graph head.

References:
  DINOv3: https://arxiv.org/abs/... (Meta AI 2025)
  MedDINOv3: https://arxiv.org/abs/2509.02379
  Extending 2D DINOv3 to 3D: https://arxiv.org/abs/2602.23962
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DINOv3 loading
# ---------------------------------------------------------------------------

# ViT-S/14 or ViT-S/16: embed_dim 384, 12 heads, 12 blocks.
# ViT-B/16: embed_dim 768, 12 heads, 12 blocks.
# DINOv3 variants require a HuggingFace access token (gated repo). If the
# token is unavailable, the DINOv2-with-registers variants are a drop-in
# architectural fallback (same embed_dim, same number of register tokens
# per HF config). MedDINOv3 weights -- when the user obtains them -- can
# be loaded into any variant with matching dims via state_dict surgery.
_DINOV3_VARIANTS = {
    # DINOv3 (gated). Patch size 16.
    "vits16":  {"hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m", "embed_dim": 384, "depth": 12, "patch": 16},
    "vitb16":  {"hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m", "embed_dim": 768, "depth": 12, "patch": 16},
    # DINOv2 + registers (open). Patch size 14, same embed_dim/depth.
    "v2s14":   {"hf_id": "facebook/dinov2-with-registers-small",    "embed_dim": 384, "depth": 12, "patch": 14},
    "v2b14":   {"hf_id": "facebook/dinov2-with-registers-base",     "embed_dim": 768, "depth": 12, "patch": 14},
    # DINOv2 (open, no registers). Fallback if even the registers variant
    # is unreachable.
    "v2s14_noreg": {"hf_id": "facebook/dinov2-small",                "embed_dim": 384, "depth": 12, "patch": 14},
}


def _load_dinov3(
    variant: str,
    cache_dir: Optional[str] = None,
    unfreeze_top_k: int = 0,
):
    """Load a DINOv3 / DINOv2 ViT via HuggingFace transformers.

    unfreeze_top_k:
        Number of transformer blocks to keep trainable, counted from the
        deepest block downward. For ViT-S with 12 blocks:
          * 0  -> fully frozen (default; cheapest but backbone cannot
                  adapt to MRA vessels).
          * 2  -> unfreeze blocks 10, 11 (~3.5M params, well-transferred
                  from natural images, tune on medical signal). Standard
                  recipe from MAE/DINO downstream fine-tune papers.
          * 4  -> unfreeze blocks 8..11 (more capacity, +7M params).
          * -1 -> unfreeze everything (rarely needed for this task).
        The final LayerNorm is always unfrozen together with the top blocks
        since the two are inseparable in HF's ViT implementation.

    Returns (model, cfg). The caller is responsible for remembering how
    many params were unfrozen (via `model.num_unfrozen_blocks`).
    """
    from transformers import AutoModel, AutoConfig

    spec = _DINOV3_VARIANTS[variant]
    cfg = AutoConfig.from_pretrained(spec["hf_id"], cache_dir=cache_dir)
    model = AutoModel.from_pretrained(spec["hf_id"], cache_dir=cache_dir)

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    if unfreeze_top_k != 0:
        # HF ViT encoder: model.encoder.layer is the ModuleList of blocks.
        # Final layer norm: model.layernorm (DINOv3) or model.layernorm
        # (DINOv2-with-registers). We probe attribute names defensively.
        blocks = None
        for path in ("encoder.layer", "encoder.layers", "blocks",
                     "transformer.layers", "layer"):
            obj = model
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, nn.ModuleList):
                blocks = obj
                break
        if blocks is None:
            raise RuntimeError(f"Could not locate transformer blocks on {type(model).__name__}")

        depth = len(blocks)
        k = depth if unfreeze_top_k < 0 else min(unfreeze_top_k, depth)
        for blk in blocks[depth - k:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        # Final layer norm (if present) is part of "top".
        for ln_name in ("layernorm", "norm", "ln_post"):
            if hasattr(model, ln_name):
                ln = getattr(model, ln_name)
                if isinstance(ln, nn.Module):
                    for p in ln.parameters():
                        p.requires_grad_(True)

        model.num_unfrozen_blocks = k  # record for logging
    else:
        model.num_unfrozen_blocks = 0

    # Keep in eval mode only if fully frozen; otherwise allow dropout / training
    # normalization statistics to behave normally.
    if model.num_unfrozen_blocks == 0:
        model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm3d(num_ch: int, groups: int = 8) -> nn.Module:
    g = min(groups, num_ch)
    while num_ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_ch)


class ConvBlock3D(nn.Module):
    """Conv3d -> GN -> GELU, optionally residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
                 residual: bool = False):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.n1 = _norm3d(out_ch)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.n2 = _norm3d(out_ch)
        self.act2 = nn.GELU()
        self.residual = residual
        if residual and (in_ch != out_ch or stride != 1):
            self.skip = nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity() if residual else None

    def forward(self, x):
        h = self.act1(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        if self.residual:
            h = h + self.skip(x)
        return self.act2(h)


# ---------------------------------------------------------------------------
# Stem (produces F1 pre-ViT)
# ---------------------------------------------------------------------------

class ShallowStem3D(nn.Module):
    def __init__(self, in_ch: int = 1, ch: Tuple[int, int] = (32, 64)):
        super().__init__()
        c1, c2 = ch
        self.block1 = nn.Sequential(
            nn.Conv3d(in_ch, c1, 3, stride=2, padding=1, bias=False),  # stride 2 -> F1
            _norm3d(c1), nn.GELU(),
            ConvBlock3D(c1, c1, residual=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(c1, c2, 3, stride=2, padding=1, bias=False),     # stride 4 -> F2 seed
            _norm3d(c2), nn.GELU(),
            ConvBlock3D(c2, c2, residual=True),
        )

    def forward(self, x):
        f1 = self.block1(x)  # [B, 32, D/2, H/2, W/2]
        f2_seed = self.block2(f1)  # [B, 64, D/4, H/4, W/4]
        return f1, f2_seed


# ---------------------------------------------------------------------------
# Tri-planar DINOv3 feature extractor
# ---------------------------------------------------------------------------

class TriPlanarDINOv3(nn.Module):
    """Run the frozen 2D DINOv3 on axial/sagittal/coronal slices.
    This produces 4 feature volumes per plane x 3 planes = 12 volumes, which
    are fused by TriPlanarFusion3D below.
    """

    # Indices of ViT blocks whose hidden states we extract. ViT-S depth=12;
    # we pick 4 roughly-equidistant layers. These become F2_vit, F3_vit, F4_vit,
    # and the F5 source (deepest).
    _EXTRACT_LAYERS: Tuple[int, ...] = (2, 5, 8, 11)  # 0-indexed

    def __init__(
        self,
        variant: str = "vits16",
        patch_size: Optional[int] = None,   # defaults to variant's native patch
        max_z_slices: int = 256,
        cache_dir: Optional[str] = None,
        unfreeze_top_k: int = 0,
    ):
        super().__init__()
        if variant not in _DINOV3_VARIANTS:
            raise ValueError(f"Unknown DINOv3 variant: {variant}")
        spec = _DINOV3_VARIANTS[variant]
        self.variant = variant
        self.patch_size = int(patch_size if patch_size is not None
                              else spec.get("patch", 16))
        self.embed_dim = spec["embed_dim"]

        self.vit, self.vit_cfg = _load_dinov3(
            variant, cache_dir=cache_dir, unfreeze_top_k=unfreeze_top_k,
        )
        # Use eager attention so output_attentions=True works for supervision.
        self.vit.set_attn_implementation("eager")
        # Any trainable ViT params -> drop the no_grad wrapper on the forward.
        self.vit_trainable = any(p.requires_grad for p in self.vit.parameters())

        # Number of prefix tokens to drop: 1 CLS + 4 registers for DINOv3.
        # Verified against config; we read it dynamically to be safe.
        num_reg = int(getattr(self.vit_cfg, "num_register_tokens", 4))
        self.num_prefix = 1 + num_reg

        # Z-channel embedding: per-slice learned positional embedding added to
        # the patch tokens. This tells downstream fusion "this is slice k of
        # the sliced axis" -- essential because the 2D ViT has no idea.
        self.max_z_slices = max_z_slices
        self.z_embed = nn.Parameter(torch.zeros(1, max_z_slices, self.embed_dim))
        nn.init.trunc_normal_(self.z_embed, std=0.02)

        # Plane identity embedding (axial=0, sagittal=1, coronal=2).
        self.plane_embed = nn.Parameter(torch.zeros(3, 1, self.embed_dim))
        nn.init.trunc_normal_(self.plane_embed, std=0.02)

    def _forward_vit_slices(
        self,
        slices_2d: torch.Tensor,
        return_last2_attn: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        slices_2d: [N, 3, h, w] float image batch of single 2D slices.
        Returns: list of 4 tensors, each [N, tokens, embed_dim], for the 4
                 extraction layers. tokens excludes CLS and registers.

        If return_last2_attn=True, also returns attn_maps: [N, 2, heads, Ph*Pw]
        — the CLS-to-patch attention from the last two blocks (blocks 10 & 11).
        These are the two trainable blocks; their attention can be supervised
        to focus on thin/rare vessel classes (VesselAttentionLoss).

        Gradients flow through the unfrozen top-K blocks only; lower blocks
        still receive grads from the forward graph, but their params have
        requires_grad=False so they are not updated.
        """
        ctx = torch.enable_grad() if self.vit_trainable else torch.no_grad()
        with ctx:
            out = self.vit(
                pixel_values=slices_2d,
                output_hidden_states=True,
                output_attentions=return_last2_attn,
            )
        # hidden_states[0] is the embedding output, [1..depth] are block outputs.
        hs = out.hidden_states  # tuple, len = depth + 1
        feats = []
        for li in self._EXTRACT_LAYERS:
            h = hs[li + 1]  # +1 because index 0 is the embedding
            h = h[:, self.num_prefix:, :]  # drop CLS + registers -> [N, P*P, C]
            feats.append(h)

        if not return_last2_attn:
            return feats

        # out.attentions is tuple of [N, heads, seq, seq] for each block.
        # We want CLS row (index 0) -> patch columns (num_prefix onward).
        # Stack the last 2 blocks: [N, 2, heads, P*P]
        attn_maps = torch.stack([
            out.attentions[-2][:, :, 0, self.num_prefix:],  # block 10
            out.attentions[-1][:, :, 0, self.num_prefix:],  # block 11
        ], dim=1)  # [N, 2, heads, P*P]
        return feats, attn_maps

    def forward_with_attn_supervision(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        target_classes: Tuple[int, ...] = (8, 9, 10, 13),
    ) -> Tuple[Dict, torch.Tensor]:
        """Forward pass that also returns an attention supervision loss.

        For each 2D slice in each tri-planar view, we downsample the GT mask
        to the ViT patch grid and compute a binary cross-entropy loss that
        pushes the CLS attention of the last two ViT blocks (the trainable
        ones) to be high on patches that contain rare/thin vessel classes
        and low on empty patches.

        Args:
            x:    [B, 1, D, H, W] image volume
            mask: [B, D, H, W] long label (0=background, 1..13=class)
            target_classes: which classes to supervise attention for

        Returns:
            encoder_features: same as forward()
            attn_loss: scalar attention supervision loss
        """
        B = x.shape[0]
        feats_by_layer: Dict[str, List[torch.Tensor]] = {
            f"L{li}": [] for li in self._EXTRACT_LAYERS
        }
        attn_losses: List[torch.Tensor] = []

        # Build binary vessel mask for target classes: 1 = hard vessel present
        vessel_mask = torch.zeros_like(mask, dtype=torch.float32)
        for c in target_classes:
            vessel_mask = vessel_mask + (mask == c).float()
        vessel_mask = vessel_mask.clamp(0, 1)  # [B, D, H, W]

        ps = self.patch_size
        CHUNK = 128

        for plane_idx, axis in enumerate((2, 3, 4)):
            slabs, _B, S, in_shape, pad_shape = self._prepare_plane_slices(x, axis=axis)
            h_in, w_in = in_shape
            h_pad, w_pad = pad_shape
            Ph_valid = (h_in + ps - 1) // ps
            Pw_valid = (w_in + ps - 1) // ps

            # Prepare the corresponding 2D label slices.
            # axis 2=D, 3=H, 4=W: slice the mask along the same axis.
            if axis == 2:
                mask_slabs = vessel_mask.reshape(B * S, 1, h_in, w_in)
            elif axis == 3:
                mask_slabs = vessel_mask.permute(0, 2, 1, 3).reshape(B * S, 1, h_in, w_in)
            elif axis == 4:
                mask_slabs = vessel_mask.permute(0, 3, 1, 2).reshape(B * S, 1, h_in, w_in)

            # Downsample mask to patch grid via average pooling (soft label).
            mask_patch = F.avg_pool2d(
                mask_slabs, kernel_size=ps, stride=ps, padding=0,
            )  # [B*S, 1, Ph_full, Pw_full]
            mask_patch = mask_patch[:, 0, :Ph_valid, :Pw_valid]  # [B*S, Ph, Pw]
            mask_patch = mask_patch.reshape(B * S, Ph_valid * Pw_valid)  # [B*S, P*P]

            out_layers: List[List[torch.Tensor]] = [[] for _ in self._EXTRACT_LAYERS]
            for i in range(0, slabs.shape[0], CHUNK):
                slabs_chunk = slabs[i:i + CHUNK]
                feats_chunk, attn_chunk = self._forward_vit_slices(
                    slabs_chunk, return_last2_attn=True,
                )
                for k, fc in enumerate(feats_chunk):
                    out_layers[k].append(fc)

                # attn_chunk: [chunk, 2, heads, P*P]
                # Average over 2 blocks and all heads -> [chunk, P*P]
                attn_avg = attn_chunk.mean(dim=(1, 2))  # [chunk, P*P]
                mask_chunk = mask_patch[i:i + CHUNK]    # [chunk, P*P]

                # Skip slices with no vessel present (avoid pushing attn on empties
                # when target class isn't in the slice, only suppression would fire).
                has_vessel = mask_chunk.sum(dim=1) > 0  # [chunk]
                if has_vessel.any():
                    # Weighted BCE: vessel patches get pos_weight to compensate
                    # for extreme class imbalance (vessels << background in 2D).
                    # pos_weight = ratio of non-vessel to vessel tokens per slice.
                    n_pos = mask_chunk[has_vessel].sum(dim=1).clamp_min(1.0)
                    n_neg = (Ph_valid * Pw_valid) - n_pos
                    pw = (n_neg / n_pos).mean().clamp(1.0, 20.0)

                    loss_chunk = F.binary_cross_entropy_with_logits(
                        attn_avg[has_vessel],
                        mask_chunk[has_vessel],
                        pos_weight=torch.tensor(float(pw.detach()), device=x.device),
                        reduction="mean",
                    )
                    attn_losses.append(loss_chunk)

            layer_tokens = [torch.cat(parts, dim=0) for parts in out_layers]
            for li, tok in zip(self._EXTRACT_LAYERS, layer_tokens):
                vol = self._tokens_to_volume(
                    tok, B=B, S=S,
                    in_shape=in_shape, pad_shape=pad_shape,
                    plane_idx=plane_idx, axis=axis,
                )
                feats_by_layer[f"L{li}"].append(vol)

        attn_loss = torch.stack(attn_losses).mean() if attn_losses else x.new_zeros(())
        return feats_by_layer, attn_loss

    def _prepare_plane_slices(self, x: torch.Tensor, axis: int):
        """Extract 2D slices along `axis`, preserving isotropic physical scale.

        x: [B, 1, D, H, W]. Dataloader MUST resample to isotropic voxel spacing
        (e.g., 0.5mm^3) before passing to the encoder, so that 1 voxel = the
        same physical distance in every axis. Given isotropic voxels, a 4-voxel
        vessel has identical diameter in every plane.

        To avoid anisotropic stretching inside the encoder, we do NOT force a
        fixed square target. Instead:
          * Slice the volume along `axis` -> batch of 2D slices with their
            native in-plane shape (h_in, w_in).
          * Pad (not stretch) each slice to the next multiple of patch_size on
            each axis using reflection padding. This keeps aspect ratio and
            vessel thickness unchanged; the padded region is masked out when we
            drop the corresponding tokens in _tokens_to_volume.
          * Norm + repeat to 3 channels for ImageNet stats.

        axis: 2 (slice along D), 3 (slice along H), 4 (slice along W). The
              specific anatomical label is irrelevant; we sweep each of the
              three axes once, producing three orthogonal views whose vessels
              share the same physical thickness.

        Returns:
          slabs:       [B*S, 3, h_pad, w_pad]
          B, S:        batch, number of slices
          in_shape:    (h_in, w_in)   pre-padding slice shape
          pad_shape:   (h_pad, w_pad) post-padding shape  (multiples of 16)
        """
        B, C, D, H, W = x.shape
        assert C == 1, "encoder expects single-channel input"
        if axis == 2:
            S, h_in, w_in = D, H, W
            slabs = x.permute(0, 2, 1, 3, 4).reshape(B * S, 1, h_in, w_in)
        elif axis == 3:
            S, h_in, w_in = H, D, W
            slabs = x.permute(0, 3, 1, 2, 4).reshape(B * S, 1, h_in, w_in)
        elif axis == 4:
            S, h_in, w_in = W, D, H
            slabs = x.permute(0, 4, 1, 2, 3).reshape(B * S, 1, h_in, w_in)
        else:
            raise ValueError(axis)

        # Pad to next multiple of patch_size (aspect-preserving, unlike resize).
        ps = self.patch_size
        h_pad = ((h_in + ps - 1) // ps) * ps
        w_pad = ((w_in + ps - 1) // ps) * ps
        pad_h = h_pad - h_in
        pad_w = w_pad - w_in
        if pad_h or pad_w:
            # F.pad order is (w_left, w_right, h_left, h_right) for last 2 dims.
            slabs = F.pad(slabs, (0, pad_w, 0, pad_h), mode="reflect")

        # Per-slice min-max norm -> [0,1]; then ImageNet stats on 3 channels.
        slabs = slabs - slabs.amin(dim=(2, 3), keepdim=True)
        slabs = slabs / (slabs.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
        slabs = slabs.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=slabs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=slabs.device).view(1, 3, 1, 1)
        slabs = (slabs - mean) / std

        return slabs, B, S, (h_in, w_in), (h_pad, w_pad)

    def _tokens_to_volume(
        self,
        tokens: torch.Tensor,       # [B*S, Ph*Pw, C]
        B: int, S: int,
        in_shape: Tuple[int, int],
        pad_shape: Tuple[int, int],
        plane_idx: int,
        axis: int,
        abs_slice_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reshape [B*S, Ph*Pw, C] -> [B, C, <sliced axis=S>, <in-plane grids>]
        with sliced axis placed back in position. Also crops the padded tokens
        so the 3D grid covers only the true volume extent.

        abs_slice_indices: [S] float tensor in [0,1]. When provided (full-volume
            mode), z_embed is computed by interpolating at these absolute brain
            coordinates rather than using patch-relative indices 0..S-1. This
            encodes WHERE in the brain each slice is, not just its position in
            the local patch. Essential for topology learning across the full CoW.
        """
        h_in, w_in = in_shape
        h_pad, w_pad = pad_shape
        ps = self.patch_size
        Ph_full, Pw_full = h_pad // ps, w_pad // ps
        Ph_valid = (h_in + ps - 1) // ps
        Pw_valid = (w_in + ps - 1) // ps
        C = tokens.shape[-1]

        if abs_slice_indices is not None:
            # Absolute brain-coordinate positional embedding.
            # Map normalized [0,1] slice positions to discrete z_embed indices.
            # This encodes WHERE in the brain each slice is (e.g. 0.5 = mid-brain
            # at the CoW level), enabling the model to learn depth-aware topology.
            indices = (abs_slice_indices * (self.max_z_slices - 1)).long().clamp(
                0, self.max_z_slices - 1
            )
            z_full = self.z_embed[:, indices, :]  # [1, S, C]
        elif S > self.max_z_slices:
            z_full = F.interpolate(
                self.z_embed.transpose(1, 2), size=S, mode="linear", align_corners=True
            ).transpose(1, 2)  # [1, S, C]
        else:
            z_full = self.z_embed[:, :S, :]

        z_full = z_full + self.plane_embed[plane_idx:plane_idx + 1]  # [1, S, C]

        # Reshape tokens to the padded 2D grid, crop the padded tokens, add z emb.
        tokens = tokens.view(B * S, Ph_full, Pw_full, C)
        tokens = tokens[:, :Ph_valid, :Pw_valid, :].contiguous()  # drop padded tokens
        tokens = tokens.view(B, S, Ph_valid * Pw_valid, C)
        tokens = tokens + z_full.unsqueeze(2)  # broadcast over tokens

        # [B, S, Ph, Pw, C] -> [B, C, S, Ph, Pw]
        vol = tokens.view(B, S, Ph_valid, Pw_valid, C).permute(0, 4, 1, 2, 3).contiguous()

        # Place S back to its original axis.
        if axis == 2:
            out = vol                                        # (B, C, D=S, H=Ph, W=Pw)
        elif axis == 3:
            out = vol.permute(0, 1, 3, 2, 4).contiguous()     # (B, C, D=Ph, H=S, W=Pw)
        elif axis == 4:
            out = vol.permute(0, 1, 3, 4, 2).contiguous()     # (B, C, D=Ph, H=Pw, W=S)
        else:
            raise ValueError(axis)
        return out

    def forward(
        self,
        x: torch.Tensor,
        full_vol: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        x: [B, 1, D, H, W]. The dataloader MUST resample to isotropic voxel
        spacing before calling this. No fixed target_side is imposed -- each
        plane's in-plane dims are padded (not stretched) to a multiple of 16.

        full_vol: [B, 1, D_full, H_full, W_full] optional. When provided,
            DINOv3 slices this FULL brain volume instead of the patch `x`.
            This gives the ViT's self-attention global context across the
            entire CoW — ICA, Pcom, and PCA are all visible in the same
            axial slice, allowing the model to learn topology (Pcom connects
            ICA to PCA) rather than just local appearance.
            The z_embed is then computed as absolute brain coordinates
            (slice_idx / total_slices), not patch-relative indices.
            The resulting feature volumes are resized to match `x`'s spatial
            dims before returning, so _build_pyramid is unchanged.

        Returns dict with keys 'L2','L5','L8','L11' mapping to a list over
        planes of feature volumes at the spatial resolution of `x`.
        """
        ctx_vol = full_vol if full_vol is not None else x
        B = ctx_vol.shape[0]

        feats_by_layer: Dict[str, List[torch.Tensor]] = {
            f"L{li}": [] for li in self._EXTRACT_LAYERS
        }

        # Full-vol path runs under no_grad: the full-brain ViT pass provides
        # global topology context from the frozen lower blocks. No gradients
        # needed — they would OOM (O(tokens^2) attention × 12 layers × backward
        # graph ≈ 30 GB). Top-4 unfrozen blocks receive gradients via the
        # separate patch pass in GraphCoWEncoder.forward().
        ctx_manager = torch.no_grad() if full_vol is not None else contextlib.nullcontext()

        with ctx_manager:
            for plane_idx, axis in enumerate((2, 3, 4)):
                slabs, _B, S, in_shape, pad_shape = self._prepare_plane_slices(
                    ctx_vol, axis=axis
                )
                abs_slice_indices: Optional[torch.Tensor] = None
                if full_vol is not None:
                    abs_slice_indices = torch.linspace(0, 1, S, device=x.device)

                # Full-brain slices: ~396 tokens/slice. Chunk=8 keeps ~1 GB peak.
                CHUNK = 8 if full_vol is not None else 128
                out_layers: List[List[torch.Tensor]] = [[] for _ in self._EXTRACT_LAYERS]
                for i in range(0, slabs.shape[0], CHUNK):
                    feats_chunk = self._forward_vit_slices(slabs[i:i + CHUNK])
                    for k, fc in enumerate(feats_chunk):
                        out_layers[k].append(fc)
                layer_tokens = [torch.cat(parts, dim=0) for parts in out_layers]

                for li, tok in zip(self._EXTRACT_LAYERS, layer_tokens):
                    vol = self._tokens_to_volume(
                        tok, B=B, S=S,
                        in_shape=in_shape, pad_shape=pad_shape,
                        plane_idx=plane_idx, axis=axis,
                        abs_slice_indices=abs_slice_indices,
                    )
                    feats_by_layer[f"L{li}"].append(vol)

        return feats_by_layer

class TriPlanarFusion3D(nn.Module):
    """Orientation-gated fusion of the 3 per-plane ViT feature volumes.

    Why the simple "upsample-all-then-sum" approach is insufficient:

      (1) A horizontal vessel's cross-section is sharp in the axial plane and
          smeared in sagittal/coronal. A vertical vessel is the opposite. A
          fixed per-channel projection weights the planes globally and cannot
          shift trust per-location as the vessel orientation changes.

      (2) Trilinear upsampling of a plane's coarse in-plane axes (patch-token
          grid 12x12 -> 192x192) invents values that were never encoded at
          that resolution. For small vessels (Pcom/Acom at 3 voxels diameter)
          this blur dominates the signal.

    Fix: per-voxel softmax gate over the three planes, combined with
    axis-aware upsampling so that each plane contributes most along the axis
    at which it is natively high-resolution.

      * Each plane is upsampled with trilinear interpolation to the target
        grid. That alone is unchanged. What we change is how the three
        expanded tensors are combined.
      * A 3D conv on the concatenated projected features predicts 3 logits
        per voxel. After softmax, those become per-voxel plane weights that
        sum to 1.
      * We add a learned axis-aware bias to the pre-softmax logits: plane p
        gets a +prior added for voxels whose local orientation aligns with
        plane p's hi-res axis. This gives the gate a head-start so it does
        not need many epochs to learn "trust axial along D."

    Compared with the naive sum:
      * planes are no longer averaged equally; the gate routes each voxel to
        its most informative plane;
      * information from the genuine hi-res axis of one plane is kept even
        when the other two planes only have coarse-grid interpolations there;
      * small directional vessels get routed to the correct plane instead of
        being averaged into noise.
    """

    # Per-plane hi-res axis (D, H, or W) in (B, C, D, H, W) layout:
    #   plane 0 (axial, slicing D)    -> hi-res on D  -> target axis index 2
    #   plane 1 (sagittal, slicing H) -> hi-res on H  -> target axis index 3
    #   plane 2 (coronal, slicing W)  -> hi-res on W  -> target axis index 4
    _HI_RES_AXIS: Tuple[int, int, int] = (2, 3, 4)

    def __init__(self, embed_dim: int, out_ch: int, gate_hidden: int = 32):
        super().__init__()
        # Per-plane 1x1x1 projection of raw ViT features to the fusion width.
        self.proj = nn.ModuleList([
            nn.Conv3d(embed_dim, out_ch, 1, bias=False) for _ in range(3)
        ])

        # Per-plane-weight gate: takes concat of 3 projected features -> 3 logits.
        # Small bottleneck keeps the gate cheap.
        self.gate = nn.Sequential(
            nn.Conv3d(3 * out_ch, gate_hidden, 1, bias=False),
            _norm3d(gate_hidden),
            nn.GELU(),
            nn.Conv3d(gate_hidden, 3, 1, bias=True),
        )

        # Axis-aware bias: a learnable scalar per plane added to that plane's
        # gate logit everywhere the voxel's local orientation aligns with the
        # plane's hi-res axis. Initialized to a small positive value so the
        # gate starts with a mild preference for each plane along its hi-res
        # axis; the net can push it negative if needed.
        self.axis_prior = nn.Parameter(torch.full((3,), 0.5))

        # Output refinement after gated sum.
        self.post_norm = _norm3d(out_ch)
        self.post_act = nn.GELU()
        self.refine = ConvBlock3D(out_ch, out_ch, residual=True)

    @staticmethod
    def _axis_alignment_score(
        feat: torch.Tensor, axis: int, eps: float = 1e-6
    ) -> torch.Tensor:
        """Cheap per-voxel indicator of how much local feature variation is
        aligned with `axis`. Large absolute gradient along `axis` relative to
        the other axes -> axis is informative here, so the plane hi-res on
        that axis should be trusted.

        feat: [B, C, D, H, W] projected feature volume (any one plane).
        Returns: [B, 1, D, H, W] in (0, 1).
        """
        # L2 over channels of finite differences along each axis, then compare
        # the target-axis magnitude against the sum over all axes.
        # Differences are computed with padding so shape is preserved.
        def _diff(t, ax):
            # forward diff with reflect padding at the far end
            slicer_a = [slice(None)] * 5
            slicer_b = [slice(None)] * 5
            slicer_a[ax] = slice(1, None)
            slicer_b[ax] = slice(None, -1)
            d = t[tuple(slicer_a)] - t[tuple(slicer_b)]
            pad = [0, 0, 0, 0, 0, 0]
            pad_idx = (4 - ax) * 2  # F.pad order is reversed: (W_l,W_r,H_l,H_r,D_l,D_r)
            pad[pad_idx + 1] = 1
            d = F.pad(d, pad, mode="replicate")
            return d

        gD = _diff(feat, 2).pow(2).sum(dim=1, keepdim=True)
        gH = _diff(feat, 3).pow(2).sum(dim=1, keepdim=True)
        gW = _diff(feat, 4).pow(2).sum(dim=1, keepdim=True)
        denom = gD + gH + gW + eps
        if axis == 2:   num = gD
        elif axis == 3: num = gH
        elif axis == 4: num = gW
        else: raise ValueError(axis)
        return (num / denom).clamp_(0.0, 1.0)

    def forward(self, per_plane: List[torch.Tensor], target_shape: Tuple[int, int, int]):
        """
        per_plane: 3 tensors [B, C_embed, d_p, h_p, w_p] (axial, sag, coronal
                   respectively). Each has exactly one hi-res axis matching
                   the volume extent; the other two are coarse patch-token grids.
        target_shape: (D_t, H_t, W_t).

        Returns fused 3D features of shape [B, out_ch, D_t, H_t, W_t].
        """
        # 1. Upsample each plane to the target grid and project to out_ch.
        #    Trilinear on coarse axes is honest interpolation of real token
        #    features; along the plane's hi-res axis it reduces to linear (or
        #    identity when sizes already match), so no information is invented.
        projected: List[torch.Tensor] = []
        for i, v in enumerate(per_plane):
            v_up = F.interpolate(v, size=target_shape, mode="trilinear",
                                 align_corners=False)
            projected.append(self.proj[i](v_up))

        # 2. Orientation-aware gating.
        #    (a) Base logits from concat of all projections.
        gate_in = torch.cat(projected, dim=1)          # [B, 3*C, D, H, W]
        logits = self.gate(gate_in)                    # [B, 3,   D, H, W]

        #    (b) Axis-aware prior added per plane. We reuse the projected
        #        feature of plane p to estimate how much the local signal
        #        varies along p's hi-res axis (high variation -> high prior).
        priors = []
        for i in range(3):
            score = self._axis_alignment_score(
                projected[i], axis=self._HI_RES_AXIS[i]
            )                                          # [B, 1, D, H, W]
            priors.append(score * self.axis_prior[i])
        logits = logits + torch.cat(priors, dim=1)      # broadcast over C=3

        #    (c) Per-voxel softmax over planes.
        weights = F.softmax(logits, dim=1)             # [B, 3, D, H, W]

        # 3. Gated sum. (weights[:, i:i+1] is [B,1,D,H,W] -- broadcasts over C)
        fused = sum(weights[:, i:i+1] * projected[i] for i in range(3))

        # 4. Post-fusion refinement.
        fused = self.post_act(self.post_norm(fused))
        return self.refine(fused)


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class GraphCoWEncoder(nn.Module):
    """Encoder producing the 5-level feature pyramid F1..F5.

    F1 (stride 2)  : from shallow 3D stem -- preserves fine detail
    F2 (stride 4)  : stem feature fused with ViT layer-2 tri-planar
    F3 (stride 8)  : ViT layer-5 tri-planar
    F4 (stride 16) : ViT layer-8 tri-planar (matches ViT native stride)
    F5 (stride 32) : ViT layer-11 tri-planar (deepest, pooled)

    Output channels per level are configurable. Defaults chosen so that a
    decoder with matching channels fits in ~15 GB at 192^3 bf16.
    """

    def __init__(
        self,
        in_ch: int = 1,
        variant: str = "vits16",
        stem_channels: Tuple[int, int] = (32, 64),
        pyramid_channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 384),
        cache_dir: Optional[str] = None,
        unfreeze_top_k: int = 0,
    ):
        super().__init__()
        assert pyramid_channels[0] == stem_channels[0], (
            "F1 must match stem first-stage channels"
        )
        self.pyramid_channels = pyramid_channels
        self.unfreeze_top_k = unfreeze_top_k

        self.stem = ShallowStem3D(in_ch=in_ch, ch=stem_channels)

        self.triplanar = TriPlanarDINOv3(
            variant=variant, patch_size=None, cache_dir=cache_dir,
            unfreeze_top_k=unfreeze_top_k,
        )
        ed = self.triplanar.embed_dim

        # Fusions for F2..F5 (one per ViT layer we extract).
        self.fuse_L2 = TriPlanarFusion3D(ed, pyramid_channels[1])   # -> F2
        self.fuse_L5 = TriPlanarFusion3D(ed, pyramid_channels[2])   # -> F3
        self.fuse_L8 = TriPlanarFusion3D(ed, pyramid_channels[3])   # -> F4
        self.fuse_L11 = TriPlanarFusion3D(ed, pyramid_channels[4])  # -> F5

        # F2 = stem's F2_seed + fused ViT-L2 at stride 4. The stem provides
        # hi-res detail, ViT provides context; both matter for small vessels.
        self.f2_merge = nn.Sequential(
            nn.Conv3d(stem_channels[1] + pyramid_channels[1], pyramid_channels[1], 1, bias=False),
            _norm3d(pyramid_channels[1]), nn.GELU(),
            ConvBlock3D(pyramid_channels[1], pyramid_channels[1], residual=True),
        )

        # Lateral smoothing conv on F3, F4, F5.
        self.post_f3 = ConvBlock3D(pyramid_channels[2], pyramid_channels[2], residual=True)
        self.post_f4 = ConvBlock3D(pyramid_channels[3], pyramid_channels[3], residual=True)
        self.post_f5 = ConvBlock3D(pyramid_channels[4], pyramid_channels[4], residual=True)

    @staticmethod
    def _target_shape(d: int, h: int, w: int, stride: int) -> Tuple[int, int, int]:
        return (d // stride, h // stride, w // stride)

    def _build_pyramid(
        self,
        per_layer: Dict[str, List[torch.Tensor]],
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build the F1..F5 pyramid from tri-planar layer tokens + stem.

        Separated from forward() so that GraphCoWNet can call it after
        forward_with_attn_supervision() returns the per_layer dict directly.
        """
        B, _, D, H, W = x.shape
        f1, f2_seed = self.stem(x)

        shp2 = self._target_shape(D, H, W, 4)
        shp3 = self._target_shape(D, H, W, 8)
        shp4 = self._target_shape(D, H, W, 16)
        shp5 = self._target_shape(D, H, W, 32)

        f2_vit = self.fuse_L2(per_layer["L2"], shp2)
        f3 = self.post_f3(self.fuse_L5(per_layer["L5"], shp3))
        f4 = self.post_f4(self.fuse_L8(per_layer["L8"], shp4))
        f5 = self.post_f5(self.fuse_L11(per_layer["L11"], shp5))
        f2 = self.f2_merge(torch.cat([f2_seed, f2_vit], dim=1))
        return {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5}

    def forward(
        self,
        x: torch.Tensor,
        full_vol: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """x: [B, 1, D, H, W] patch. Requires D,H,W divisible by 32.

        full_vol: [B, 1, D_full, H_full, W_full] optional full brain volume.
            When provided, DINOv3 processes full-brain slices (global context)
            instead of patch slices. The stem still processes the patch for
            fine-grained local detail. This is the key architectural change
            for global topology learning.
        """
        B, _, D, H, W = x.shape
        for name, s in (("D", D), ("H", H), ("W", W)):
            if s % 32 != 0:
                raise ValueError(f"input {name}={s} must be divisible by 32")

        if full_vol is not None:
            # Two-pass strategy:
            # Pass 1 (no_grad): full brain → global topology context features
            # Pass 2 (with_grad): patch → fine-detail features + gradient flow
            #   through the top-4 unfrozen ViT blocks
            # Fuse by adding: global context enriches the patch features.
            global_feats = self.triplanar(x, full_vol=full_vol)
            patch_feats = self.triplanar(x, full_vol=None)
            # Fuse: resize global to match patch spatial dims, then add.
            per_layer = {}
            for k in patch_feats:
                fused = []
                for g, p in zip(global_feats[k], patch_feats[k]):
                    if g.shape != p.shape:
                        g = F.interpolate(g, size=p.shape[-3:],
                                          mode="trilinear", align_corners=False)
                    fused.append(g + p)
                per_layer[k] = fused
        else:
            per_layer = self.triplanar(x)
        return self._build_pyramid(per_layer, x)

    # Convenience: parameter groups for optimizer.
    def trainable_parameter_groups(
        self,
        lr_base: float = 1e-4,
        lr_vit: float = 1e-5,
    ):
        """Return param groups. The ViT unfrozen blocks (if any) get a much
        lower LR than the adapter -- standard recipe for fine-tuning pretrained
        transformers so they don't drift away from their pretrained optimum.
        """
        vit_ids = {id(p) for p in self.triplanar.vit.parameters() if p.requires_grad}
        vit_params, other_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (vit_params if id(p) in vit_ids else other_params).append(p)
        groups = [{"params": other_params, "lr": lr_base}]
        if vit_params:
            groups.append({"params": vit_params, "lr": lr_vit})
        return groups


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="vits16", choices=list(_DINOV3_VARIANTS.keys()))
    p.add_argument("--patch", type=int, nargs=3, default=[192, 192, 192],
                   metavar=("D", "H", "W"),
                   help="patch size; anisotropic patches are supported as long "
                        "as the voxel spacing is isotropic (e.g. 0.5mm^3).")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    enc = GraphCoWEncoder(variant=args.variant).to(device)
    if dtype == torch.bfloat16:
        enc = enc.to(dtype)  # ViT frozen, bf16 is safe

    D, H, W = args.patch
    x = torch.randn(1, 1, D, H, W, device=device, dtype=dtype)
    with torch.cuda.amp.autocast(enabled=dtype == torch.bfloat16, dtype=torch.bfloat16):
        out = enc(x)

    for k, v in out.items():
        print(f"{k}: {tuple(v.shape)}  dtype={v.dtype}")

    n_all = sum(p.numel() for p in enc.parameters())
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"Total params: {n_all/1e6:.2f}M, trainable: {n_train/1e6:.2f}M")
    if device == "cuda":
        print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
