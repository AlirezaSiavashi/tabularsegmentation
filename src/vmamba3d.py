"""
vmamba3d.py — Pretrained VMamba-Tiny Navigator with 2D→3D Weight Inflation
==========================================================================

Replaces the from-scratch GlobalNavigator with a pretrained VMamba-Tiny backbone
whose 2D weights are inflated to 3D (I3D/Video Swin technique) plus 3D DWConv
for locality preservation.

Architecture:
  Input [B,1,D/4,H/4,W/4]  (after nav_down=4 in NavBrushModel)
    → PatchEmbed3D(1, 96, k=4, s=4)  → [B,96,D/16,H/16,W/16]   (inflated pretrained)
    → 2× VSSBlock3D(96)              → [B,96,D/16,H/16,W/16]   (inflated pretrained)
    → Proj Conv3d(96, tok_dim, 1)    → [B,tok_dim,D/16,H/16,W/16] (random init)
    → 2× BiMambaSSM3D(tok_dim)       → [B,tok_dim,D/16,H/16,W/16] (random init)
  Output: tokens at stride=16, tok_dim=128

Weight inflation (2D → 3D):
  - Conv2d: replicate kernel along z-axis, divide by kernel_depth
  - DWConv2d: same as Conv2d (groups doesn't affect kernel shape)
  - RGB→grayscale: sum over 3 input channels
  - Linear/LayerNorm/bias: copy directly
  - SSM params (A, B, C, D, dt): NOT transferred (random init)

References:
  - Swin-UMamba (MICCAI 2024): ImageNet-pretrained VMamba for medical segmentation
  - I3D (Carreira & Zisserman, 2017): 2D→3D weight inflation for video
  - UlikeMamba_3d (2025): 3D DWConv gives +1.9 Dice vs 1D DWConv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: axis permutation (shared with train.py's BiMambaSSM)
# ---------------------------------------------------------------------------
def _permute_for_axis(x: torch.Tensor, order: str) -> torch.Tensor:
    """Permute spatial dims of [B,C,D,H,W] according to axis order string."""
    axes = {"d": 2, "h": 3, "w": 4}
    perm = [0, 1, axes[order[0]], axes[order[1]], axes[order[2]]]
    return x.permute(*perm).contiguous()


# ---------------------------------------------------------------------------
# VSSBlock3D: VMamba-inspired block for 3D volumes
# ---------------------------------------------------------------------------
class VSSBlock3D(nn.Module):
    """
    VMamba Visual State Space block adapted for 3D.

    Structure mirrors VMamba's VSSBlock:
      SSM branch: LN → in_proj → split(x,z) → DWConv3D → SiLU
                  → multi-axis BiMamba scan → LN → gate(z) → out_proj → + residual
      MLP branch: LN → fc1 → GELU → fc2 → + residual

    Pretrained layers (from VMamba-Tiny Stage 1, inflated 2D→3D):
      in_proj, dwconv3d, out_norm, out_proj, norm, norm2, fc1, fc2

    Randomly initialized:
      ssm_fwd, ssm_bwd (Mamba SSM instances)
    """

    def __init__(
        self,
        dim: int,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        axes: List[str] = None,
        mamba_dropout: float = 0.05,
    ):
        super().__init__()
        if axes is None:
            axes = ["dhw", "hwd", "wdh"]

        d_inner = int(dim * ssm_ratio)
        mlp_hidden = int(dim * mlp_ratio)

        # === SSM branch ===
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.dwconv3d = nn.Conv3d(
            d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True
        )
        self.act = nn.SiLU()

        # Bidirectional Mamba SSM (expand=1 since we already expanded via in_proj)
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise RuntimeError("mamba_ssm not installed") from e

        # d_conv=4 (standard default; d_conv=1 unsupported by causal_conv1d CUDA kernel)
        # expand=1 since we already expanded via in_proj
        self.ssm_fwd = Mamba(d_model=d_inner, d_state=32, d_conv=4, expand=1)
        self.ssm_bwd = Mamba(d_model=d_inner, d_state=32, d_conv=4, expand=1)
        self.axes = axes
        self.ssm_drop = nn.Dropout(mamba_dropout)

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

        # === MLP branch ===
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.mlp_act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, dim)

    def _scan_axes(self, x_ssm: torch.Tensor) -> torch.Tensor:
        """Multi-axis bidirectional Mamba scan on 3D token grid."""
        B = x_ssm.shape[0]
        C = x_ssm.shape[1]
        outs = []
        for ax in self.axes:
            xp = _permute_for_axis(x_ssm, ax)
            _, _, Dp, Hp, Wp = xp.shape
            seq = xp.permute(0, 2, 3, 4, 1).reshape(B, Dp * Hp * Wp, C)

            y_f = self.ssm_fwd(seq)
            y_b = torch.flip(self.ssm_bwd(torch.flip(seq, dims=(1,))), dims=(1,))
            y = y_f + y_b

            y = y.reshape(B, Dp, Hp, Wp, C).permute(0, 4, 1, 2, 3).contiguous()
            # Unpermute back to original axis order
            cur = {
                "d": 2 + ax.index("d"),
                "h": 2 + ax.index("h"),
                "w": 2 + ax.index("w"),
            }
            y = y.permute(0, 1, cur["d"], cur["h"], cur["w"]).contiguous()
            outs.append(y)
        return torch.stack(outs, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → [B, C, D, H, W]"""
        B, C, D, H, W = x.shape

        # --- SSM branch ---
        x_cl = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        h = self.norm(x_cl)
        h = self.in_proj(h)  # [B, D, H, W, 2*d_inner]
        x_ssm, z = h.chunk(2, dim=-1)  # each [B, D, H, W, d_inner]

        # 3D DWConv for locality (the key fix over standard Mamba)
        x_ssm = x_ssm.permute(0, 4, 1, 2, 3).contiguous()  # [B, d_inner, D, H, W]
        x_ssm = self.act(self.dwconv3d(x_ssm))

        # Multi-axis bidirectional SSM scan
        y_ssm = self._scan_axes(x_ssm)  # [B, d_inner, D, H, W]
        y_ssm = self.ssm_drop(y_ssm)

        # Gate and project
        y_ssm = y_ssm.permute(0, 2, 3, 4, 1)  # [B, D, H, W, d_inner]
        y_ssm = self.out_norm(y_ssm)
        z = self.act(z)
        out = y_ssm * z
        out = self.out_proj(out)  # [B, D, H, W, C]
        out = out.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]

        x = x + out

        # --- MLP branch ---
        x_cl = x.permute(0, 2, 3, 4, 1)
        h = self.norm2(x_cl)
        h = self.fc2(self.mlp_act(self.fc1(h)))
        x = x + h.permute(0, 4, 1, 2, 3).contiguous()

        return x


# ---------------------------------------------------------------------------
# BiMambaSSM3D: Enhanced BiMambaSSM with 3D DWConv for locality
# ---------------------------------------------------------------------------
class BiMambaSSM3D(nn.Module):
    """
    BiMambaSSM with added 3D depthwise convolution for locality preservation.

    Same as BiMambaSSM in train.py but with a 3D DWConv applied to the token
    grid BEFORE flattening to 1D for the Mamba scan. This preserves 3D spatial
    adjacency that the 1D flattening destroys.
    """

    def __init__(self, dim: int, axes: List[str], dropout: float):
        super().__init__()
        if axes is None:
            axes = ["dhw", "hwd", "wdh"]

        self.norm = nn.LayerNorm(dim)
        self.dwconv3d = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.dwconv_act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.axes = axes

        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise RuntimeError("mamba_ssm not installed") from e

        self.mf = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)
        self.mb = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)

    def _run_seq(self, seq: torch.Tensor) -> torch.Tensor:
        y = self.norm(seq)
        y_f = self.mf(y)
        y_r = torch.flip(self.mb(torch.flip(y, dims=(1,))), dims=(1,))
        y = y_f + y_r
        y = self.dropout(y)
        return seq + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → [B, C, D, H, W]"""
        B, C, D, H, W = x.shape

        # 3D DWConv for spatial locality before flattening
        x = x + self.dwconv_act(self.dwconv3d(x))

        # Multi-axis bidirectional Mamba scan
        outs = []
        for ax in self.axes:
            xp = _permute_for_axis(x, ax)
            _, _, Dp, Hp, Wp = xp.shape
            seq = xp.permute(0, 2, 3, 4, 1).reshape(B, Dp * Hp * Wp, C)
            seq = self._run_seq(seq)
            y = seq.reshape(B, Dp, Hp, Wp, C).permute(0, 4, 1, 2, 3).contiguous()
            cur = {
                "d": 2 + ax.index("d"),
                "h": 2 + ax.index("h"),
                "w": 2 + ax.index("w"),
            }
            y = y.permute(0, 1, cur["d"], cur["h"], cur["w"]).contiguous()
            outs.append(y)
        return torch.stack(outs, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# PretrainedVMamba3DNavigator: drop-in replacement for GlobalNavigator
# ---------------------------------------------------------------------------
class PretrainedVMamba3DNavigator(nn.Module):
    """
    Navigator using VMamba-Tiny pretrained backbone with 2D→3D inflation.

    Drop-in replacement for GlobalNavigator — same interface:
      Input:  low_img [B, 1, D/nav_down, H/nav_down, W/nav_down]
      Output: tokens  [B, tok_dim, Dt, Ht, Wt]

    Architecture:
      PatchEmbed3D(1→96, stride=4)    ← pretrained, inflated
      2× VSSBlock3D(96)               ← pretrained conv/linear/MLP, random SSM
      Proj Conv3d(96→tok_dim)          ← random init
      2× BiMambaSSM3D(tok_dim)        ← random init with 3D DWConv

    Total stride relative to low_img: 4 (from PatchEmbed)
    Total stride relative to full image: nav_down(4) × PatchEmbed(4) = 16
    """

    def __init__(
        self,
        in_ch: int,
        tok_dim: int,
        vmamba_dim: int = 96,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        num_vss_blocks: int = 2,
        num_mamba_layers: int = 2,
        mamba_axes: List[str] = None,
        gn_groups: int = 16,
        mamba_dropout: float = 0.05,
    ):
        super().__init__()
        if mamba_axes is None:
            mamba_axes = ["dhw", "hwd", "wdh"]

        self.vmamba_dim = vmamba_dim

        # Patch embedding (inflated from 2D Conv2d(3,96,4,4))
        self.patch_embed = nn.Conv3d(
            in_ch, vmamba_dim, kernel_size=4, stride=4, bias=True
        )
        self.patch_norm = nn.LayerNorm(vmamba_dim)

        # VSS blocks (from VMamba-Tiny Stage 1)
        self.vss_blocks = nn.ModuleList(
            [
                VSSBlock3D(
                    dim=vmamba_dim,
                    ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio,
                    axes=mamba_axes,
                    mamba_dropout=mamba_dropout,
                )
                for _ in range(num_vss_blocks)
            ]
        )

        # Project from vmamba_dim to tok_dim
        self.proj = nn.Sequential(
            nn.Conv3d(vmamba_dim, tok_dim, 1, bias=False),
            nn.GroupNorm(min(gn_groups, tok_dim), tok_dim),
            nn.SiLU(),
        )

        # Additional Mamba layers for global context (with 3D DWConv)
        self.mamba = nn.ModuleList(
            [
                BiMambaSSM3D(tok_dim, axes=mamba_axes, dropout=mamba_dropout)
                for _ in range(num_mamba_layers)
            ]
        )

    def forward(self, low_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low_img: [B, in_ch, D', H', W'] where D'=D/nav_down etc.
        Returns:
            tokens: [B, tok_dim, Dt, Ht, Wt] where Dt=D'/4 etc.
        """
        # Patch embedding
        with torch.backends.cudnn.flags(enabled=False):
            x = self.patch_embed(low_img)  # [B, 96, D'/4, H'/4, W'/4]

        B, C, D, H, W = x.shape

        # Apply patch norm (channel-last for LayerNorm)
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        x = self.patch_norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]

        # VSS blocks
        for blk in self.vss_blocks:
            x = blk(x)

        # Project to tok_dim
        x = self.proj(x)

        # Global sequence modeling with 3D-locality-aware Mamba
        for blk in self.mamba:
            x = blk(x)

        return x


# ---------------------------------------------------------------------------
# Weight inflation: 2D VMamba-Tiny → 3D PretrainedVMamba3DNavigator
# ---------------------------------------------------------------------------
def inflate_conv2d_weight(w_2d: torch.Tensor, kernel_depth: int) -> torch.Tensor:
    """
    Inflate 2D conv kernel to 3D by replicating along z-axis.

    From I3D (Carreira & Zisserman, 2017):
    w_3d[:,:,i,:,:] = w_2d / kernel_depth  for all i

    This preserves output magnitude: if a 2D feature f is constant along z,
    then Conv3d(f) = Conv2d(f) (same output value).

    Args:
        w_2d: [C_out, C_in, kH, kW] or [C_out, 1, kH, kW] for DWConv
        kernel_depth: depth of 3D kernel (typically same as kH)
    Returns:
        w_3d: [C_out, C_in, kernel_depth, kH, kW]
    """
    return w_2d.unsqueeze(2).repeat(1, 1, kernel_depth, 1, 1) / kernel_depth


def inflate_vmamba_tiny_to_3d(
    nav: PretrainedVMamba3DNavigator,
    ckpt_path: str,
    num_vss_blocks: int = 2,
) -> Dict[str, str]:
    """
    Load VMamba-Tiny 2D pretrained weights and inflate them into our 3D navigator.

    Weight mapping (actual VMamba-Tiny key names):
      VMamba-Tiny (2D)                                        → Our 3D Navigator
      ────────────────────────────────────────────────────────────────────────────
      patch_embed.proj.weight [96,3,4,4]                      → patch_embed.weight [96,1,4,4,4] (RGB→gray + inflate)
      patch_embed.proj.bias [96]                              → patch_embed.bias
      patch_embed.norm.weight [96]                            → patch_norm.weight (LayerNorm)
      patch_embed.norm.bias [96]                              → patch_norm.bias
      layers.0.blocks.{j}.ln_1.weight                        → vss_blocks.{j}.norm.weight
      layers.0.blocks.{j}.ln_1.bias                          → vss_blocks.{j}.norm.bias
      layers.0.blocks.{j}.self_attention.in_proj.weight       → vss_blocks.{j}.in_proj.weight
      layers.0.blocks.{j}.self_attention.conv2d.weight        → vss_blocks.{j}.dwconv3d.weight (inflate)
      layers.0.blocks.{j}.self_attention.conv2d.bias          → vss_blocks.{j}.dwconv3d.bias
      layers.0.blocks.{j}.self_attention.out_norm.weight      → vss_blocks.{j}.out_norm.weight
      layers.0.blocks.{j}.self_attention.out_norm.bias        → vss_blocks.{j}.out_norm.bias
      layers.0.blocks.{j}.self_attention.out_proj.weight      → vss_blocks.{j}.out_proj.weight

    NOT transferred (randomly initialized):
      - SSM params in VSSBlock3D (ssm_fwd.*, ssm_bwd.*)
      - MLP branch in VSSBlock3D (norm2, fc1, fc2) — VMamba-Tiny has NO MLP
      - Projection layer (proj.*)
      - BiMambaSSM3D layers (mamba.*)

    Args:
        nav: PretrainedVMamba3DNavigator instance
        ckpt_path: path to vmamba_tiny_e292.pth
        num_vss_blocks: how many VSS blocks to load (default 2 = VMamba Stage 1)

    Returns:
        Dict with 'loaded', 'skipped', 'missing' key lists for logging
    """
    logger.info(f"Loading VMamba-Tiny pretrained weights from {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # VMamba checkpoints may wrap state_dict under 'model' key
    if "model" in raw:
        sd_2d = raw["model"]
    elif "state_dict" in raw:
        sd_2d = raw["state_dict"]
    else:
        sd_2d = raw

    loaded = []
    skipped = []

    # --- Patch Embedding ---
    # Conv2d(3, 96, 4, 4) → Conv3d(1, 96, 4, 4, 4)
    pe_key = "patch_embed.proj.weight"
    if pe_key not in sd_2d:
        pe_key = "patch_embed.0.weight"  # alternative naming

    if pe_key in sd_2d:
        w_2d = sd_2d[pe_key]  # [96, 3, 4, 4]
        # Sum RGB channels → grayscale
        w_gray = w_2d.sum(dim=1, keepdim=True)  # [96, 1, 4, 4]
        # Inflate to 3D
        w_3d = inflate_conv2d_weight(w_gray, kernel_depth=4)  # [96, 1, 4, 4, 4]
        nav.patch_embed.weight.data.copy_(w_3d)
        loaded.append(f"patch_embed.weight ← {pe_key} (RGB→gray + inflate)")
    else:
        skipped.append(f"patch_embed.weight: key '{pe_key}' not found")

    # Patch embed bias
    pe_bias_key = pe_key.replace("weight", "bias")
    if pe_bias_key in sd_2d:
        nav.patch_embed.bias.data.copy_(sd_2d[pe_bias_key])
        loaded.append(f"patch_embed.bias ← {pe_bias_key}")

    # Patch norm (LayerNorm)
    for suffix in ["weight", "bias"]:
        for pn_key in [f"patch_embed.norm.{suffix}", f"patch_embed.2.{suffix}"]:
            if pn_key in sd_2d:
                getattr(nav.patch_norm, suffix).data.copy_(sd_2d[pn_key])
                loaded.append(f"patch_norm.{suffix} ← {pn_key}")
                break
        else:
            skipped.append(f"patch_norm.{suffix}: no matching key found")

    # --- VSS Blocks (Stage 1 = layers.0) ---
    for j in range(num_vss_blocks):
        blk = nav.vss_blocks[j]
        prefix_2d = f"layers.0.blocks.{j}"

        # Weight mapping table: (our_attr, vmamba_key_suffix, needs_inflate)
        # Actual VMamba-Tiny keys use "ln_1" for norm and "self_attention" for SS2D
        # VMamba-Tiny has NO MLP branch (norm2/fc1/fc2 are randomly initialized)
        mappings = [
            # LayerNorm (VMamba uses "ln_1", we use "norm")
            ("norm.weight", "ln_1.weight", False),
            ("norm.bias", "ln_1.bias", False),
            # SS2D branch (VMamba uses "self_attention", we use separate attrs)
            ("in_proj.weight", "self_attention.in_proj.weight", False),
            ("dwconv3d.weight", "self_attention.conv2d.weight", True),  # 2D→3D inflate
            ("dwconv3d.bias", "self_attention.conv2d.bias", False),
            ("out_norm.weight", "self_attention.out_norm.weight", False),
            ("out_norm.bias", "self_attention.out_norm.bias", False),
            ("out_proj.weight", "self_attention.out_proj.weight", False),
            # NOTE: MLP branch (norm2, fc1, fc2) not in VMamba-Tiny → stays random init
        ]

        for our_attr, vm_suffix, needs_inflate in mappings:
            vm_key = f"{prefix_2d}.{vm_suffix}"
            if vm_key not in sd_2d:
                skipped.append(f"vss_blocks.{j}.{our_attr}: key '{vm_key}' not found")
                continue

            w = sd_2d[vm_key]
            param = blk
            for part in our_attr.split("."):
                param = getattr(param, part)

            if needs_inflate:
                # DWConv2d [C, 1, 3, 3] → DWConv3d [C, 1, 3, 3, 3]
                w = inflate_conv2d_weight(w, kernel_depth=3)

            if w.shape != param.data.shape:
                skipped.append(
                    f"vss_blocks.{j}.{our_attr}: shape mismatch "
                    f"(pretrained {w.shape} vs ours {param.data.shape})"
                )
                continue

            param.data.copy_(w)
            tag = " (inflated)" if needs_inflate else ""
            loaded.append(f"vss_blocks.{j}.{our_attr} ← {vm_key}{tag}")

    # --- Summary ---
    n_loaded = len(loaded)
    n_skipped = len(skipped)
    n_total_params = sum(1 for _ in nav.named_parameters())
    n_random = n_total_params - n_loaded

    logger.info(
        f"VMamba-Tiny inflation complete: "
        f"{n_loaded} params loaded, {n_skipped} skipped, {n_random} randomly initialized"
    )
    if skipped:
        logger.warning(f"Skipped weights: {skipped}")

    for msg in loaded[:5]:
        logger.info(f"  [loaded] {msg}")
    if n_loaded > 5:
        logger.info(f"  ... and {n_loaded - 5} more")

    return {"loaded": loaded, "skipped": skipped}


# ---------------------------------------------------------------------------
# Helper: get parameter groups for optimizer
# ---------------------------------------------------------------------------
def get_pretrained_nav_param_groups(
    nav: PretrainedVMamba3DNavigator,
    lr_pretrained: float = 0.10,
    lr_new: float = 1.00,
    lr_mamba: float = 0.30,
) -> List[Dict]:
    """
    Split navigator parameters into groups with different learning rate scales.

    Groups:
      1. Pretrained backbone (patch_embed, patch_norm, vss_blocks conv/linear/mlp): lr_pretrained
      2. VSSBlock3D SSM params (ssm_fwd, ssm_bwd): lr_mamba (partially transferable)
      3. New layers (proj, mamba): lr_new

    Returns list of dicts with 'params' and 'lr_scale' keys.
    """
    pretrained_params = []
    ssm_params = []
    new_params = []

    pretrained_prefixes = (
        "patch_embed.",
        "patch_norm.",
    )
    ssm_suffixes = ("ssm_fwd.", "ssm_bwd.")
    new_prefixes = ("proj.", "mamba.")

    for name, param in nav.named_parameters():
        if not param.requires_grad:
            continue

        if any(name.startswith(p) for p in pretrained_prefixes):
            pretrained_params.append(param)
        elif any(name.startswith(p) for p in new_prefixes):
            new_params.append(param)
        elif "vss_blocks" in name:
            # Inside VSSBlock3D: separate SSM from pretrained layers
            if any(s in name for s in ssm_suffixes):
                ssm_params.append(param)
            else:
                pretrained_params.append(param)
        else:
            new_params.append(param)

    groups = []
    if pretrained_params:
        groups.append({"params": pretrained_params, "lr_scale": lr_pretrained})
    if ssm_params:
        groups.append({"params": ssm_params, "lr_scale": lr_mamba})
    if new_params:
        groups.append({"params": new_params, "lr_scale": lr_new})

    logger.info(
        f"Nav param groups: pretrained={len(pretrained_params)} (lr×{lr_pretrained}), "
        f"ssm={len(ssm_params)} (lr×{lr_mamba}), "
        f"new={len(new_params)} (lr×{lr_new})"
    )
    return groups
