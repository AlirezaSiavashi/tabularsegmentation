"""
GraphCoW-Net — Stream A (Voxel Decoder)

A 5-level residual U-Net decoder consuming the encoder pyramid {F1..F5} and
producing per-voxel class logits at full input resolution. The two deepest
decoder stages (u3, u4) apply Dynamic Snake Convolution (SnakeConv3D), which
adapts sampling positions along curvilinear structures and is our strongest
prior for small tubular vessels (Pcom/Acom, 3rd-A2). Stages u1, u2, u5 use
standard residual conv blocks because snake conv is expensive at the largest
grids and unnecessary at the coarsest.

Design principles for the TMI submission (and why each matters):

  1. Residual decoder blocks. ResEnc nnU-Net ablations consistently show
     residual decoder > plain. The skip-within-block is cheap and helps
     small-vessel gradient flow.

  2. Deep supervision. Auxiliary heads at u1 (1/2), u2 (1/4), u3 (1/8) with
     descending loss weights. Necessary for Pcom/Acom because their voxel
     count is so small that the full-res head gradient alone is noisy.

  3. Dynamic Snake Conv at u3/u4 only. At u3 (stride 8) and u4 (stride 16)
     vessels are 2-6 voxels wide -- exactly where a curvature-following
     sampling pattern helps. At u1/u2 the grid is too large (snake is O(k)
     grid_samples), and at u5 the grid is too coarse (vessel is sub-voxel).

  4. Channel widths match encoder pyramid (384, 256, 128, 64, 32) halved
     at each decoder stage, keeping memory balanced. u1 uses 16 channels
     (below the 32-ch F1) because a 96^3 activation at 32 channels is
     >100MB in bf16 and we want headroom for Stream B.

  5. Hook-in points for Stream B (graph decoder conditioning). `cond_q4`
     and `cond_q3` are reserved inputs to `forward()`. For now they default
     to None and are no-ops; when Stream B is added, its node-query
     features get FiLM-modulated into u4 and u3. This is the same hook
     pattern we use in `pyramid_mamba_tr.py` so a reader familiar with
     that code finds the new decoder navigable.

  6. Gradient checkpointing on u1 and u2. Those are the largest activations
     (96^3 and 48^3). Checkpointing saves ~3 GB at the cost of one extra
     forward during backward. Toggleable via the `grad_ckpt` flag.

Inputs:
  features: Dict[str, Tensor] with keys "F1".."F5" from GraphCoWEncoder.
  cond_q4, cond_q3: optional conditioning tensors from Stream B
                    (placeholder; currently required shape is the feature map
                     at the corresponding stride, same channels as the decoder
                     stage).

Outputs:
  Dict with:
    "logits":     [B, num_classes, D, H, W]    full-resolution prediction
    "aux":        dict of auxiliary logits for deep supervision at strides
                  2, 4, 8 if `deep_supervision=True` else empty dict.
    "features":   dict of decoder stage outputs {"u1".."u5"} for use by
                  Stream B (which reads u3 and u4 to refine graph queries).

Memory (A100 40 GB, bf16, 192^3, batch=1, grad_ckpt=True):
  u5 block:     ~30 MB activation
  u4 + DSConv:  ~300 MB
  u3 + DSConv:  ~1.5 GB    (dominant -- DSConv k=9 at 24^3)
  u2 block:     ~250 MB
  u1 block:     ~700 MB    (ckpted -> freed during fwd)
  output head:  ~200 MB
  Grads:       ~2 GB
  Total decoder forward+backward: ~7-8 GB
  Encoder uses ~9 GB (see graphcow_encoder.py)
  -> ~17 GB used, leaves ~20 GB for Stream B + optimizer + CUDA workspace.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .graphcow_encoder import ConvBlock3D, _norm3d

# SnakeConv3D lives in src/snake_conv.py (existing, battle-tested).
# We import via a relative path from the models/ package.
try:
    from ..snake_conv import SnakeConv3D
except ImportError:  # pragma: no cover -- supports both script and package use
    from snake_conv import SnakeConv3D


# ---------------------------------------------------------------------------
# FiLM-style conditioning gate (reserved for Stream B)
# ---------------------------------------------------------------------------

class FiLMGate3D(nn.Module):
    """Per-voxel FiLM modulation: out = (1 + gamma) * x + beta.

    gamma and beta are predicted from a conditioning tensor `c` that is
    brought to x's spatial shape via trilinear upsampling. Kept small and
    identity-initialized so inserting this module does not change the
    decoder's behavior when `c` is None.

    This is the interface point where Stream B (the graph decoder) injects
    node-query features to refine the voxel decoder. For the Stream A-only
    training we pass c=None and this module is a no-op.
    """

    def __init__(self, x_ch: int, c_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(c_ch, 2 * x_ch, kernel_size=1, bias=True)
        # Identity init: gamma starts at 0, beta starts at 0 -> x passes through.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        if c is None:
            return x
        # Bring c to x's spatial shape. Trilinear is honest here: c is a coarse
        # context feature by design (graph queries are O(1) per class).
        if c.shape[-3:] != x.shape[-3:]:
            c = F.interpolate(c, size=x.shape[-3:], mode="trilinear",
                              align_corners=False)
        gb = self.proj(c)
        gamma, beta = gb.chunk(2, dim=1)
        return (1.0 + gamma) * x + beta


# ---------------------------------------------------------------------------
# Decoder stage: upsample -> concat skip -> (optional DSConv) -> residual block
# ---------------------------------------------------------------------------

class DecoderStage(nn.Module):
    """One decoder stage: upsample lower-res, merge with encoder skip, refine.

    Steps:
      (1) Upsample `deeper` 2x via transposed conv (learned). We use transposed
          conv rather than trilinear+conv because (a) this is a decoder not an
          image-upsampler, (b) nnU-Net results show transposed-conv reliably
          matches trilinear+1x1 in Dice on medical volumes, (c) it produces
          cleaner gradients for small-vessel signal propagation in our limited
          bench tests on v9-v15.
      (2) Concat with the encoder skip at this resolution.
      (3) Apply FiLM conditioning gate (no-op if c is None).
      (4) Either a SnakeConv3D (u3/u4) or a standard ConvBlock3D (others),
          then one more ConvBlock3D for post-snake smoothing.
      (5) Optional gradient checkpointing of the refinement block.

    Args:
      in_deeper:   channels of the deeper (lower-res) feature map before upsample.
      in_skip:     channels of the encoder skip at this resolution.
      out_ch:      channels after this stage.
      cond_ch:     channels of the optional Stream B conditioning (0 to disable).
      use_snake:   enable Dynamic Snake Convolution at this stage.
      snake_k:     SnakeConv kernel size (odd, typically 7 at u4, 9 at u3).
      snake_drop:  SnakeConv per-view dropout probability.
      grad_ckpt:   if True, wrap the refinement pass in torch.utils.checkpoint.
    """

    def __init__(
        self,
        in_deeper: int,
        in_skip: int,
        out_ch: int,
        cond_ch: int = 0,
        use_snake: bool = False,
        snake_k: int = 9,
        snake_drop: float = 0.5,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.use_snake = use_snake
        self.grad_ckpt = grad_ckpt

        # Learned 2x upsample. Stride-2 transposed conv is standard for
        # nnU-Net residual decoders.
        self.up = nn.ConvTranspose3d(in_deeper, out_ch, kernel_size=2, stride=2,
                                     bias=False)
        self.up_norm = _norm3d(out_ch)
        self.up_act = nn.GELU()

        # Skip projection: skip channels may differ from out_ch.
        merged_ch = out_ch + in_skip
        self.merge_proj = nn.Conv3d(merged_ch, out_ch, kernel_size=1, bias=False)
        self.merge_norm = _norm3d(out_ch)
        self.merge_act = nn.GELU()

        # Optional Stream-B conditioning gate (FiLM).
        self.film = FiLMGate3D(out_ch, cond_ch) if cond_ch > 0 else None

        # Refinement: SnakeConv3D if requested, else ConvBlock3D.
        # A trailing ConvBlock3D smooths the snake output and ensures the
        # stage ends with residual-block structure consistent across levels.
        if use_snake:
            self.refine1 = SnakeConv3D(out_ch, out_ch, kernel_size=snake_k,
                                       drop_p=snake_drop)
            self.refine1_norm = _norm3d(out_ch)
            self.refine1_act = nn.GELU()
        else:
            self.refine1 = ConvBlock3D(out_ch, out_ch, residual=True)
            self.refine1_norm = None
            self.refine1_act = None

        self.refine2 = ConvBlock3D(out_ch, out_ch, residual=True)

    def _refine(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_snake:
            h = self.refine1(x)
            h = self.refine1_act(self.refine1_norm(h))
            h = h + x  # residual wrap around SnakeConv for stability
        else:
            h = self.refine1(x)
        return self.refine2(h)

    def forward(
        self,
        deeper: torch.Tensor,
        skip: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # (1) Upsample the deeper features.
        x = self.up_act(self.up_norm(self.up(deeper)))

        # Resolve any sub-voxel size mismatch with the skip (rare but possible
        # when input dims are not a multiple of 32 along every axis).
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear",
                              align_corners=False)

        # (2) Concat and project.
        x = torch.cat([x, skip], dim=1)
        x = self.merge_act(self.merge_norm(self.merge_proj(x)))

        # (3) Stream-B conditioning (no-op if cond is None).
        if self.film is not None:
            x = self.film(x, cond)

        # (4) Refinement, with optional grad-ckpt to trade compute for memory.
        if self.grad_ckpt and self.training:
            x = cp.checkpoint(self._refine, x, use_reentrant=False)
        else:
            x = self._refine(x)

        return x


# ---------------------------------------------------------------------------
# Bottleneck over F5
# ---------------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """Small residual block operating on F5 before the first upsample.

    Why have one at all: F5 is the coarsest semantic feature (stride 32, 6^3
    at 192^3 input). A single residual block here consolidates the fused
    tri-planar ViT features before handing off to u5. Without it, u5 is
    directly reshaping fused-ViT tokens, which our pilot runs showed is
    under-trained for CoW multi-class.
    """

    def __init__(self, ch: int, expansion: float = 1.0):
        super().__init__()
        mid = max(8, int(round(ch * expansion)))
        self.block1 = ConvBlock3D(ch, mid, residual=False)
        self.block2 = ConvBlock3D(mid, ch, residual=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


# ---------------------------------------------------------------------------
# Auxiliary deep-supervision head
# ---------------------------------------------------------------------------

class DeepSupHead(nn.Module):
    """1x1x1 conv producing class logits at an intermediate resolution."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv3d(in_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Full voxel decoder
# ---------------------------------------------------------------------------

class GraphCoWVoxelDecoder(nn.Module):
    """Stream A: voxel decoder producing per-voxel class logits.

    Expected encoder pyramid channels (default): (F1=32, F2=64, F3=128, F4=256, F5=384).
    Decoder channel plan (default):              (u5=256, u4=128, u3=64, u2=32, u1=16).

    Snake conv is active at u3 and u4 by default. You can disable with
    `use_snake_u3=False` / `use_snake_u4=False` when ablating.

    Stream-B hooks: `cond_q4` injects at u4, `cond_q3` injects at u3. Each
    must match the spatial size of the encoder feature at that stride (or be
    coarser -- FiLMGate3D upsamples). Leave as None when training Stream A alone.
    """

    def __init__(
        self,
        num_classes: int = 14,
        enc_channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 384),
        dec_channels: Tuple[int, int, int, int, int] = (16, 32, 64, 128, 256),
        cond_channels: Tuple[int, int] = (0, 0),   # (u4, u3) Stream-B cond widths
        use_snake_u3: bool = True,
        use_snake_u4: bool = True,
        snake_k_u3: int = 9,
        snake_k_u4: int = 7,
        snake_drop: float = 0.5,
        deep_supervision: bool = True,
        grad_ckpt: bool = True,
        bottleneck_expansion: float = 1.0,
    ):
        super().__init__()
        if len(dec_channels) != 5 or len(enc_channels) != 5:
            raise ValueError("need 5 channel widths for both encoder and decoder")
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.dec_channels = dec_channels

        c_f1, c_f2, c_f3, c_f4, c_f5 = enc_channels
        c_u1, c_u2, c_u3, c_u4, c_u5 = dec_channels
        cond_u4, cond_u3 = cond_channels

        # Bottleneck operating on F5 directly. Output channels = c_u5.
        # We adapt F5 to c_u5 first via a 1x1x1, then the residual block.
        self.f5_adapt = nn.Sequential(
            nn.Conv3d(c_f5, c_u5, kernel_size=1, bias=False),
            _norm3d(c_u5),
            nn.GELU(),
        )
        self.bottleneck = BottleneckBlock(c_u5, expansion=bottleneck_expansion)

        # Decoder stages, coarsest to finest.
        # u5 block is already the bottleneck output (no upsample needed above F5).
        # u4: upsample u5 -> merge with F4 skip. Snake at u4 helps medium vessels.
        self.stage_u4 = DecoderStage(
            in_deeper=c_u5, in_skip=c_f4, out_ch=c_u4,
            cond_ch=cond_u4,
            use_snake=use_snake_u4, snake_k=snake_k_u4, snake_drop=snake_drop,
            grad_ckpt=False,  # u4 activation is small; don't ckpt
        )
        # u3: upsample u4 -> merge with F3 skip. Snake at u3 is where small
        # vessels (Pcom/Acom) benefit most.
        self.stage_u3 = DecoderStage(
            in_deeper=c_u4, in_skip=c_f3, out_ch=c_u3,
            cond_ch=cond_u3,
            use_snake=use_snake_u3, snake_k=snake_k_u3, snake_drop=snake_drop,
            grad_ckpt=False,  # DSConv at u3 + ckpt is subtle; keep off initially
        )
        # u2: upsample u3 -> merge with F2 skip.
        self.stage_u2 = DecoderStage(
            in_deeper=c_u3, in_skip=c_f2, out_ch=c_u2,
            cond_ch=0, use_snake=False,
            grad_ckpt=grad_ckpt,
        )
        # u1: upsample u2 -> merge with F1 skip (stride 2). Finest feature map
        # before the final full-res projection.
        self.stage_u1 = DecoderStage(
            in_deeper=c_u2, in_skip=c_f1, out_ch=c_u1,
            cond_ch=0, use_snake=False,
            grad_ckpt=grad_ckpt,
        )

        # Final full-res projection: u1 is at stride 2. We upsample to input
        # resolution via a stride-2 transposed conv and apply a 1x1x1 head.
        self.final_up = nn.ConvTranspose3d(c_u1, c_u1, kernel_size=2, stride=2,
                                           bias=False)
        self.final_norm = _norm3d(c_u1)
        self.final_act = nn.GELU()
        self.final_head = nn.Conv3d(c_u1, num_classes, kernel_size=1, bias=True)

        # Deep supervision heads at u3 (stride 8), u2 (stride 4), u1 (stride 2).
        # (u4 is skipped -- at stride 16 the DS signal is too coarse for small
        # vessels and tends to destabilize DSConv training in pilot runs.)
        if deep_supervision:
            self.aux_u3 = DeepSupHead(c_u3, num_classes)
            self.aux_u2 = DeepSupHead(c_u2, num_classes)
            self.aux_u1 = DeepSupHead(c_u1, num_classes)
        else:
            self.aux_u3 = self.aux_u2 = self.aux_u1 = None

        self._init_heads()

    def _init_heads(self):
        # Heads initialized with small weights so early training has mild
        # logits, avoiding saturation in the multi-class CE before the trunk
        # has adapted.
        for m in (self.final_head, self.aux_u3, self.aux_u2, self.aux_u1):
            if m is None:
                continue
            nn.init.normal_(m.weight if isinstance(m, nn.Conv3d) else m.head.weight,
                            std=0.01)
            bias = m.bias if isinstance(m, nn.Conv3d) else m.head.bias
            nn.init.zeros_(bias)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        cond_q4: Optional[torch.Tensor] = None,
        cond_q3: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encoder pyramid features. Expected shapes at 192^3 input:
        #   F1: [B, 32, 96, 96, 96]
        #   F2: [B, 64, 48, 48, 48]
        #   F3: [B, 128, 24, 24, 24]
        #   F4: [B, 256, 12, 12, 12]
        #   F5: [B, 384, 6, 6, 6]
        f1, f2, f3, f4, f5 = (features[k] for k in ("F1", "F2", "F3", "F4", "F5"))

        # Bottleneck on F5 -> u5.
        u5 = self.bottleneck(self.f5_adapt(f5))

        # Progressive upsample with skips.
        u4 = self.stage_u4(u5, f4, cond=cond_q4)
        u3 = self.stage_u3(u4, f3, cond=cond_q3)
        u2 = self.stage_u2(u3, f2, cond=None)
        u1 = self.stage_u1(u2, f1, cond=None)

        # Final full-res.
        top = self.final_act(self.final_norm(self.final_up(u1)))
        logits = self.final_head(top)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "features": {"u1": u1, "u2": u2, "u3": u3, "u4": u4, "u5": u5},
        }

        if self.deep_supervision:
            out["aux"] = {
                # stride 8, 4, 2 respectively (relative to input)
                "u3": self.aux_u3(u3),
                "u2": self.aux_u2(u2),
                "u1": self.aux_u1(u1),
            }
        else:
            out["aux"] = {}

        return out

    # --- convenience for the training script ---------------------------------

    @staticmethod
    def aux_loss_weights() -> Dict[str, float]:
        """Default loss weights for the deep-supervision heads.

        Weights follow the nnU-Net pattern: main = 1.0, then ~1/2^level for
        each coarser head. Normalized so they sum to 1.0 for clarity.
        """
        raw = {"main": 1.0, "u1": 0.5, "u2": 0.25, "u3": 0.125}
        s = sum(raw.values())
        return {k: v / s for k, v in raw.items()}

    def upsample_aux_to_target(
        self, aux_logits: Dict[str, torch.Tensor],
        target_shape: Tuple[int, int, int],
    ) -> Dict[str, torch.Tensor]:
        """Upsample each aux map to the target (label) shape, ready for CE/Dice.

        Used in the training loop when the loss is computed at full resolution.
        Alternative: downsample the label to each aux stride (cheaper, nnU-Net
        default). Caller decides; this helper exists for the upsample path.
        """
        return {
            k: F.interpolate(v, size=target_shape, mode="trilinear",
                             align_corners=False)
            for k, v in aux_logits.items()
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=int, nargs=3, default=[192, 192, 192],
                   metavar=("D", "H", "W"))
    p.add_argument("--classes", type=int, default=14)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    D, H, W = args.patch
    enc_ch = (32, 64, 128, 256, 384)
    fake = {
        "F1": torch.randn(1, enc_ch[0], D // 2,  H // 2,  W // 2,  device=device, dtype=dtype),
        "F2": torch.randn(1, enc_ch[1], D // 4,  H // 4,  W // 4,  device=device, dtype=dtype),
        "F3": torch.randn(1, enc_ch[2], D // 8,  H // 8,  W // 8,  device=device, dtype=dtype),
        "F4": torch.randn(1, enc_ch[3], D // 16, H // 16, W // 16, device=device, dtype=dtype),
        "F5": torch.randn(1, enc_ch[4], D // 32, H // 32, W // 32, device=device, dtype=dtype),
    }

    dec = GraphCoWVoxelDecoder(num_classes=args.classes, enc_channels=enc_ch).to(device)
    if dtype == torch.bfloat16:
        dec = dec.to(dtype)
    dec.train()

    with torch.cuda.amp.autocast(enabled=dtype == torch.bfloat16, dtype=torch.bfloat16):
        out = dec(fake)

    print("Decoder output shapes:")
    print(f"  logits     : {tuple(out['logits'].shape)}   dtype={out['logits'].dtype}")
    for k, v in out["features"].items():
        print(f"  feat {k}    : {tuple(v.shape)}")
    for k, v in out["aux"].items():
        print(f"  aux  {k}    : {tuple(v.shape)}")

    n_all = sum(p.numel() for p in dec.parameters())
    n_train = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"Total params: {n_all/1e6:.2f}M, trainable: {n_train/1e6:.2f}M")
    if device == "cuda":
        print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
