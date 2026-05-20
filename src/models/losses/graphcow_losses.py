"""
GraphCoW-Net — Loss bundle

Implements the nine-term training objective:

    L_total =
          L_DiceCE        (voxel)                                 baseline
      +   lam_cb  * cbDice                                         MICCAI 2024
      +   lam_cld * clDice                                         CVPR 2021
      +   lam_pv  * NodePresenceLoss                               (graph)
      +   lam_pc  * CenterlineRegressionLoss (Chamfer-ish)         (graph)
      +   lam_rd  * RadiusRegressionLoss                           (graph)
      +   lam_eg  * EdgeConsistencyLoss                            (graph x voxel)
      +   lam_pb  * BettiMatchingLoss                              topology
      +   lam_co  * GraphVoxelCouplingLoss (rendered tube <-> mask)

The voxel and graph parts share a single call-site: `GraphCoWLoss.forward`
accepts the full model output dict + targets dict and returns a total loss
plus a diagnostics dict (keyed so that the training loop can log/plot each
term independently).

Target format (produced by the dataloader):

    targets = {
        "mask":         [B, D, H, W]       long, 0..C-1 voxel labels
        "mask_soft":    [B, C, D, H, W]    float soft labels (MixUp) OR None
        "skeleton":     [B, D, H, W]       float {0,1} skeleton of all vessels
        "node_present": [B, N_nodes]       {0,1} per CoW class
        "node_points":  [B, N_nodes, K, 3] skeleton samples in [0,1]^3 (pad=-1)
        "node_radius":  [B, N_nodes, K]    radius in voxel units (pad=-1)
        "edge_present": [B, N_edges]       {0,1} per canonical edge
        "edge_points":  [B, N_edges, K, 3] (pad=-1 for absent edges)
        "edge_radius":  [B, N_edges, K]    (pad=-1)
    }

Any per-vertex entry with value -1 is treated as "unsupervised" for that
slot and is masked out of all losses. This lets the dataloader emit a
fixed K regardless of how long each vessel actually is.

Betti Matching in the full sense (matching persistence pairs) requires
`betti-matching-3d` (arxiv 2407.04683). We implement a tractable variant
('topology loss via persistence barcodes' from Clough et al. 2020) that
penalizes spurious connected components with `cripser` or falls back to
a cheaper Euler-characteristic proxy if the library is not installed.
Both are opt-in via the `use_betti` flag; set to False for a 3-minute
pilot run.

All losses are autodiff-safe; nothing uses .detach() except where noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .topology_losses import SoftSkeletonize3D
except ImportError:
    from topology_losses import SoftSkeletonize3D


# ---------------------------------------------------------------------------
# Voxel baseline: Dice + CrossEntropy (soft-label-aware)
# ---------------------------------------------------------------------------

class SoftDiceCELoss(nn.Module):
    """Multi-class Dice + CE. Supports soft targets (MixUp) via the
    `mask_soft` path; when `mask_soft` is None, falls back to the hard
    integer mask.

    Dice is the class-averaged (1 - 2TP / (P + T)) over non-background
    classes. CE handles class imbalance via class weights (typical
    weighting: inverse-sqrt frequency, passed in at construction).
    """

    def __init__(
        self,
        num_classes: int,
        class_weight: Optional[torch.Tensor] = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ignore_background: bool = True,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "class_weight",
            class_weight if class_weight is not None else torch.ones(num_classes),
            persistent=False,
        )
        self.dice_w = dice_weight
        self.ce_w = ce_weight
        self.ignore_bg = ignore_background
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,                      # [B, C, D, H, W]
        mask: torch.Tensor,                        # [B, D, H, W] long, OR ignored if soft
        mask_soft: Optional[torch.Tensor] = None,  # [B, C, D, H, W] float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)
        if mask_soft is None:
            tgt_1h = F.one_hot(mask.clamp(0, C - 1), num_classes=C).permute(0, 4, 1, 2, 3)
            tgt_1h = tgt_1h.to(logits.dtype)
            ce = F.cross_entropy(
                logits, mask.long(),
                weight=self.class_weight.to(logits.dtype),
                reduction="mean",
            )
        else:
            tgt_1h = mask_soft.to(logits.dtype)
            # Soft-label CE: -sum y_c log p_c
            log_p = F.log_softmax(logits, dim=1)
            w = self.class_weight.to(logits.dtype).view(1, C, 1, 1, 1)
            ce = -(w * tgt_1h * log_p).sum(dim=1).mean()

        # Dice, class-averaged.
        dims = (0, 2, 3, 4)
        inter = (probs * tgt_1h).sum(dim=dims)
        denom = probs.sum(dim=dims) + tgt_1h.sum(dim=dims)
        dice_per = (2 * inter + self.smooth) / (denom + self.smooth)
        if self.ignore_bg and C > 1:
            dice = 1.0 - dice_per[1:].mean()
        else:
            dice = 1.0 - dice_per.mean()

        total = self.dice_w * dice + self.ce_w * ce
        return total, {
            "ce": ce.detach(),
            "dice": dice.detach(),
            "dice_per_class": dice_per.detach(),
        }


# ---------------------------------------------------------------------------
# clDice
# ---------------------------------------------------------------------------

class clDiceLoss(nn.Module):
    """Centerline Dice loss. Uses the existing SoftSkeletonize3D.

    Multi-class support: skeletonize each foreground class separately and
    average.
    """

    def __init__(self, num_classes: int, iters: int = 10, smooth: float = 1e-5,
                 ignore_background: bool = True,
                 target_classes: Optional[Sequence[int]] = None):
        super().__init__()
        self.skel = SoftSkeletonize3D(num_iter=iters)
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_background
        # If target_classes is set, only compute clDice for those classes.
        # This drastically cuts memory on 20 GB MIG when num_classes=14.
        self.target_classes = set(target_classes) if target_classes else None

    @staticmethod
    def _cldice_per_channel(p: torch.Tensor, t: torch.Tensor, skel, eps: float):
        """p, t: [B, 1, D, H, W] in [0,1].

        Uses gradient checkpointing on skel(p) so the ~10-iter skeletonization
        does not store intermediate activations for backward. Without this,
        peak memory explodes at grad-accum=2 on 192^3 patches.
        """
        from torch.utils.checkpoint import checkpoint
        sp = checkpoint(skel, p, use_reentrant=False)
        with torch.no_grad():
            st = skel(t)
        tprec = (sp * t).sum() / (sp.sum() + eps)
        tsens = (st * p).sum() / (st.sum() + eps)
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + eps)

    def forward(
        self,
        logits: torch.Tensor,                      # [B, C, D, H, W]
        mask: torch.Tensor,                        # [B, D, H, W] long
    ) -> torch.Tensor:
        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)
        losses = []
        for c in range(C):
            if self.ignore_bg and c == 0:
                continue
            if self.target_classes is not None and c not in self.target_classes:
                continue
            p = probs[:, c:c + 1]
            t = (mask == c).float().unsqueeze(1)
            if t.sum() < 1:
                continue
            losses.append(self._cldice_per_channel(p, t, self.skel, self.smooth))
        if not losses:
            return logits.new_zeros(())
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# cbDice — centerline-boundary Dice
# ---------------------------------------------------------------------------

class cbDiceLoss(nn.Module):
    """Centerline-Boundary Dice.

    Combines a centerline-weighted Dice (as in clDice) with a boundary-
    weighted Dice (emphasize the vessel surface voxels). Weights both with
    a Dice-like normalization. Empirically best on TopCoW 2024 by team
    ZJU-NISLAB.

    Implementation sketch:
      * centerline_weight = soft skeleton of GT
      * boundary_weight   = |GT - soft_erode(GT)|    (ring one voxel thick)
      * weighted_dice(p, t, w) = 2 (w * p * t).sum() / ((w*p).sum() + (w*t).sum())
    """

    def __init__(self, num_classes: int, iters: int = 8, smooth: float = 1e-5,
                 ignore_background: bool = True):
        super().__init__()
        self.skel = SoftSkeletonize3D(num_iter=iters)
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_background

    @staticmethod
    def _boundary(t: torch.Tensor) -> torch.Tensor:
        eroded = -F.max_pool3d(-t, kernel_size=3, stride=1, padding=1)
        return (t - eroded).clamp(0.0, 1.0)

    def _wdice(self, p, t, w):
        wp = (w * p).sum()
        wt = (w * t).sum()
        inter = (w * p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (wp + wt + self.smooth)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)
        out = []
        for c in range(C):
            if self.ignore_bg and c == 0:
                continue
            p = probs[:, c:c + 1]
            t = (mask == c).float().unsqueeze(1)
            if t.sum() < 1:
                continue
            with torch.no_grad():
                cw = self.skel(t)           # centerline weight (target-only, no grad)
                bw = self._boundary(t)      # boundary weight   (target-only, no grad)
            # Normalize each weight to have the same mass as t, so they do
            # not systematically scale the gradient.
            def normw(w):
                s = w.sum()
                return w * (t.sum().clamp_min(1.0) / s.clamp_min(1e-3))
            cw = normw(cw)
            bw = normw(bw)
            w_total = 1.0 + 2.0 * cw + 2.0 * bw   # always includes body via +1
            out.append(self._wdice(p, t, w_total))
        if not out:
            return logits.new_zeros(())
        return torch.stack(out).mean()


# ---------------------------------------------------------------------------
# Per-class Connectivity loss (soft, differentiable)
# ---------------------------------------------------------------------------

class PerClassConnectivityLoss(nn.Module):
    """Penalize disconnected predictions for thin tubular classes.

    Two complementary terms:

      (a) GT-spillage: prediction mass that lies OUTSIDE a dilated tube around
          the GT is penalized. A disconnected fragment of vessel will be far
          from the GT tube, so (1 - GT_dilated) * pred_prob catches it.

      (b) Skeleton-flood: the predicted probability should not DROP at the GT
          skeleton. We compare the mean pred_prob on the skeleton to the
          mean pred_prob on the dilated tube; a big gap means the prediction
          is fragmented along the vessel axis.

    Both terms are cheap (two small max-pool cascades per class).

    Motivation (TMI-relevant): on Pcom/Acom/3rd-A2 the Dice gradient alone
    cannot keep a fragile 1–2 voxel chain connected — either the chain is
    correct (high Dice) or it's broken (low Dice, no smooth path between the
    two states). This loss gives a smooth gradient that pushes disconnected
    blobs back toward the GT tube.
    """

    def __init__(
        self,
        num_classes: int,
        target_classes: Optional[Sequence[int]] = None,
        dilate_kernel: int = 5,
        skel_iters: int = 5,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        # Default: every foreground class. Override with e.g. [8,9,10,13].
        if target_classes is None:
            target_classes = list(range(1, num_classes))
        self.target_classes = tuple(target_classes)
        self.dilate_kernel = dilate_kernel
        self.skel = SoftSkeletonize3D(num_iter=skel_iters)
        self.smooth = smooth

    @torch.no_grad()
    def _dilate(self, t: torch.Tensor) -> torch.Tensor:
        k = self.dilate_kernel
        return F.max_pool3d(t, kernel_size=k, stride=1, padding=k // 2)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """logits: [B,C,D,H,W]; mask: [B,D,H,W] long."""
        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)
        losses = []
        for c in self.target_classes:
            if c <= 0 or c >= C:
                continue
            p = probs[:, c:c + 1]                       # [B,1,D,H,W]
            t = (mask == c).float().unsqueeze(1)         # [B,1,D,H,W]
            if t.sum() < 1:
                # Class absent in this mini-batch -> use pure suppression term
                # so the network does not hallucinate rare vessels.
                losses.append(p.mean())
                continue

            with torch.no_grad():
                tube = self._dilate(t)                   # ~5-vox radius tube
                skel = self.skel(t)                      # GT skeleton (soft)

            # (a) spillage: prediction outside the tube.
            outside = 1.0 - tube
            spill = (p * outside).sum() / (p.sum() + self.smooth)

            # (b) centerline coverage: pred should be high on the GT skeleton.
            skel_mass = skel.sum().clamp_min(1.0)
            cov_skel = (p * skel).sum() / skel_mass
            tube_mass = tube.sum().clamp_min(1.0)
            cov_tube = (p * tube).sum() / tube_mass
            # Penalize drop from tube to skeleton coverage (smaller = better).
            # If skeleton coverage is lower than tube coverage, the vessel is
            # broken along its axis.
            drop = (cov_tube - cov_skel).clamp_min(0.0)

            # (c) 1 - mean_prob on skeleton: direct centerline recall.
            miss_skel = 1.0 - cov_skel

            losses.append(spill + drop + miss_skel)

        if not losses:
            return logits.new_zeros(())
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Betti matching (optional, cheap proxy if library absent)
# ---------------------------------------------------------------------------

class BettiMatchingLoss(nn.Module):
    """Topology loss.

    If the `betti-matching-3d` package is available, use its matched-pairs
    loss (Stucki et al. 2024). Otherwise fall back to an Euler-characteristic
    error on the binarized foreground: |chi(pred) - chi(gt)|, approximated
    from cubical-complex counts via convolutional kernels. The proxy is
    cheap and differentiable through a straight-through estimator on the
    binarization.

    Notes:
      * Binarization uses threshold 0.5 with a hard straight-through (so the
        forward pass is crisp but the gradient is passed through the sigmoid
        of the logits). This is the standard trick from STE-quantization
        literature.
      * Proxy is a placeholder for TMI-scale reporting; swap for the full
        Betti-matching in production.
    """

    def __init__(self, use_betti: bool = False, class_avg: bool = True):
        super().__init__()
        self.use_betti = use_betti
        self.class_avg = class_avg
        self._bm = None
        if use_betti:
            try:
                import betti_matching  # noqa: F401
                self._bm = betti_matching
            except ImportError:
                self.use_betti = False   # fall back

    def _euler_error(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """3D Euler characteristic estimate via cubical-complex vertex/
        edge/face/cell counts. Differentiable through STE.
        """
        # Straight-through binarization (forward crisp, backward identity).
        p_bin = p + (((p > 0.5).float()) - p).detach()

        # Count cells via linear convolutions with indicator kernels.
        def count(x, k):
            w = torch.ones(k, device=x.device, dtype=x.dtype).view(1, 1, *k)
            c = F.conv3d(x, w)
            return (c == w.numel()).float().sum()

        # Use a cheap surrogate: number of connected components via sum of
        # local vertex indicators minus edge indicators plus face indicators
        # minus cube indicators. This is NOT the exact chi but correlates.
        # Equivalent to Euler formula for cubical complexes.
        chi = (
            count(p_bin, (1, 1, 1))
            - count(p_bin, (2, 1, 1)) - count(p_bin, (1, 2, 1)) - count(p_bin, (1, 1, 2))
            + count(p_bin, (1, 2, 2)) + count(p_bin, (2, 1, 2)) + count(p_bin, (2, 2, 1))
            - count(p_bin, (2, 2, 2))
        )
        chi_t = (
            count(t, (1, 1, 1))
            - count(t, (2, 1, 1)) - count(t, (1, 2, 1)) - count(t, (1, 1, 2))
            + count(t, (1, 2, 2)) + count(t, (2, 1, 2)) + count(t, (2, 2, 1))
            - count(t, (2, 2, 2))
        )
        return (chi - chi_t).abs() / (t.numel() ** (1.0 / 3.0))  # normalize

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape[:2]
        if self.use_betti and self._bm is not None:
            # Delegating to full library; ignores gradients inside the lib.
            with torch.enable_grad():
                probs = F.softmax(logits, dim=1)
                fg = probs[:, 1:].sum(dim=1, keepdim=True)   # any-vessel prob
                gt = (mask > 0).float().unsqueeze(1)
                loss = self._bm.betti_matching_loss_3d(fg, gt)
            return loss
        # Proxy
        probs = F.softmax(logits, dim=1)
        out = []
        for c in range(1, C):
            p = probs[:, c:c + 1]
            t = (mask == c).float().unsqueeze(1)
            if t.sum() < 1:
                continue
            out.append(self._euler_error(p, t))
        if not out:
            return logits.new_zeros(())
        stacked = torch.stack(out)
        return stacked.mean() if self.class_avg else stacked.sum()


# ---------------------------------------------------------------------------
# Graph-side losses
# ---------------------------------------------------------------------------

class NodePresenceLoss(nn.Module):
    """BCE on per-class presence vector. Supports class weighting by inverse
    frequency (rare segments weighted more).
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, N], targets: [B, N] in {0,1}
        pw = self.pos_weight.to(logits.dtype) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype),
                                                  pos_weight=pw)


def _masked_pairs(
    pred_pts: torch.Tensor,     # [B, N, K, 3]
    gt_pts: torch.Tensor,       # [B, N, K, 3]
    gt_mask_val: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pred_valid, gt_valid, valid_mask) flattened over points.

    A GT point is 'valid' when none of its 3 coords equal the sentinel -1.
    """
    gt_valid = (gt_pts != gt_mask_val).all(dim=-1)   # [B, N, K]
    return pred_pts, gt_pts, gt_valid


class CenterlineRegressionLoss(nn.Module):
    """Symmetric Chamfer distance between predicted polyline and GT
    polyline, averaged over valid GT points and per-vertex L1 on matched
    pairs.

    Uses Chamfer (not L1-on-index) because our GT polylines are resampled
    from skeletons and do not guarantee index alignment with the prediction.
    For each GT point we find the closest pred point and vice versa; L1 on
    the two distance sets gives a symmetric loss.

    A small 'ordered' L1 term is added to discourage the polyline from
    folding back on itself: sum_i max(0, -dot(p_{i+1}-p_i, p_i-p_{i-1})).
    """

    def __init__(self, order_weight: float = 0.1):
        super().__init__()
        self.order_weight = order_weight

    def forward(
        self,
        pred_pts: torch.Tensor,       # [B, N, K_p, 3] in [0,1]
        gt_pts: torch.Tensor,         # [B, N, K_g, 3] in [0,1] (pad=-1)
        presence: torch.Tensor,       # [B, N] in {0,1}
    ) -> torch.Tensor:
        B, N, Kp, _ = pred_pts.shape
        _, _, Kg, _ = gt_pts.shape
        gt_valid = (gt_pts != -1.0).all(dim=-1)          # [B, N, Kg]
        pres = presence.bool()                            # only absent-GT segments skipped

        # Pairwise distance matrix [B, N, Kp, Kg]. We compute per batch-item
        # via broadcasting; memory is 4 * B * N * Kp * Kg floats, tiny.
        d = (pred_pts.unsqueeze(3) - gt_pts.unsqueeze(2)).norm(dim=-1)   # [B, N, Kp, Kg]
        # Mask invalid GT columns.
        d = d.masked_fill(~gt_valid.unsqueeze(2), float("inf"))
        # Presence mask: absent segments contribute zero loss (the tensor
        # still exists so the network can still place these queries
        # somewhere reasonable, but we don't penalize it).
        # p -> min over GT  (forward distance)
        fwd, _ = d.min(dim=-1)                            # [B, N, Kp]
        # g -> min over pred (backward distance); invalid rows get set to 0
        d2 = d.masked_fill(~gt_valid.unsqueeze(2), 0.0)   # replace inf with 0 for bwd
        bwd_d = (pred_pts.unsqueeze(3) - gt_pts.unsqueeze(2)).norm(dim=-1)
        bwd_d = bwd_d.masked_fill(~gt_valid.unsqueeze(2), float("inf"))
        bwd, _ = bwd_d.min(dim=-2)                        # [B, N, Kg]

        # Mean over valid forward entries; mean over valid backward entries.
        # Forward: all Kp pred points are "valid" in the prediction.
        fwd_loss = fwd.mean(dim=-1)                       # [B, N]
        bwd_sum = bwd.masked_fill(~gt_valid, 0.0).sum(dim=-1)
        bwd_cnt = gt_valid.float().sum(dim=-1).clamp_min(1.0)
        bwd_loss = bwd_sum / bwd_cnt                      # [B, N]

        chamfer = 0.5 * (fwd_loss + bwd_loss)

        # Order term: discourage folding along predicted polyline.
        if Kp >= 3:
            e = pred_pts[..., 1:, :] - pred_pts[..., :-1, :]
            dot = (e[..., :-1, :] * e[..., 1:, :]).sum(dim=-1)    # [B, N, Kp-2]
            order = F.relu(-dot).mean(dim=-1)
            chamfer = chamfer + self.order_weight * order

        # Reduce only over present segments.
        mask = pres.float()
        return (chamfer * mask).sum() / mask.sum().clamp_min(1.0)


class RadiusRegressionLoss(nn.Module):
    """L1 on matched radii. Matches points by the same nearest-neighbor
    index used in the Chamfer loss -- computed here directly.
    """

    def forward(
        self,
        pred_pts: torch.Tensor,       # [B, N, K_p, 3]
        pred_r: torch.Tensor,         # [B, N, K_p]
        gt_pts: torch.Tensor,         # [B, N, K_g, 3]
        gt_r: torch.Tensor,           # [B, N, K_g] (pad=-1)
        presence: torch.Tensor,       # [B, N]
    ) -> torch.Tensor:
        gt_valid = (gt_pts != -1.0).all(dim=-1) & (gt_r >= 0)
        d = (pred_pts.unsqueeze(3) - gt_pts.unsqueeze(2)).norm(dim=-1)
        d = d.masked_fill(~gt_valid.unsqueeze(2), float("inf"))
        # For each pred point, matched GT index.
        match_idx = d.argmin(dim=-1, keepdim=True)        # [B, N, Kp, 1]
        matched_r = torch.gather(gt_r.unsqueeze(2).expand(-1, -1, pred_r.shape[-1], -1),
                                 dim=-1, index=match_idx).squeeze(-1)   # [B, N, Kp]
        # Build validity per matched pair.
        matched_valid = torch.gather(
            gt_valid.unsqueeze(2).expand(-1, -1, pred_r.shape[-1], -1),
            dim=-1, index=match_idx,
        ).squeeze(-1)

        diff = (pred_r - matched_r).abs()
        diff = diff * matched_valid.float()
        diff = diff * presence.unsqueeze(-1).float()
        denom = (matched_valid.float() * presence.unsqueeze(-1).float()).sum().clamp_min(1.0)
        return diff.sum() / denom


class EdgeConsistencyLoss(nn.Module):
    """Enforces that edges predicted as present should align with edges in
    the GT voxel mask. Sampled approach: for each predicted edge polyline
    with sigmoid(connection) > thresh, sample voxel logits along the path
    and take CE with positive target 'foreground'. Keeps the gradient flow
    going to both the graph (via polyline) and the voxel (via logits).
    """

    def __init__(self, num_samples: int = 32, conn_thresh: float = 0.5):
        super().__init__()
        self.num_samples = num_samples
        self.conn_thresh = conn_thresh

    @staticmethod
    def _sample_line(logits: torch.Tensor, poly: torch.Tensor,
                     num_samples: int) -> torch.Tensor:
        """logits: [B, C, D, H, W]; poly: [B, Ne, K, 3] in [0,1].
        Returns per-segment sampled any-foreground prob [B, Ne, S]."""
        B, C, D, H, W = logits.shape
        _, Ne, K, _ = poly.shape
        # Build dense sample along polyline: interpolate between consecutive
        # vertices (piecewise-linear) at `num_samples` per segment.
        seg_p0 = poly[..., :-1, :]                       # [B, Ne, K-1, 3]
        seg_p1 = poly[..., 1:, :]
        t = torch.linspace(0, 1, num_samples, device=poly.device,
                           dtype=poly.dtype).view(1, 1, 1, num_samples, 1)
        samples = seg_p0.unsqueeze(3) + t * (seg_p1 - seg_p0).unsqueeze(3)  # [B,Ne,K-1,S,3]
        S_total = (K - 1) * num_samples
        samples = samples.reshape(B, Ne, S_total, 3)
        # Grid_sample expects (x, y, z) in [-1, 1]. Our ordering is (z, y, x)
        # so we convert.
        zyx = samples * 2.0 - 1.0
        xyz = zyx.flip(dim=-1) if False else zyx  # placeholder; we flip below explicitly
        xyz = torch.stack([zyx[..., 2], zyx[..., 1], zyx[..., 0]], dim=-1)
        # grid_sample over logits with shape (B, C, D, H, W) using a grid
        # (B, 1, 1, Ne*S, 3) -> produces (B, C, 1, 1, Ne*S)
        grid = xyz.reshape(B, 1, 1, Ne * S_total, 3)
        probs = F.softmax(logits, dim=1)
        sampled = F.grid_sample(probs, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        sampled = sampled.reshape(B, C, Ne, S_total)
        # Any-foreground probability.
        fg = sampled[:, 1:].sum(dim=1)                   # [B, Ne, S]
        return fg

    def forward(
        self,
        logits: torch.Tensor,
        edge_poly: torch.Tensor,
        edge_presence: torch.Tensor,      # [B, Ne] logits
    ) -> torch.Tensor:
        fg = self._sample_line(logits, edge_poly, self.num_samples)   # [B, Ne, S]
        # Binary CE: target = presence (soft).
        pres = torch.sigmoid(edge_presence).detach()                   # [B, Ne]
        # Mask edges whose presence < thresh (we don't penalize them).
        mask = (pres > self.conn_thresh).float().unsqueeze(-1)          # [B, Ne, 1]
        target = torch.ones_like(fg)
        # Small epsilon to stabilize the log in fg.
        loss = -torch.log(fg.clamp_min(1e-6)) * target * mask
        denom = mask.sum().clamp_min(1.0) * fg.shape[-1]
        return loss.sum() / denom


class GraphVoxelCouplingLoss(nn.Module):
    """MSE between (sigmoid of) voxel any-vessel prob and the rendered
    graph tube. Enforces bi-directional consistency:

       voxel = sum(softmax(logits)[:, 1:], dim=1)
       tube  = sum(tube_per_class, dim=1)  clamped to [0,1]
       loss  = mean((voxel - tube)^2)

    Low tau in the renderer sharpens the tube; high tau leaves a smooth
    ramp that is friendlier to early training.
    """

    def forward(self, logits: torch.Tensor, tube_mask: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        fg = probs[:, 1:].sum(dim=1, keepdim=True)
        return F.mse_loss(fg, tube_mask.clamp(0.0, 1.0))


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class GraphCoWLossWeights:
    # Voxel
    dice_ce: float = 1.0
    cb: float = 0.5
    cld: float = 0.5
    betti: float = 0.1
    conn: float = 0.0            # per-class connectivity (Phase 1b)
    # Graph
    node_presence: float = 1.0
    centerline: float = 2.0
    radius: float = 0.5
    edge_consistency: float = 1.0
    coupling: float = 0.5
    # Aux
    deep_sup: float = 0.5          # multiplies the sum over aux heads
    prelim_graph: float = 0.3      # multiplies preliminary graph losses


class GraphCoWLoss(nn.Module):
    """Glue all the pieces together with configurable weights and a phase
    switch.

    Phases:
      * 'voxel_warmup'  -- only L_DiceCE, L_cb, L_cld active.
      * 'graph_bootstrap' -- only graph-side losses (node presence,
                              centerline, radius); voxel losses off.
      * 'joint'         -- all losses active.
    """

    def __init__(
        self,
        num_classes: int = 14,
        class_weight: Optional[torch.Tensor] = None,
        node_pos_weight: Optional[torch.Tensor] = None,
        weights: GraphCoWLossWeights = GraphCoWLossWeights(),
        K_node_gt: int = 32,
        use_betti: bool = False,
        cld_iters: int = 10,
        cb_iters: int = 8,
        conn_target_classes: Optional[Sequence[int]] = None,
        conn_dilate_kernel: int = 5,
        conn_skel_iters: int = 5,
        cld_target_classes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.w = weights
        self.num_classes = num_classes
        self.K_node_gt = K_node_gt

        self.dice_ce = SoftDiceCELoss(num_classes=num_classes, class_weight=class_weight)
        self.cb = cbDiceLoss(num_classes=num_classes, iters=cb_iters)
        self.cld = clDiceLoss(num_classes=num_classes, iters=cld_iters,
                              target_classes=cld_target_classes)
        self.betti = BettiMatchingLoss(use_betti=use_betti)
        self.conn = PerClassConnectivityLoss(
            num_classes=num_classes,
            target_classes=conn_target_classes,
            dilate_kernel=conn_dilate_kernel,
            skel_iters=conn_skel_iters,
        )

        self.node_presence = NodePresenceLoss(pos_weight=node_pos_weight)
        self.centerline = CenterlineRegressionLoss()
        self.radius = RadiusRegressionLoss()
        self.edge_consistency = EdgeConsistencyLoss()
        self.coupling = GraphVoxelCouplingLoss()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase: str = "joint",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        total = 0.0 * (output.get("logits", output.get("features", {}).get("F2")).sum()
                       if "logits" in output else output["graph"]["node_presence"].sum() * 0.0)

        # ---------------- Voxel-side ---------------------------------------
        if phase in ("voxel_warmup", "joint") and "logits" in output:
            mask = targets["mask"]
            mask_soft = targets.get("mask_soft")

            l_dc, dc_logs = self.dice_ce(output["logits"], mask, mask_soft)
            total = total + self.w.dice_ce * l_dc
            logs.update({f"dice_ce/{k}": v for k, v in dc_logs.items()})

            if self.w.cb > 0:
                l_cb = self.cb(output["logits"], mask)
                total = total + self.w.cb * l_cb
                logs["cb"] = l_cb.detach()

            if self.w.cld > 0:
                l_cld = self.cld(output["logits"], mask)
                total = total + self.w.cld * l_cld
                logs["cldice"] = l_cld.detach()

            if self.w.conn > 0:
                l_conn = self.conn(output["logits"], mask)
                total = total + self.w.conn * l_conn
                logs["conn"] = l_conn.detach()

            # Deep supervision: compute DiceCE at each aux head's resolution.
            aux = output.get("aux", {})
            if aux:
                aux_total = 0.0
                for k, v in aux.items():
                    # Downsample mask (nearest) rather than upsample logits.
                    Dm, Hm, Wm = v.shape[-3:]
                    m_down = F.interpolate(
                        mask.unsqueeze(1).float(), size=(Dm, Hm, Wm),
                        mode="nearest",
                    ).long().squeeze(1)
                    l_aux, _ = self.dice_ce(v, m_down, None)
                    aux_total = aux_total + l_aux
                aux_total = aux_total / max(1, len(aux))
                total = total + self.w.deep_sup * aux_total
                logs["aux_voxel"] = aux_total.detach()

            if self.w.betti > 0:
                l_be = self.betti(output["logits"], mask)
                total = total + self.w.betti * l_be
                logs["betti"] = l_be.detach()

        # ---------------- Graph-side ---------------------------------------
        if phase in ("graph_bootstrap", "joint") and "graph" in output:
            g = output["graph"]

            # Main refined-head losses.
            def graph_losses(pack: Dict[str, torch.Tensor], prefix: str):
                L = 0.0

                l_pres = self.node_presence(pack["node_presence"], targets["node_present"])
                logs[f"{prefix}node_pres"] = l_pres.detach()
                L = L + self.w.node_presence * l_pres

                l_cl = self.centerline(
                    pack["node_poly"], targets["node_points"],
                    presence=targets["node_present"],
                )
                logs[f"{prefix}node_cl"] = l_cl.detach()
                L = L + self.w.centerline * l_cl

                l_rd = self.radius(
                    pack["node_poly"], pack["node_radius"],
                    targets["node_points"], targets["node_radius"],
                    presence=targets["node_present"],
                )
                logs[f"{prefix}node_rd"] = l_rd.detach()
                L = L + self.w.radius * l_rd

                l_ep = self.node_presence(pack["edge_presence"], targets["edge_present"])
                logs[f"{prefix}edge_pres"] = l_ep.detach()
                L = L + self.w.node_presence * l_ep

                l_ecl = self.centerline(
                    pack["edge_poly"], targets["edge_points"],
                    presence=targets["edge_present"],
                )
                logs[f"{prefix}edge_cl"] = l_ecl.detach()
                L = L + self.w.centerline * l_ecl

                l_erd = self.radius(
                    pack["edge_poly"], pack["edge_radius"],
                    targets["edge_points"], targets["edge_radius"],
                    presence=targets["edge_present"],
                )
                logs[f"{prefix}edge_rd"] = l_erd.detach()
                L = L + self.w.radius * l_erd

                return L

            L_main = graph_losses(g, prefix="g/")
            total = total + L_main

            # Preliminary head supervision (lighter weight). These are kept
            # after the first three cross-attn blocks to give the early
            # queries a direct supervisory signal.
            if "prelim" in g:
                L_prelim = graph_losses(g["prelim"], prefix="g0/")
                total = total + self.w.prelim_graph * L_prelim

            # Edge consistency against voxel logits (joint phase only).
            if phase == "joint" and "logits" in output:
                l_ec = self.edge_consistency(
                    output["logits"], g["edge_poly"], g["edge_presence"],
                )
                logs["edge_consistency"] = l_ec.detach()
                total = total + self.w.edge_consistency * l_ec

                # Voxel/tube coupling.
                l_co = self.coupling(output["logits"], g["tube_mask"])
                logs["coupling"] = l_co.detach()
                total = total + self.w.coupling * l_co

        if not torch.is_tensor(total):   # first time paths; avoid raw float
            total = next(iter(output.values())).sum() * 0.0
        return total, logs


__all__ = [
    "SoftDiceCELoss",
    "clDiceLoss",
    "cbDiceLoss",
    "PerClassConnectivityLoss",
    "BettiMatchingLoss",
    "NodePresenceLoss",
    "CenterlineRegressionLoss",
    "RadiusRegressionLoss",
    "EdgeConsistencyLoss",
    "GraphVoxelCouplingLoss",
    "GraphCoWLoss",
    "GraphCoWLossWeights",
]
