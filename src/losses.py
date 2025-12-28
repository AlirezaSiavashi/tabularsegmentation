from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import one_hot

def soft_dice_loss(prob: torch.Tensor, gt_1h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(2, prob.ndim))
    inter = (prob * gt_1h).sum(dim=dims)
    den = prob.sum(dim=dims) + gt_1h.sum(dim=dims)
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()

def tversky_loss(prob: torch.Tensor, gt_1h: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
    # alpha: FP weight, beta: FN weight
    dims = tuple(range(2, prob.ndim))
    tp = (prob * gt_1h).sum(dim=dims)
    fp = (prob * (1 - gt_1h)).sum(dim=dims)
    fn = ((1 - prob) * gt_1h).sum(dim=dims)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tv.mean()

def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-x, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
    p2 = -F.max_pool3d(-x, kernel_size=(1,3,1), stride=1, padding=(0,1,0))
    p3 = -F.max_pool3d(-x, kernel_size=(1,1,3), stride=1, padding=(0,0,1))
    return torch.min(torch.min(p1, p2), p3)

def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    p1 = F.max_pool3d(x, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
    p2 = F.max_pool3d(x, kernel_size=(1,3,1), stride=1, padding=(0,1,0))
    p3 = F.max_pool3d(x, kernel_size=(1,1,3), stride=1, padding=(0,0,1))
    return torch.max(torch.max(p1, p2), p3)

def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))

def soft_skel(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    # Soft skeletonization (iterative thinning)
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cldice_loss(prob: torch.Tensor, gt_1h: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute clDice over foreground channels only (skip background).
    prob, gt_1h: (B,C,D,H,W)
    """
    if prob.shape[1] <= 1:
        return torch.tensor(0.0, device=prob.device)

    p = prob[:, 1:, ...]
    g = gt_1h[:, 1:, ...]

    skel_p = soft_skel(p, iters=iters)
    skel_g = soft_skel(g, iters=iters)

    # topology precision/recall
    tprec = ( (skel_p * g).sum() + eps ) / ( skel_p.sum() + eps )
    tsens = ( (skel_g * p).sum() + eps ) / ( skel_g.sum() + eps )
    cl = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cl

def skeleton_recall_loss(prob: torch.Tensor, gt_1h: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """
    Penalize missing skeleton voxels: 1 - recall(skel(gt) covered by pred).
    """
    if prob.shape[1] <= 1:
        return torch.tensor(0.0, device=prob.device)

    p = prob[:, 1:, ...]
    g = gt_1h[:, 1:, ...]
    skel_g = soft_skel(g, iters=iters)
    recall = ((skel_g * p).sum() + eps) / (skel_g.sum() + eps)
    return 1.0 - recall

# Optional TCLoss hook (Persistent Homology + Hausdorff on persistence diagrams)
# The DSCNet paper defines TCLoss as CE + sum_n d*_H(Dgm(O), Dgm(L)) (Eq. 9). 
class OptionalTCLoss(nn.Module):
    def __init__(self, enabled: bool = False, weight: float = 0.0):
        super().__init__()
        self.enabled = enabled
        self.weight = float(weight)

    def forward(self, prob_fg: torch.Tensor, gt_fg: torch.Tensor) -> torch.Tensor:
        """
        prob_fg, gt_fg: (B,1,D,H,W) in [0,1]
        Returns a scalar.
        If you install a PH lib (e.g., gudhi/ripser/torch-topological), implement here.
        """
        if (not self.enabled) or self.weight <= 0:
            return torch.tensor(0.0, device=prob_fg.device)

        # Placeholder: returns 0 so training is still runnable.
        # Implement PD computation + bidirectional Hausdorff distance if you add PH dependency.
        return torch.tensor(0.0, device=prob_fg.device)

class VesselLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_dice: float = 0.5,
        w_tversky: float = 0.5,
        w_cldice: float = 0.3,
        w_skelrec: float = 0.2,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        skel_iters: int = 10,
        tc_enabled: bool = False,
        tc_weight: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_dice = w_dice
        self.w_tversky = w_tversky
        self.w_cldice = w_cldice
        self.w_skelrec = w_skelrec
        self.alpha = tversky_alpha
        self.beta = tversky_beta
        self.skel_iters = skel_iters
        self.tc = OptionalTCLoss(enabled=tc_enabled, weight=tc_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,C,D,H,W)
        labels: (B,1,D,H,W) int
        """
        prob = torch.softmax(logits, dim=1)
        gt_1h = one_hot(labels, self.num_classes)

        ld = soft_dice_loss(prob, gt_1h)
        lt = tversky_loss(prob, gt_1h, alpha=self.alpha, beta=self.beta)
        lc = cldice_loss(prob, gt_1h, iters=self.skel_iters)
        ls = skeleton_recall_loss(prob, gt_1h, iters=self.skel_iters)

        # Optional topology PH term (foreground)
        if self.num_classes > 1:
            pf = prob[:, 1:2, ...]           # first foreground channel
            gf = gt_1h[:, 1:2, ...]
            ltc = self.tc(pf, gf) * self.tc.weight
        else:
            ltc = torch.tensor(0.0, device=logits.device)

        loss = self.w_dice * ld + self.w_tversky * lt + self.w_cldice * lc + self.w_skelrec * ls + ltc
        return loss
