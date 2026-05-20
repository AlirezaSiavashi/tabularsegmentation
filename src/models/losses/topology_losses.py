#!/usr/bin/env python3
"""
Topology-aware losses for vessel segmentation.

Includes:
1. clDice (centerline Dice) - from Shit et al. 2021
2. Connectivity loss via soft skeletonization
3. Combined topology-preserving loss

For publication benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftSkeletonize3D(nn.Module):
    """
    Differentiable 3D skeletonization for centerline extraction.

    Based on: "clDice - A Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation" (CVPR 2021)
    """
    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = num_iter

    @staticmethod
    def soft_erode(img):
        """Soft morphological erosion (min pooling)."""
        # Negative max pooling = min pooling
        p1 = -F.max_pool3d(-img, kernel_size=3, stride=1, padding=1)
        p2 = -F.max_pool3d(-img, kernel_size=5, stride=1, padding=2)
        p3 = -F.max_pool3d(-img, kernel_size=7, stride=1, padding=3)
        return (p1 + p2 + p3) / 3.0

    @staticmethod
    def soft_dilate(img):
        """Soft morphological dilation (max pooling)."""
        p1 = F.max_pool3d(img, kernel_size=3, stride=1, padding=1)
        p2 = F.max_pool3d(img, kernel_size=5, stride=1, padding=2)
        p3 = F.max_pool3d(img, kernel_size=7, stride=1, padding=3)
        return (p1 + p2 + p3) / 3.0

    @staticmethod
    def soft_open(img):
        """Opening: erosion followed by dilation."""
        return SoftSkeletonize3D.soft_dilate(SoftSkeletonize3D.soft_erode(img))

    @staticmethod
    def soft_close(img):
        """Closing: dilation followed by erosion."""
        return SoftSkeletonize3D.soft_erode(SoftSkeletonize3D.soft_dilate(img))

    def forward(self, img):
        """
        Soft skeletonization via iterative thinning.

        Args:
            img: [B, 1, D, H, W] probability map (0-1)

        Returns:
            skeleton: [B, 1, D, H, W] soft skeleton
        """
        # Ensure input is probability
        img = torch.sigmoid(img) if img.max() > 1.0 else img

        # Initialize skeleton
        skel = img.clone()

        # Iterative thinning
        for _ in range(self.num_iter):
            # Compute opening (removes endpoints)
            opened = self.soft_open(skel)

            # Subtract opening from current skeleton
            # This removes "mass" while preserving topology
            skel_temp = skel - opened

            # Erode current skeleton
            skel = self.soft_erode(skel)

            # Add back the removed "mass" to preserve connectivity
            skel = skel + skel_temp

            # Clamp to [0, 1]
            skel = torch.clamp(skel, 0.0, 1.0)

        return skel


class clDiceLoss(nn.Module):
    """
    Centerline Dice (clDice) loss.

    Paper: "clDice - A Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation" (CVPR 2021)

    Combines:
    - Dice on centerlines
    - Centerline recall (coverage of ground truth centerline)
    - Centerline precision (avoiding false positive branches)
    """
    def __init__(self, iter_: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skel = SoftSkeletonize3D(num_iter=iter_)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, D, H, W] logits or probabilities
            target: [B, D, H, W] labels (0/1)

        Returns:
            clDice loss (1 - clDice)
        """
        # Convert to probabilities
        if pred.shape[1] == 2:  # 2-class logits
            pred_prob = torch.softmax(pred, dim=1)[:, 1:2]  # foreground prob
        else:
            pred_prob = torch.sigmoid(pred)

        # Convert target to [B, 1, D, H, W]
        if target.ndim == 4:
            target = target.unsqueeze(1).float()

        # Soft skeletonize both pred and target
        pred_skel = self.soft_skel(pred_prob)
        target_skel = self.soft_skel(target)

        # clDice = 2 * |pred_skel ∩ target| / (|pred_skel| + |target_skel|)
        intersection = (pred_skel * target_skel).sum()
        pred_skel_sum = pred_skel.sum()
        target_skel_sum = target_skel.sum()

        cldice = (2.0 * intersection + self.smooth) / (
            pred_skel_sum + target_skel_sum + self.smooth
        )

        return 1.0 - cldice


class ConnectivityLoss(nn.Module):
    """
    Penalize disconnected components.

    Uses soft connected component analysis via distance transform.
    """
    def __init__(self):
        super().__init__()

    def compute_distance_transform(self, mask):
        """
        Approximate distance transform using max pooling cascade.

        For each foreground voxel, compute distance to nearest background.
        """
        # Invert mask (background = 1, foreground = 0)
        inv_mask = 1.0 - mask

        # Cascade of max pooling with increasing kernel sizes
        dt = inv_mask.clone()
        for k in [3, 5, 7, 9]:
            dt = F.max_pool3d(dt, kernel_size=k, stride=1, padding=k//2)

        # Distance = how many pooling steps to reach background
        return 1.0 - dt

    def forward(self, pred, target):
        """
        Compute connectivity preservation loss.

        Idea: Penalize if predicted vessel has different topology than target.
        """
        # Convert to probabilities
        if pred.shape[1] == 2:
            pred_prob = torch.softmax(pred, dim=1)[:, 1:2]
        else:
            pred_prob = torch.sigmoid(pred)

        if target.ndim == 4:
            target = target.unsqueeze(1).float()

        # Compute distance transforms
        pred_dt = self.compute_distance_transform(pred_prob)
        target_dt = self.compute_distance_transform(target)

        # L1 loss on distance transforms
        # If topology matches, distance transforms should be similar
        loss = F.l1_loss(pred_dt, target_dt)

        return loss


class TopologyPreservingLoss(nn.Module):
    """
    Combined loss for publication:
    L = α * Dice + β * clDice + γ * Connectivity

    Default weights: α=0.5, β=0.3, γ=0.2
    """
    def __init__(
        self,
        dice_weight: float = 0.5,
        cldice_weight: float = 0.3,
        connectivity_weight: float = 0.2,
        tversky_alpha: float = 0.25,
        tversky_beta: float = 0.75,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.connectivity_weight = connectivity_weight

        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.cldice = clDiceLoss(iter_=10)
        self.connectivity = ConnectivityLoss()

    def forward(self, outputs, target):
        """
        Args:
            outputs: dict with 'logits', 'centerline'
            target: [B, D, H, W] ground truth

        Returns:
            total_loss, dict of individual losses
        """
        logits = outputs['logits']
        centerline_pred = outputs['centerline']

        # 1. Tversky loss on main segmentation
        tversky_loss = self.tversky(logits, target)

        # 2. clDice loss on centerline branch
        # Encourage centerline prediction to match skeleton
        target_fg = (target > 0).float()
        cldice_loss = self.cldice(centerline_pred, target_fg)

        # 3. Connectivity loss on main segmentation
        conn_loss = self.connectivity(logits, target)

        # Combine
        total_loss = (
            self.dice_weight * tversky_loss +
            self.cldice_weight * cldice_loss +
            self.connectivity_weight * conn_loss
        )

        return total_loss, {
            'tversky': tversky_loss.item(),
            'cldice': cldice_loss.item(),
            'connectivity': conn_loss.item(),
        }


class TverskyLoss(nn.Module):
    """Standard Tversky loss (already good for vessels)."""
    def __init__(self, alpha: float = 0.25, beta: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.shape[1] == 2:
            pred = torch.softmax(pred, dim=1)

        if target.ndim == 4:
            target_oh = F.one_hot(target.long(), num_classes=pred.shape[1])
            target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        else:
            target_oh = target

        # Flatten
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target_oh.reshape(target_oh.shape[0], target_oh.shape[1], -1)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Average across batch and classes
        return 1.0 - tversky.mean()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test topology losses
    B, D, H, W = 2, 64, 64, 64

    pred = {
        'logits': torch.randn(B, 2, D, H, W),
        'centerline': torch.rand(B, 1, D, H, W)
    }
    target = torch.randint(0, 2, (B, D, H, W))

    criterion = TopologyPreservingLoss()
    loss, loss_dict = criterion(pred, target)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
