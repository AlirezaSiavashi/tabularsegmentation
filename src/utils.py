import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    labels: (B, 1, D, H, W) integer
    returns: (B, C, D, H, W)
    """
    if labels.ndim != 5:
        raise ValueError(f"labels must be 5D, got {labels.shape}")
    b = labels.shape[0]
    out = torch.zeros((b, num_classes, *labels.shape[2:]), device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels.long().clamp_min(0).clamp_max(num_classes - 1), 1.0)
    return out

@torch.no_grad()
def dice_per_class(prob: torch.Tensor, gt_1h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    prob: (B,C,...) softmax probabilities
    gt_1h: (B,C,...) one-hot
    returns: (C,) dice averaged over batch
    """
    dims = tuple(range(2, prob.ndim))
    inter = (prob * gt_1h).sum(dim=dims)
    den = prob.sum(dim=dims) + gt_1h.sum(dim=dims)
    dice = (2 * inter + eps) / (den + eps)
    return dice.mean(dim=0)
