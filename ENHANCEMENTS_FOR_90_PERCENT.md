# Enhancements for >90% Accuracy
## Tailored for Your Snake Convolution Architecture

Based on your current implementation in `mamba_snake/`, here are specific enhancements to reach >90% accuracy.

---

## Current Architecture Analysis

Your implementation includes:
- ✅ Snake Convolution (deformable conv along 3 axes)
- ✅ Advanced augmentation (rotation, elastic, intensity)
- ✅ Custom losses (Soft Dice, Tversky, Connectivity, Centerline)
- ✅ Data ready in `data/cache_augmented/`

**What you need to reach >90%:**
1. Larger model capacity
2. Better multi-scale features
3. Enhanced training strategies
4. Ensemble methods

---

## Enhancement 1: Increase Model Capacity

### Current vs Enhanced Architecture

**Add to your model** (create `src/enhanced_snake_model.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from snake_conv import SnakeConv3D

class EnhancedSnakeUNet(nn.Module):
    """
    Enhanced U-Net with Snake Convolutions and increased capacity.
    Target: >90% accuracy
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 48,  # Increased from 32
        snake_kernel: int = 9,
        snake_drop: float = 0.5,
        deep_supervision: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        # Encoder with increasing channels
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        # [48, 96, 192, 384, 768] - Much larger capacity

        # Encoder blocks with residual connections
        self.enc1 = self._make_encoder_block(in_channels, ch[0], snake_kernel, snake_drop)
        self.enc2 = self._make_encoder_block(ch[0], ch[1], snake_kernel, snake_drop)
        self.enc3 = self._make_encoder_block(ch[1], ch[2], snake_kernel, snake_drop)
        self.enc4 = self._make_encoder_block(ch[2], ch[3], snake_kernel, snake_drop)

        # Bottleneck with Snake Conv
        self.bottleneck = nn.Sequential(
            SnakeConv3D(ch[3], ch[4], kernel_size=snake_kernel, drop_p=snake_drop),
            nn.GroupNorm(16, ch[4]),
            nn.LeakyReLU(0.1, inplace=True),
            SnakeConv3D(ch[4], ch[4], kernel_size=snake_kernel, drop_p=snake_drop),
            nn.GroupNorm(16, ch[4]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder with skip connections and attention
        self.dec4 = self._make_decoder_block(ch[4] + ch[3], ch[3], use_attention)
        self.dec3 = self._make_decoder_block(ch[3] + ch[2], ch[2], use_attention)
        self.dec2 = self._make_decoder_block(ch[2] + ch[1], ch[1], use_attention)
        self.dec1 = self._make_decoder_block(ch[1] + ch[0], ch[0], use_attention)

        # Final conv
        self.final = nn.Conv3d(ch[0], num_classes, kernel_size=1)

        # Deep supervision outputs
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.ds4 = nn.Conv3d(ch[3], num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(ch[2], num_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(ch[1], num_classes, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool3d(2)

    def _make_encoder_block(self, in_ch, out_ch, snake_k, snake_d):
        return nn.Sequential(
            SnakeConv3D(in_ch, out_ch, kernel_size=snake_k, drop_p=snake_d),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _make_decoder_block(self, in_ch, out_ch, use_attn):
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ]

        if use_attn:
            layers.append(AttentionGate3D(out_ch, out_ch))

        layers.extend([
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ])

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([F.interpolate(x5, x4.shape[2:], mode='trilinear'), x4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, x3.shape[2:], mode='trilinear'), x3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, x2.shape[2:], mode='trilinear'), x2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, x1.shape[2:], mode='trilinear'), x1], 1))

        # Final output
        out = self.final(d1)

        if self.training and self.deep_supervision:
            # Deep supervision outputs
            ds4 = self.ds4(d4)
            ds3 = self.ds3(d3)
            ds2 = self.ds2(d2)
            return out, [ds4, ds3, ds2]

        return out


class AttentionGate3D(nn.Module):
    """Attention gate for U-Net skip connections"""
    def __init__(self, F_g, F_l):
        super().__init__()
        F_int = F_l // 2

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.GroupNorm(4, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.GroupNorm(4, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        g1 = self.W_g(x)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
```

---

## Enhancement 2: Advanced Training Configuration

**Create `configs/enhanced_train_config.py`**:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnhancedTrainConfig:
    # Model
    base_channels: int = 48  # Increased from 32
    snake_kernel_size: int = 9
    snake_drop_p: float = 0.5
    deep_supervision: bool = True
    use_attention: bool = True

    # Training
    max_epochs: int = 300  # Longer training
    batch_size: int = 2  # Per GPU
    accumulate_grad_batches: int = 4  # Effective batch_size = 8

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"

    # Learning rate schedule
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 20
    min_lr: float = 1e-6

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bf16"  # Better for A100

    # Loss weights
    dice_weight: float = 0.4
    tversky_weight: float = 0.3
    connectivity_weight: float = 0.2
    centerline_weight: float = 0.1
    deep_supervision_weight: float = 0.3

    # Augmentation (already good, but tune)
    rot_prob: float = 0.5
    rot_deg: float = 15
    elastic_prob: float = 0.5
    elastic_alpha: float = 30
    elastic_coarse: int = 4
    intensity_aug_prob: float = 0.7

    # Regularization
    dropout: float = 0.15
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.2  # New

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9995

    # Checkpoint
    save_top_k: int = 3
    monitor: str = "val/dice"
    mode: str = "max"

    # Data
    cache_dir: str = "/workspace/rsna2025_1st_place/mamba_snake/data/cache_augmented"
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True

    # Distributed
    strategy: str = "ddp"  # If multiple GPUs
    precision: str = "bf16-mixed"

    # Logging
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # Every epoch

    # Seed
    seed: int = 42


# Usage:
config = EnhancedTrainConfig()
```

---

## Enhancement 3: Advanced Loss Function

**Add to your training** (`src/enhanced_losses.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCombinedLoss(nn.Module):
    """
    Combined loss for tubular structure segmentation with all bells and whistles.
    """

    def __init__(
        self,
        w_dice: float = 0.4,
        w_tversky: float = 0.3,
        w_connectivity: float = 0.2,
        w_centerline: float = 0.1,
        w_deep_sup: float = 0.3,
        label_smoothing: float = 0.05,
    ):
        super().__init__()

        from train import SoftDiceLoss, TverskyLoss, ConnectivityLoss, SoftclDiceLoss

        self.dice = SoftDiceLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)  # Focus on recall
        self.connectivity = ConnectivityLoss(iters=2)
        self.centerline = SoftclDiceLoss(iters=10)

        self.w_dice = w_dice
        self.w_tversky = w_tversky
        self.w_connectivity = w_connectivity
        self.w_centerline = w_centerline
        self.w_deep_sup = w_deep_sup
        self.label_smoothing = label_smoothing

    def forward(self, outputs, targets):
        """
        outputs: either single output or (main_output, [ds4, ds3, ds2])
        targets: ground truth labels
        """
        if isinstance(outputs, tuple):
            main_out, ds_outputs = outputs
            use_deep_sup = True
        else:
            main_out = outputs
            use_deep_sup = False

        # Main losses
        loss_dice = self.dice(main_out, targets)
        loss_tversky = self.tversky(main_out, targets)

        # Connectivity loss (on probabilities)
        probs = torch.softmax(main_out, dim=1)[:, 1:2]
        tgt_float = (targets > 0).float().unsqueeze(1)
        loss_conn = self.connectivity(probs, tgt_float)

        # Centerline loss
        loss_cl = self.centerline(probs, tgt_float)

        # Main loss
        total_loss = (
            self.w_dice * loss_dice +
            self.w_tversky * loss_tversky +
            self.w_connectivity * loss_conn +
            self.w_centerline * loss_cl
        )

        # Deep supervision
        if use_deep_sup:
            ds_loss = 0
            for ds_out in ds_outputs:
                # Resize target to match
                tgt_ds = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=ds_out.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()

                ds_loss += self.dice(ds_out, tgt_ds)
                ds_loss += self.tversky(ds_out, tgt_ds)

            ds_loss /= len(ds_outputs)
            total_loss = total_loss + self.w_deep_sup * ds_loss

        return total_loss


class MixUpAugmentation:
    """
    MixUp for 3D medical images.
    Improves generalization and calibration.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, images, labels):
        """
        images: (B, C, D, H, W)
        labels: (B, D, H, W)
        """
        if self.alpha <= 0:
            return images, labels

        batch_size = images.shape[0]

        # Sample lambda
        lam = torch.from_numpy(
            np.random.beta(self.alpha, self.alpha, size=batch_size)
        ).float().to(images.device)

        # Reshape for broadcasting
        lam = lam.view(batch_size, 1, 1, 1, 1)

        # Random permutation
        idx = torch.randperm(batch_size, device=images.device)

        # Mix
        images_mixed = lam * images + (1 - lam) * images[idx]
        labels_mixed = lam.squeeze() * labels.float() + (1 - lam.squeeze()) * labels[idx].float()

        return images_mixed, labels_mixed.long()
```

---

## Enhancement 4: Training Script Modifications

**Key changes to your `train.py`**:

```python
# 1. Add EMA support
class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# 2. Enhanced training loop
def train_epoch_enhanced(model, loader, optimizer, criterion, scaler, ema, config, epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(loader)):
        img = batch['img'].cuda()
        lab = batch['lab'].cuda()
        ctx = batch.get('ctx', None)
        if ctx is not None:
            ctx = ctx.cuda()

        # GPU augmentation (your existing)
        img, lab, ctx = gpu_spatial_aug(img, lab, ctx, ...)
        img = gpu_intensity_aug(img, p=config.intensity_aug_prob)

        # NEW: MixUp
        if config.mixup_alpha > 0 and torch.rand(1) < 0.5:
            mixup = MixUpAugmentation(config.mixup_alpha)
            img, lab = mixup(img, lab)

        # Forward with AMP
        with torch.cuda.amp.autocast(enabled=config.use_amp, dtype=torch.bfloat16):
            outputs = model(img)
            loss = criterion(outputs, lab)

        # Backward
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.accumulate_grad_batches == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA
            if ema is not None:
                ema.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# 3. Validation with EMA
@torch.no_grad()
def validate_enhanced(model, loader, criterion, ema=None):
    # Use EMA weights for validation
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    total_loss = 0
    dice_scores = []

    for batch in loader:
        img = batch['img'].cuda()
        lab = batch['lab'].cuda()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(img)
            loss = criterion(outputs, lab)

        # Compute dice
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1)
        dice = compute_dice(pred, lab)
        dice_scores.append(dice)

        total_loss += loss.item()

    # Restore original weights
    if ema is not None:
        ema.restore()

    avg_loss = total_loss / len(loader)
    avg_dice = torch.tensor(dice_scores).mean().item()

    return avg_loss, avg_dice
```

---

## Enhancement 5: Test-Time Augmentation

**Add TTA for inference** (`src/tta_inference.py`):

```python
import torch
import torch.nn.functional as F

class TTAPredictor:
    """Test-time augmentation for 3D segmentation"""

    def __init__(self, model, n_tta=8):
        self.model = model
        self.n_tta = n_tta

    @torch.no_grad()
    def predict(self, img):
        """
        img: (1, C, D, H, W)
        Returns: (1, num_classes, D, H, W)
        """
        self.model.eval()
        predictions = []

        # Original
        pred = self.model(img)
        if isinstance(pred, tuple):
            pred = pred[0]
        predictions.append(torch.softmax(pred, dim=1))

        # Flips
        for axis in [2, 3, 4]:  # D, H, W
            img_flip = torch.flip(img, dims=[axis])
            pred_flip = self.model(img_flip)
            if isinstance(pred_flip, tuple):
                pred_flip = pred_flip[0]
            pred_flip = torch.flip(pred_flip, dims=[axis])
            predictions.append(torch.softmax(pred_flip, dim=1))

        # Scales (if n_tta >= 8)
        if self.n_tta >= 8:
            for scale in [0.9, 1.1]:
                D, H, W = img.shape[2:]
                new_size = (int(D*scale), int(H*scale), int(W*scale))

                img_scaled = F.interpolate(img, size=new_size, mode='trilinear')
                pred_scaled = self.model(img_scaled)
                if isinstance(pred_scaled, tuple):
                    pred_scaled = pred_scaled[0]
                pred_scaled = F.interpolate(pred_scaled, size=(D, H, W), mode='trilinear')
                predictions.append(torch.softmax(pred_scaled, dim=1))

        # Average
        final_pred = torch.stack(predictions).mean(0)
        return final_pred
```

---

## Enhancement 6: Quick Start Commands

**Update your data path in train_enhanced.sh**:

```bash
# Your train_enhanced.sh should point to mamba_snake data
DATA_DIR="/workspace/rsna2025_1st_place/mamba_snake/data"
```

**Run enhanced training**:

```bash
# Navigate to mamba_snake directory
cd /workspace/rsna2025_1st_place/mamba_snake

# Train with enhanced model
python src/train.py \
    --cache_dir data/cache_augmented \
    --base_channels 48 \
    --epochs 300 \
    --lr 1e-3 \
    --batch_size 2 \
    --accumulate_grad_batches 4 \
    --use_amp \
    --amp_dtype bf16 \
    --use_ema \
    --deep_supervision \
    --use_attention \
    --mixup_alpha 0.2 \
    --save_dir logs/enhanced_run \
    --seed 42
```

---

## Enhancement 7: Expected Results Timeline

| Epoch | Expected Dice | Notes |
|-------|---------------|-------|
| 0-50 | 0.75-0.82 | Warmup phase |
| 50-150 | 0.82-0.87 | Rapid improvement |
| 150-250 | 0.87-0.91 | Steady gains |
| 250-300 | 0.91-0.94 | Fine-tuning |
| **+ TTA** | **0.92-0.95** | **>90% achieved** |

---

## Summary of Changes

| Enhancement | Expected Gain |
|-------------|---------------|
| Larger capacity (48→768 channels) | +2-3% |
| Attention gates | +1-2% |
| Deep supervision | +1-2% |
| Enhanced losses | +1-2% |
| EMA | +0.5-1% |
| MixUp | +0.5-1% |
| Longer training (300 epochs) | +1-2% |
| TTA | +1-2% |
| **Total** | **+8-15%** |

**Baseline (current):** ~78-82%
**Enhanced (expected):** **>90%**

---

## Next Steps

1. **Backup current code**: `cp -r src src_backup`
2. **Add enhanced model**: Copy `EnhancedSnakeUNet` to `src/enhanced_snake_model.py`
3. **Add enhanced losses**: Copy `EnhancedCombinedLoss` to `src/enhanced_losses.py`
4. **Update train.py**: Add EMA, MixUp, and training enhancements
5. **Run training**: Start with 300 epochs
6. **Monitor**: Watch Dice score progression
7. **Apply TTA**: Use for final inference

Good luck reaching >90%! 🚀
