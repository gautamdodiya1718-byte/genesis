"""
training/losses.py
-------------------
Loss functions for VAE and diffusion training.

NEW in Genesis v0.2:
  - PerceptualLoss: VGG-16 feature-level LPIPS-style loss.
    Critical for sharp VAE reconstructions — without it, VAE outputs
    are blurry because pixel MSE/L1 penalizes all errors equally
    regardless of perceptual importance.
  - SSIMLoss: Structural similarity index loss.
  - ReconstructionLoss: Combined configurable reconstruction loss.
  - AdversarialLoss + PatchDiscriminator: optional GAN loss for VAE.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────────────────────
# Perceptual Loss (LPIPS-style using VGG-16)
# ─────────────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """
    Extracts multi-scale features from a pretrained VGG-16 network.
    These features encode perceptual similarity — matching them
    produces sharper, more realistic-looking reconstructions.
    """

    VGG_LAYERS = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 16,
        "relu4_3": 23,
    }

    def __init__(self, layers: List[str] = None):
        super().__init__()
        import torchvision.models as M

        self.layers_to_use = layers or list(self.VGG_LAYERS.keys())
        max_layer = max(self.VGG_LAYERS[l] for l in self.layers_to_use)

        vgg = M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer + 1])

        # Freeze VGG permanently
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features at multiple scales.

        Args:
            x: (B, 3, H, W) images in [-1, 1]

        Returns:
            List of feature tensors at each selected layer
        """
        # Normalize from [-1,1] to VGG ImageNet range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1) / 2  # [-1,1] → [0,1]
        x = (x - mean) / std

        feats = []
        target_indices = set(self.VGG_LAYERS[l] for l in self.layers_to_use)

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in target_indices:
                feats.append(x)

        return feats


class PerceptualLoss(nn.Module):
    """
    LPIPS-style perceptual loss using VGG-16 features.

    Compares reconstructed and target images at multiple feature scales.
    Multi-scale comparison penalizes both fine texture (early layers)
    and semantic structure (later layers).

    Usage:
        loss_fn = PerceptualLoss()
        loss = loss_fn(reconstructed, target)   # both in [-1, 1]
    """

    def __init__(self, weights: List[float] = None):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        # Weight each layer's contribution
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) reconstructed images in [-1, 1]
            target: (B, 3, H, W) ground truth images in [-1, 1]

        Returns:
            Scalar perceptual loss
        """
        pred_feats   = self.vgg(pred)
        target_feats = self.vgg(target)

        loss = torch.tensor(0.0, device=pred.device)
        for w, pf, tf in zip(self.weights, pred_feats, target_feats):
            loss = loss + w * F.l1_loss(pf, tf)

        return loss


# ─────────────────────────────────────────────────────────────
# SSIM Loss
# ─────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss.
    Better than L1/L2 for measuring perceptual reconstruction quality.
    Considers luminance, contrast, and structure simultaneously.
    """

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._gaussian_window(window_size, channels))

    def _gaussian_window(self, size: int, channels: int) -> torch.Tensor:
        sigma = 1.5
        coords = torch.arange(size).float() - size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel.repeat(channels, 1, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        pred   = (pred   + 1) / 2
        target = (target + 1) / 2

        pad = self.window_size // 2
        mu1 = F.conv2d(pred,   self.window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channels)

        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sig1_sq = F.conv2d(pred   * pred,   self.window, padding=pad, groups=self.channels) - mu1_sq
        sig2_sq = F.conv2d(target * target, self.window, padding=pad, groups=self.channels) - mu2_sq
        sig12   = F.conv2d(pred   * target, self.window, padding=pad, groups=self.channels) - mu12

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))

        return 1.0 - ssim.mean()


# ─────────────────────────────────────────────────────────────
# Combined Reconstruction Loss
# ─────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """
    Configurable reconstruction loss combining:
      - L1 or L2 pixel loss
      - Perceptual loss (VGG features)
      - SSIM loss
      - KL divergence

    Used as the primary VAE training objective.
    """

    def __init__(
        self,
        recon_type: str = "l1",          # "l1" | "l2"
        use_perceptual: bool = True,
        use_ssim: bool = False,
        perceptual_weight: float = 1.0,
        ssim_weight: float = 0.0,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        self.recon_type = recon_type
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.kl_weight = kl_weight

        self.perceptual_loss = PerceptualLoss() if use_perceptual else None
        self.ssim_loss = SSIMLoss() if use_ssim else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kl: torch.Tensor,
    ) -> tuple:
        """
        Args:
            pred:   (B, 3, H, W) reconstructed images in [-1, 1]
            target: (B, 3, H, W) ground truth images in [-1, 1]
            kl:     Scalar KL divergence

        Returns:
            total_loss, loss_dict (for logging)
        """
        # Pixel reconstruction loss
        if self.recon_type == "l1":
            recon = F.l1_loss(pred, target)
        else:
            recon = F.mse_loss(pred, target)

        losses = {"recon": recon.item(), "kl": kl.item()}
        total = recon + self.kl_weight * kl

        # Perceptual loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perc = self.perceptual_loss(pred, target)
            total = total + self.perceptual_weight * perc
            losses["perceptual"] = perc.item()

        # SSIM loss
        if self.ssim_loss is not None and self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target)
            total = total + self.ssim_weight * ssim
            losses["ssim"] = ssim.item()

        losses["total"] = total.item()
        return total, losses


# ─────────────────────────────────────────────────────────────
# PatchGAN Discriminator (optional adversarial loss for VAE)
# ─────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator for adversarial VAE training.
    Classifies overlapping image patches as real/fake.
    Improves texture sharpness significantly when used with the VAE.

    Enable by setting use_adversarial_loss: true in config.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, norm=False),   # 256→128
            *block(base_channels, base_channels * 2),          # 128→64
            *block(base_channels * 2, base_channels * 4),      # 64→32
            *block(base_channels * 4, base_channels * 8, stride=1),  # 32→32
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),          # patch output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdversarialLoss(nn.Module):
    """
    Hinge adversarial loss for PatchGAN.
    Generator loss + discriminator loss computed separately.
    """

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (real_loss + fake_loss)
