"""
models/vae/vae.py
-----------------
Full KL-regularized Variational Autoencoder.
Combines Encoder + Decoder with:
  - Reparameterization trick for differentiable sampling
  - KL divergence loss term
  - Latent scaling for diffusion model compatibility

The latent space of this VAE is the "pixel space" for the diffusion model.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .encoder import Encoder
from .decoder import Decoder
from core.config import ConfigNode


class DiagonalGaussian:
    """
    Diagonal Gaussian distribution in latent space.
    Wraps mean + log_var tensors with sampling and KL computation.
    """

    def __init__(self, mean: torch.Tensor, log_var: torch.Tensor):
        self.mean = mean
        self.log_var = log_var
        self.std = torch.exp(0.5 * log_var)
        self.var = torch.exp(log_var)

    def sample(self) -> torch.Tensor:
        """Reparameterization trick: z = mean + std * eps, eps ~ N(0,I)"""
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        """Deterministic sample (for reconstruction, not for generation)."""
        return self.mean

    def kl_loss(self) -> torch.Tensor:
        """
        KL divergence from N(mean, var) to N(0, I).
        Formula: 0.5 * sum(mean^2 + var - log(var) - 1)
        Returns mean over batch and spatial dims.
        """
        return 0.5 * torch.mean(
            self.mean.pow(2) + self.var - self.log_var - 1.0
        )


class VAE(nn.Module):
    """
    Full VAE for latent diffusion.

    Typical usage during DIFFUSION TRAINING:
      z = vae.encode(image)    # latent (scaled)
      loss = diffusion(z, ...)

    Typical usage during INFERENCE:
      z = diffusion_sample(...)
      image = vae.decode(z)    # back to pixel space
    """

    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            latent_channels=cfg.latent_channels,
            base_channels=cfg.base_channels,
            channel_multipliers=cfg.channel_multipliers,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            dropout=cfg.dropout,
        )

        self.decoder = Decoder(
            out_channels=cfg.in_channels,
            latent_channels=cfg.latent_channels,
            base_channels=cfg.base_channels,
            channel_multipliers=cfg.channel_multipliers,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            dropout=cfg.dropout,
        )

        # Latent scaling factor (empirically chosen, matches SD convention)
        # Multiplying by this makes latent variance ~1, stabilizing diffusion training
        self.scale_factor = 0.18215

    def encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            x:      (B, C, H, W) normalized image in [-1, 1]
            sample: If True, sample from distribution. If False, use mean.

        Returns:
            z: (B, latent_channels, h, w) scaled latent
        """
        mean, log_var = self.encoder(x)
        dist = DiagonalGaussian(mean, log_var)
        z = dist.sample() if sample else dist.mode()

        # Scale latent to unit variance for diffusion
        return z * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel space.

        Args:
            z: (B, latent_channels, h, w) scaled latent

        Returns:
            x_recon: (B, C, H, W) reconstructed image in [-1, 1]
        """
        # Undo scale before decoding
        z = z / self.scale_factor
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass for training.

        Args:
            x: (B, C, H, W) input image

        Returns:
            x_recon: (B, C, H, W) reconstructed image
            kl_loss: scalar KL divergence loss
        """
        mean, log_var = self.encoder(x)
        dist = DiagonalGaussian(mean, log_var)
        z = dist.sample()

        x_recon = self.decoder(z)
        kl_loss = dist.kl_loss()

        return x_recon, kl_loss

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Utility for batch encoding during diffusion training (no grad needed).
        Returns scaled latents ready for noise addition.
        """
        return self.encode(images, sample=True)
