"""
models/diffusion/diffusion.py
------------------------------
LatentDiffusion — top-level model class wrapping all components.

NEW in Genesis v0.2: This is the missing glue layer that was listed
in the roadmap. It unifies:
  - VAE (encoder + decoder)
  - UNet (denoiser)
  - TextEncoder (conditioning)
  - NoiseScheduler (forward/reverse process)

into a single, serializable model object.

Usage:
    # Training
    model = LatentDiffusion(cfg)
    loss = model.training_step(images, captions)

    # Inference
    images = model.sample(prompts, num_steps=50, guidance_scale=7.5)

    # Save / Load
    model.save("checkpoints/epoch_10")
    model = LatentDiffusion.load("checkpoints/epoch_10", cfg)
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class LatentDiffusion(nn.Module):
    """
    Unified Latent Diffusion Model.

    Architecture:
      image → VAE.encode → z_0
      [z_0 + noise t] + text_embeddings → UNet → predicted_noise
      predicted_noise → scheduler.reverse → z_0_clean → VAE.decode → image

    The VAE and TextEncoder are typically frozen during diffusion training.
    Only the UNet is trained.
    """

    def __init__(
        self,
        cfg,
        vae: nn.Module,
        unet: nn.Module,
        text_encoder: nn.Module,
        scheduler,
    ):
        super().__init__()
        self.cfg = cfg
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.scheduler = scheduler

        # Freeze VAE and text encoder by default
        self._freeze_component(self.vae)
        self._freeze_component(self.text_encoder)

        logger.info(
            f"LatentDiffusion initialized | "
            f"UNet params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}"
        )

    def _freeze_component(self, module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    # ── Core forward (training) ────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        captions: List[str],
        cfg_dropout_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Full training forward pass.

        Args:
            images:           (B, 3, H, W) normalized images in [-1, 1]
            captions:         List of B text captions
            cfg_dropout_prob: Fraction of captions to replace with "" for CFG

        Returns:
            loss: Scalar MSE denoising loss
        """
        B = images.shape[0]
        device = images.device

        # 1. Encode to latent space (no grad — VAE frozen)
        with torch.no_grad():
            z0 = self.vae.encode(images)                # (B, 4, h, w)

        # 2. Encode text (no grad — text encoder frozen)
        # Apply CFG dropout: randomly replace captions with empty string
        if cfg_dropout_prob > 0:
            mask = torch.rand(B) < cfg_dropout_prob
            captions = [
                "" if mask[i].item() else cap
                for i, cap in enumerate(captions)
            ]

        with torch.no_grad():
            context = self.text_encoder(captions)       # (B, L, D)

        # 3. Sample random timesteps and noise
        t = torch.randint(0, self.scheduler.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z0)

        # 4. Forward diffusion: add noise to latents
        z_t = self.scheduler.add_noise(z0, noise, t)

        # 5. Predict noise with U-Net
        noise_pred = self.unet(z_t, t, context=context)

        # 6. Compute target and loss
        target = self.scheduler.get_target(noise, z0, t)
        loss = F.mse_loss(noise_pred, target)

        return loss

    # ── Sampling (inference) ───────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        eta: float = 0.0,
        seed: Optional[int] = None,
        use_ddim: bool = True,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompts:           List of text prompts
            negative_prompts:  List of negative prompts (or None)
            num_steps:         Denoising steps
            guidance_scale:    CFG scale (1.0 = no guidance)
            height, width:     Output resolution
            eta:               DDIM stochasticity (0=deterministic, 1=DDPM)
            seed:              Reproducibility seed
            use_ddim:          DDIM (fast) vs DDPM (stochastic)

        Returns:
            List of PIL Images
        """
        device = next(self.unet.parameters()).device
        B = len(prompts)

        if seed is not None:
            torch.manual_seed(seed)

        # Encode text conditioning
        neg = negative_prompts or [""] * B
        context_cond   = self.text_encoder(prompts)      # (B, L, D)
        context_uncond = self.text_encoder(neg)           # (B, L, D)

        # Combined batch for CFG: [uncond; cond]
        context = torch.cat([context_uncond, context_cond], dim=0)  # (2B, L, D)

        # Start from pure Gaussian noise in latent space
        latent_h = height // 8
        latent_w = width // 8
        z = torch.randn(B, 4, latent_h, latent_w, device=device)

        # Build timestep schedule
        if use_ddim:
            timesteps = self.scheduler.make_ddim_timesteps(num_steps)
        else:
            timesteps = list(reversed(range(0, self.scheduler.num_timesteps)))[:num_steps]

        # Denoising loop
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Double batch for CFG
            z_double = torch.cat([z, z], dim=0)
            t_double = torch.cat([t, t], dim=0)

            # U-Net forward (uncond and cond in single pass)
            noise_pred_double = self.unet(z_double, t_double, context=context)
            noise_uncond, noise_cond = noise_pred_double.chunk(2, dim=0)

            # Classifier-free guidance
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Reverse step
            if use_ddim:
                z = self.scheduler.ddim_step(z, noise_pred, t_val, eta=eta)
            else:
                z = self.scheduler.ddpm_step(z, noise_pred, t_val)

            if i % 10 == 0:
                logger.debug(f"  Denoising step {i+1}/{len(timesteps)}")

        # Decode latents to images
        images_tensor = self.vae.decode(z)             # (B, 3, H, W) in [-1, 1]
        images_tensor = (images_tensor.clamp(-1, 1) + 1) / 2  # [0, 1]
        images_tensor = (images_tensor * 255).byte().cpu()

        pil_images = []
        for img_t in images_tensor:
            arr = img_t.permute(1, 2, 0).numpy()
            pil_images.append(Image.fromarray(arr))

        return pil_images

    @torch.no_grad()
    def sample_img2img(
        self,
        prompts: List[str],
        init_images: List[Image.Image],
        strength: float = 0.8,
        negative_prompts: Optional[List[str]] = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Image-to-image generation via partial noising + denoising."""
        import torchvision.transforms.functional as TF
        device = next(self.unet.parameters()).device
        B = len(prompts)

        if seed is not None:
            torch.manual_seed(seed)

        # Encode init images to latent space
        init_tensors = []
        for img in init_images:
            t = TF.to_tensor(img).unsqueeze(0).to(device)  # [0,1]
            t = t * 2 - 1  # [-1,1]
            init_tensors.append(t)
        init_batch = torch.cat(init_tensors, dim=0)
        z0 = self.vae.encode(init_batch)

        # Compute start timestep from strength
        start_step = int(self.scheduler.num_timesteps * (1 - strength))
        timesteps = self.scheduler.make_ddim_timesteps(num_steps)
        timesteps = [t for t in timesteps if t >= start_step]

        # Add noise at the start timestep
        noise = torch.randn_like(z0)
        t_start = torch.full((B,), timesteps[0], device=device, dtype=torch.long)
        z = self.scheduler.add_noise(z0, noise, t_start)

        # Encode text
        neg = negative_prompts or [""] * B
        context_cond   = self.text_encoder(prompts)
        context_uncond = self.text_encoder(neg)
        context = torch.cat([context_uncond, context_cond], dim=0)

        # Denoising loop (partial — from start_step)
        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            z_double = torch.cat([z, z], dim=0)
            t_double = torch.cat([t, t], dim=0)
            noise_pred_double = self.unet(z_double, t_double, context=context)
            noise_uncond, noise_cond = noise_pred_double.chunk(2, dim=0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            z = self.scheduler.ddim_step(z, noise_pred, t_val)

        images_tensor = self.vae.decode(z)
        images_tensor = (images_tensor.clamp(-1, 1) + 1) / 2
        images_tensor = (images_tensor * 255).byte().cpu()

        return [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in images_tensor]

    # ── Serialization ──────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save UNet weights (VAE and text encoder are pretrained, not saved)."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.unet.state_dict(), os.path.join(directory, "unet.pt"))
        logger.info(f"Saved UNet weights → {directory}")

    @classmethod
    def load_unet_weights(cls, model: "LatentDiffusion", directory: str) -> None:
        """Load UNet weights into an existing LatentDiffusion instance."""
        path = os.path.join(directory, "unet.pt")
        state = torch.load(path, map_location="cpu")
        model.unet.load_state_dict(state)
        logger.info(f"Loaded UNet weights ← {directory}")

    # ── Unfreeze helpers ────────────────────────────────────────

    def unfreeze_vae(self) -> None:
        """Allow VAE fine-tuning (e.g. during VAE training phase)."""
        for p in self.vae.parameters():
            p.requires_grad_(True)
        self.vae.train()

    def freeze_vae(self) -> None:
        self._freeze_component(self.vae)

    def unfreeze_text_encoder(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad_(True)
        self.text_encoder.train()

    def trainable_params(self) -> list:
        """Return only the trainable (UNet) parameters for the optimizer."""
        return [p for p in self.unet.parameters() if p.requires_grad]
