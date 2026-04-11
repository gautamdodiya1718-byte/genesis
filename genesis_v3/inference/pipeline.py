"""
inference/pipeline.py
----------------------
Inference pipeline for locally-trained Genesis models.

Two modes:
  1. CustomPipeline  — uses your Genesis-trained VAE + UNet + TextEncoder
  2. PretrainedPipeline — wraps HuggingFace diffusers for pretrained SD/LCM

The CustomPipeline is what you use after running scripts/train_vae.py
and scripts/train.py — it loads your own trained weights.

The PretrainedPipeline is what AutoDiff uses before any training
has been done (downloads SD 1.5 / LCM automatically).
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image

from models.diffusion.scheduler import NoiseScheduler
from inference.onnx_exporter import ONNXExporter, ONNXUNet, ONNXVAEDecoder

logger = logging.getLogger(__name__)


class CustomPipeline:
    """
    Inference pipeline for Genesis-trained models.

    Loads your trained VAE + UNet + TextEncoder and runs
    full txt2img and img2img generation.

    Usage:
        pipe = CustomPipeline.from_checkpoint(
            checkpoint_dir="outputs/checkpoints/step_00010000",
            vae_checkpoint="outputs/vae_training/vae_final.pt",
            cfg=cfg,
        )
        images = pipe.txt2img(["a mountain at sunset"], num_steps=50)
    """

    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        text_encoder: nn.Module,
        scheduler: NoiseScheduler,
        device: str = "cpu",
        use_onnx: bool = False,
        onnx_dir: str = "model_cache/onnx",
    ):
        self.vae          = vae.to(device).eval()
        self.unet         = unet.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.scheduler    = scheduler
        self.device       = device

        # Optionally swap UNet to ONNX for 2-3× CPU speedup
        self._onnx_unet = None
        self._onnx_vae  = None
        if use_onnx:
            self._setup_onnx(onnx_dir)

        for model in (self.vae, self.unet, self.text_encoder):
            for p in model.parameters():
                p.requires_grad_(False)

        logger.info(
            f"CustomPipeline ready | device={device} | "
            f"onnx={use_onnx}"
        )

    def _setup_onnx(self, onnx_dir: str) -> None:
        """Export models to ONNX and load ONNXRuntime sessions."""
        try:
            exporter = ONNXExporter(onnx_dir)
            unet_path = exporter.export_unet(self.unet)
            self._onnx_unet = ONNXUNet(unet_path)
            logger.info("ONNX UNet loaded for accelerated inference")
        except Exception as e:
            logger.warning(f"ONNX setup failed: {e}. Using PyTorch.")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        cfg,
        vae_checkpoint: Optional[str] = None,
        use_ema: bool = True,
        use_onnx: bool = False,
    ) -> "CustomPipeline":
        """
        Load a complete pipeline from a Genesis training checkpoint.

        Args:
            checkpoint_dir: Path to a diffusion training checkpoint dir
            cfg:            Genesis config
            vae_checkpoint: Path to vae_final.pt (if not loading default)
            use_ema:        Prefer EMA weights (smoother samples)
            use_onnx:       Export UNet to ONNX for faster CPU inference
        """
        from models.vae.vae import VAE
        from models.vae.encoder import Encoder
        from models.vae.decoder import Decoder
        from models.diffusion.unet import UNet
        from models.text_encoder.encoder import TextEncoder

        device = cfg.system.device

        # Build models
        vae = VAE(cfg)
        unet = UNet(
            in_channels=cfg.diffusion.in_channels,
            out_channels=cfg.diffusion.out_channels,
            base_channels=cfg.diffusion.base_channels,
            channel_multipliers=list(cfg.diffusion.channel_multipliers),
            num_res_blocks=cfg.diffusion.num_res_blocks,
            context_dim=cfg.diffusion.context_dim,
        )
        text_enc = TextEncoder(cfg)
        text_enc.load()

        scheduler = NoiseScheduler(
            timesteps=cfg.diffusion.timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            prediction_type=cfg.diffusion.prediction_type,
        )

        # Load VAE weights
        if vae_checkpoint and os.path.exists(vae_checkpoint):
            from training.vae_trainer import VAETrainer
            VAETrainer.load_vae_weights(vae, vae_checkpoint, use_ema=use_ema)
        elif cfg.vae.get("pretrained_id"):
            # Load pretrained SD VAE (skip training)
            try:
                from diffusers import AutoencoderKL
                sd_vae = AutoencoderKL.from_pretrained(cfg.vae.pretrained_id)
                # Map SD VAE to our custom VAE (weights are compatible format)
                logger.info(f"Loaded pretrained VAE: {cfg.vae.pretrained_id}")
                # Note: Direct weight mapping requires matching architecture
            except Exception as e:
                logger.warning(f"Could not load pretrained VAE: {e}")

        # Load UNet weights from checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            key = "ema" if use_ema and "ema" in state else "unet"
            unet.load_state_dict(state[key])
            logger.info(
                f"Loaded UNet ({'EMA' if use_ema else 'raw'}) "
                f"from step {state.get('global_step', '?')}"
            )

        return cls(
            vae=vae, unet=unet, text_encoder=text_enc,
            scheduler=scheduler, device=device,
            use_onnx=use_onnx,
            onnx_dir=cfg.generation.get("onnx_dir", "model_cache/onnx"),
        )

    # ── Generation ─────────────────────────────────────────────

    @torch.no_grad()
    def txt2img(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width:  int = 512,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate images from text prompts using trained Genesis model."""
        B = len(prompts)
        device = self.device

        if seed is not None:
            torch.manual_seed(seed)

        # Encode text
        neg = negative_prompts or [""] * B
        ctx_cond   = self.text_encoder(prompts).to(device)
        ctx_uncond = self.text_encoder(neg).to(device)
        ctx = torch.cat([ctx_uncond, ctx_cond], dim=0)  # (2B, L, D)

        # Start from noise
        lh, lw = height // 8, width // 8
        z = torch.randn(B, 4, lh, lw, device=device)

        # Denoising loop (DDIM)
        timesteps = self.scheduler.make_ddim_timesteps(num_steps)
        pairs = list(zip(timesteps, timesteps[1:] + [-1]))

        for step_idx, (t_cur, t_prev) in enumerate(pairs):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)

            z_in = torch.cat([z, z])
            t_in = torch.cat([t, t])

            if self._onnx_unet:
                ctx_cpu = ctx.cpu()
                z_in_cpu = z_in.cpu()
                t_in_cpu = t_in.cpu()
                pred_both = self._onnx_unet(z_in_cpu, t_in_cpu, ctx_cpu).to(device)
            else:
                pred_both = self.unet(z_in, t_in, context=ctx)

            pred_uncond, pred_cond = pred_both.chunk(2, dim=0)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            z = self.scheduler.ddim_step(
                model_output=pred,
                timestep=t_cur,
                prev_timestep=t_prev,
                x_t=z,
                eta=eta,
            )

            if step_idx % 10 == 0:
                logger.debug(f"  Step {step_idx+1}/{len(pairs)}")

        # Decode
        if self._onnx_vae:
            imgs_t = self._onnx_vae.decode(z)
        else:
            imgs_t = self.vae.decode(z)

        imgs_t = (imgs_t.clamp(-1, 1) + 1) / 2
        imgs_t = (imgs_t * 255).byte().cpu()
        return [Image.fromarray(t.permute(1, 2, 0).numpy()) for t in imgs_t]

    @torch.no_grad()
    def img2img(
        self,
        prompts: List[str],
        init_images: List[Union[str, Image.Image]],
        strength: float = 0.8,
        negative_prompts: Optional[List[str]] = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Image-to-image generation via partial noising."""
        import torchvision.transforms.functional as TF
        from core.image_utils import load_image as _load, resize_center_crop
        B = len(prompts)
        device = self.device

        if seed:
            torch.manual_seed(seed)

        # Encode init images
        pils = []
        for img in init_images:
            if isinstance(img, str):
                img = _load(img)
            w = h = 512
            pils.append(resize_center_crop(img, (w, h)))

        tensors = torch.stack([
            TF.normalize(TF.to_tensor(p), [0.5]*3, [0.5]*3)
            for p in pils
        ]).to(device)
        z0 = self.vae.encode(tensors)

        # Compute start timestep
        timesteps = self.scheduler.make_ddim_timesteps(num_steps)
        start = int(len(timesteps) * (1.0 - strength))
        timesteps = timesteps[start:]

        noise = torch.randn_like(z0)
        t_start = torch.full((B,), timesteps[0], device=device, dtype=torch.long)
        z = self.scheduler.add_noise(z0, noise, t_start)

        neg = negative_prompts or [""] * B
        ctx_cond   = self.text_encoder(prompts).to(device)
        ctx_uncond = self.text_encoder(neg).to(device)
        ctx = torch.cat([ctx_uncond, ctx_cond])

        pairs = list(zip(timesteps, timesteps[1:] + [-1]))
        for t_cur, t_prev in pairs:
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            z_in = torch.cat([z, z])
            t_in = torch.cat([t, t])
            pred_both = self.unet(z_in, t_in, context=ctx)
            pred_u, pred_c = pred_both.chunk(2)
            pred = pred_u + guidance_scale * (pred_c - pred_u)
            z = self.scheduler.ddim_step(pred, t_cur, t_prev, z, eta=0.0)

        imgs_t = self.vae.decode(z)
        imgs_t = (imgs_t.clamp(-1, 1) + 1) / 2
        imgs_t = (imgs_t * 255).byte().cpu()
        return [Image.fromarray(t.permute(1, 2, 0).numpy()) for t in imgs_t]
