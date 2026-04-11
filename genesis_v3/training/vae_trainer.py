"""
training/vae_trainer.py
------------------------
VAE Training Pipeline — Phase 1 of the Genesis training process.

NEW in Genesis v0.2: This was the #1 item on the roadmap.
Without this, custom VAEs cannot be trained and the system
must rely on SD's pretrained VAE.

Training objective:
  L_total = L_recon + λ_kl * L_KL + λ_perc * L_perceptual
          [+ λ_adv * L_adversarial]  (optional)

Two-phase training workflow:
  1. VAETrainer.train() → saves vae_final.pt
  2. DiffusionTrainer loads frozen VAE from vae_final.pt and trains U-Net

Key design decisions:
  - Perceptual loss is critical for sharp reconstructions
  - KL weight kept very small (1e-6) to avoid blurry posteriors
  - Optional PatchGAN discriminator for even sharper textures
  - EMA on VAE weights for stable eval samples
"""

from __future__ import annotations
import os
import time
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from training.losses import ReconstructionLoss, PatchDiscriminator, AdversarialLoss

logger = logging.getLogger(__name__)


class VAETrainer:
    """
    Trains the VAE encoder and decoder from scratch (or fine-tunes
    a pretrained VAE on a new domain).

    Supports:
      - L1 + Perceptual loss (recommended default)
      - SSIM loss (optional)
      - PatchGAN adversarial loss (optional, for sharpest results)
      - Mixed precision (fp16/bf16)
      - Gradient checkpointing
      - Periodic reconstruction samples saved to disk
    """

    def __init__(
        self,
        cfg,
        vae: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg
        self.vt = cfg.vae_training   # shorthand
        self.device = torch.device(cfg.system.device)

        # Model
        self.vae = vae.to(self.device)
        self.vae.train()

        # EMA
        self.ema_vae = deepcopy(self.vae)
        self.ema_vae.eval()
        for p in self.ema_vae.parameters():
            p.requires_grad_(False)
        self._ema_decay = 0.999

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.loss_fn = ReconstructionLoss(
            recon_type=self.vt.get("recon_loss", "l1"),
            use_perceptual=self.vt.get("use_perceptual_loss", True),
            perceptual_weight=self.vt.get("perceptual_weight", 1.0),
            kl_weight=self.vt.get("kl_weight", 1e-6),
        ).to(self.device)

        # Adversarial components (optional)
        self.use_adv = self.vt.get("use_adversarial_loss", False)
        if self.use_adv:
            self.discriminator = PatchDiscriminator().to(self.device)
            self.adv_loss = AdversarialLoss()
            self.opt_d = AdamW(
                self.discriminator.parameters(),
                lr=self.vt.get("learning_rate", 1e-4),
                weight_decay=1e-6,
            )
            logger.info("PatchGAN discriminator enabled")

        # Optimizer & scheduler
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=self.vt.get("learning_rate", 1e-4),
            weight_decay=self.vt.get("weight_decay", 1e-6),
        )

        total_steps = len(train_loader) * self.vt.get("epochs", 50)
        warmup = self.vt.get("warmup_steps", 100)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup),
                CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup, 1), eta_min=1e-7),
            ],
            milestones=[warmup],
        )

        # Mixed precision
        precision = cfg.training.get("mixed_precision", "no")
        self.use_amp = precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.global_step = 0
        self.output_dir = Path(cfg.system.output_dir) / "vae_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"VAETrainer ready | "
            f"params={sum(p.numel() for p in vae.parameters()):,} | "
            f"device={self.device} | amp={self.use_amp} | adv={self.use_adv}"
        )

    # ── Main train loop ────────────────────────────────────────

    def train(self) -> None:
        """Run the full VAE training loop."""
        epochs = self.vt.get("epochs", 50)
        save_every = self.vt.get("save_every_epochs", 5)
        log_imgs_every = self.vt.get("log_images_every", 500)

        logger.info(f"Starting VAE training for {epochs} epochs")
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_losses = self._train_epoch(epoch, log_imgs_every)
            elapsed = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"loss={train_losses['total']:.4f} | "
                f"recon={train_losses.get('recon', 0):.4f} | "
                f"perc={train_losses.get('perceptual', 0):.4f} | "
                f"kl={train_losses.get('kl', 0):.6f} | "
                f"time={elapsed:.0f}s"
            )

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                logger.info(f"  Val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("vae_best.pt")

            # Periodic save
            if epoch % save_every == 0:
                self._save_checkpoint(f"vae_epoch_{epoch:03d}.pt")

        # Save final
        self._save_checkpoint("vae_final.pt")
        logger.info("VAE training complete")

    def _train_epoch(self, epoch: int, log_imgs_every: int) -> dict:
        """Train for one epoch. Returns averaged loss dict."""
        self.vae.train()
        running = {}
        n_batches = 0

        for batch in self.train_loader:
            images = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            loss, loss_dict = self._train_step(images)

            # Accumulate losses
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

            # Log sample images
            if self.global_step % log_imgs_every == 0:
                self._log_reconstructions(images[:4])

            if self.global_step % 100 == 0:
                lr = self.scheduler.get_last_lr()[0]
                avg_loss = running.get("total", 0) / n_batches
                logger.debug(
                    f"  Step {self.global_step} | loss={avg_loss:.4f} | lr={lr:.2e}"
                )

            self.global_step += 1

        return {k: v / n_batches for k, v in running.items()}

    def _train_step(self, images: torch.Tensor) -> tuple:
        """Single training step. Returns (loss, loss_dict)."""
        # ── Generator / VAE step ──────────────────────────────
        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            recon, kl = self.vae(images)
            loss, loss_dict = self.loss_fn(recon, images, kl)

            # Adversarial generator loss
            if self.use_adv:
                fake_logits = self.discriminator(recon)
                adv_g = self.adv_loss.generator_loss(fake_logits)
                loss = loss + 0.1 * adv_g
                loss_dict["adv_g"] = adv_g.item()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # ── Discriminator step (if using adversarial loss) ────
        if self.use_adv:
            self.opt_d.zero_grad()
            with autocast(enabled=self.use_amp):
                real_logits = self.discriminator(images.detach())
                fake_logits = self.discriminator(recon.detach())
                loss_d = self.adv_loss.discriminator_loss(real_logits, fake_logits)
            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_d)
            self.scaler.update()
            loss_dict["disc"] = loss_d.item()

        # ── EMA update ────────────────────────────────────────
        with torch.no_grad():
            for ema_p, p in zip(self.ema_vae.parameters(), self.vae.parameters()):
                ema_p.data.mul_(self._ema_decay).add_(p.data, alpha=1.0 - self._ema_decay)

        return loss, loss_dict

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation loop. Returns average total loss."""
        self.ema_vae.eval()
        total = 0.0
        n = 0

        for batch in self.val_loader:
            images = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            recon, kl = self.ema_vae(images)
            _, loss_dict = self.loss_fn(recon, images, kl)
            total += loss_dict["total"]
            n += 1

        return total / max(n, 1)

    @torch.no_grad()
    def _log_reconstructions(self, images: torch.Tensor) -> None:
        """Save a grid of original vs reconstructed images for visual inspection."""
        self.ema_vae.eval()
        recon, _ = self.ema_vae(images)

        # Denormalize [-1,1] → [0,255]
        def to_pil(t):
            t = ((t.clamp(-1, 1) + 1) / 2 * 255).byte()
            return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())

        out_dir = self.output_dir / "samples"
        out_dir.mkdir(exist_ok=True)

        for i, (orig, rec) in enumerate(zip(images, recon)):
            to_pil(orig).save(str(out_dir / f"step{self.global_step:06d}_orig_{i}.png"))
            to_pil(rec).save(str(out_dir  / f"step{self.global_step:06d}_recon_{i}.png"))

        self.ema_vae.train()

    def _save_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        torch.save({
            "vae": self.vae.state_dict(),
            "ema_vae": self.ema_vae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
        }, str(path))
        logger.info(f"Saved VAE checkpoint → {path}")

    @classmethod
    def load_vae_weights(
        cls,
        vae: nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
    ) -> nn.Module:
        """
        Load VAE weights from a training checkpoint.
        Used to load the trained VAE into the diffusion training pipeline.
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        key = "ema_vae" if use_ema and "ema_vae" in state else "vae"
        vae.load_state_dict(state[key])
        logger.info(f"Loaded VAE weights ({'EMA' if use_ema else 'raw'}) ← {checkpoint_path}")
        return vae
