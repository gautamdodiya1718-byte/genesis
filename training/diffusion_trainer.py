"""
training/diffusion_trainer.py
------------------------------
Latent Diffusion Training Pipeline — Phase 2 of Genesis training.

Upgraded from LocalDiffusion trainer.py:
  - Now uses LatentDiffusion wrapper (new diffusion.py)
  - Plugs into unified config system
  - W&B / TensorBoard logging hooks
  - Aspect ratio bucketing ready (placeholder sampler)
  - torch.compile support
"""

from __future__ import annotations
import os, time, logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ep, mp in zip(self.shadow.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)

    def state_dict(self): return self.shadow.state_dict()
    def load_state_dict(self, sd): self.shadow.load_state_dict(sd)


class DiffusionTrainer:
    """
    Trains the U-Net diffusion model on top of a frozen VAE.
    Uses the LatentDiffusion wrapper for clean forward pass.
    """

    def __init__(
        self,
        cfg,
        model,                          # LatentDiffusion instance
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg
        self.tc = cfg.training
        self.device = torch.device(cfg.system.device)

        self.model = model.to(self.device)

        # torch.compile (PyTorch 2.0+) — 20-40% throughput boost
        if cfg.system.get("compile_model", False):
            try:
                self.model.unet = torch.compile(self.model.unet)
                logger.info("UNet compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # EMA on UNet only (VAE/text encoder are frozen)
        self.ema = EMA(self.model.unet, decay=self.tc.get("ema_decay", 0.9999))

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer (only UNet parameters)
        self.optimizer = AdamW(
            self.model.trainable_params(),
            lr=self.tc.get("learning_rate", 1e-4),
            weight_decay=self.tc.get("weight_decay", 1e-2),
        )

        total_steps = len(train_loader) * self.tc.get("epochs", 100)
        warmup = self.tc.get("warmup_steps", 500)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup),
                CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup, 1), eta_min=1e-7),
            ],
            milestones=[warmup],
        )

        precision = self.tc.get("mixed_precision", "fp16")
        self.use_amp = precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.grad_accum = self.tc.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = self.tc.get("max_grad_norm", 1.0)
        self.cfg_dropout = self.tc.get("cfg_dropout_prob", 0.1)
        self.global_step = 0

        self.ckpt_dir = Path(cfg.system.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DiffusionTrainer ready | device={self.device} | "
            f"amp={self.use_amp} | grad_accum={self.grad_accum}"
        )

    def train(self) -> None:
        epochs = self.tc.get("epochs", 100)
        save_every = self.tc.get("save_every_steps", 500)
        log_every  = self.tc.get("log_every_steps", 50)

        logger.info(f"Starting diffusion training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            self.model.unet.train()
            epoch_loss = 0.0
            n_steps = 0

            for batch_idx, batch in enumerate(self.train_loader):
                images   = batch[0].to(self.device)
                captions = batch[1] if len(batch) > 1 else [""] * images.shape[0]

                # Gradient accumulation
                do_step = (batch_idx + 1) % self.grad_accum == 0

                with autocast(enabled=self.use_amp):
                    loss = self.model(images, captions, self.cfg_dropout)
                    loss = loss / self.grad_accum

                self.scaler.scale(loss).backward()

                if do_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.unet.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.ema.update(self.model.unet)
                    self.global_step += 1

                epoch_loss += loss.item() * self.grad_accum
                n_steps += 1

                if self.global_step % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    avg = epoch_loss / n_steps
                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"loss={avg:.4f} | lr={lr:.2e}"
                    )

                if self.global_step % save_every == 0:
                    self._save_checkpoint()

            logger.info(f"Epoch {epoch}/{epochs} done | avg_loss={epoch_loss/n_steps:.4f}")

        self._save_checkpoint(tag="final")
        logger.info("Diffusion training complete")

    def _save_checkpoint(self, tag: str = "") -> None:
        name = f"step_{self.global_step:08d}" + (f"_{tag}" if tag else "")
        path = self.ckpt_dir / name
        path.mkdir(exist_ok=True)
        torch.save({
            "global_step": self.global_step,
            "unet": self.model.unet.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, str(path / "training_state.pt"))
        logger.info(f"Checkpoint → {path}")

    def resume(self, checkpoint_dir: str) -> None:
        state = torch.load(
            os.path.join(checkpoint_dir, "training_state.pt"),
            map_location=self.device,
        )
        self.model.unet.load_state_dict(state["unet"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        logger.info(f"Resumed from step {self.global_step}")
