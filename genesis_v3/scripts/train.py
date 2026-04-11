#!/usr/bin/env python3
"""
scripts/train.py
-----------------
Train the Genesis diffusion U-Net (Phase 2 — run AFTER train_vae.py).

Examples:
    # Train on Genesis dataset
    python scripts/train.py

    # Train on custom image+caption directory
    python scripts/train.py --image_dir /data/images --captions_json /data/captions.json

    # Resume from checkpoint
    python scripts/train.py --resume outputs/checkpoints/step_00005000

    # With config overrides
    python scripts/train.py training.learning_rate=5e-5 training.batch_size=8

    # Use pretrained VAE (skip train_vae.py)
    python scripts/train.py --pretrained_vae stabilityai/sd-vae-ft-mse
"""

import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
import logging


def main():
    parser = argparse.ArgumentParser(description="Train Genesis diffusion model")
    parser.add_argument("--config",         default="configs/base.yaml")
    parser.add_argument("--image_dir",      default=None, help="Image directory (defaults to Genesis dataset)")
    parser.add_argument("--captions_json",  default=None, help="JSON file mapping filename→caption")
    parser.add_argument("--resume",         default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--vae_checkpoint", default="outputs/vae_training/vae_final.pt",
                        help="Trained VAE checkpoint (from train_vae.py)")
    parser.add_argument("--pretrained_vae", default=None, help="HuggingFace VAE ID to use instead")
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--batch_size",     type=int,   default=None)
    parser.add_argument("--steps",          type=int,   default=None)
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    overrides = parse_cli_overrides(args.overrides)
    if args.epochs:     overrides["training.epochs"] = args.epochs
    if args.batch_size: overrides["training.batch_size"] = args.batch_size
    if args.pretrained_vae: overrides["vae.pretrained_id"] = args.pretrained_vae

    cfg = load_config(args.config, overrides=overrides)
    setup_logging(cfg.system.log_level)
    logger = logging.getLogger(__name__)

    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from pathlib import Path

    from models.vae.vae import VAE
    from models.diffusion.unet import UNet
    from models.diffusion.scheduler import NoiseScheduler
    from models.text_encoder.encoder import TextEncoder
    from models.diffusion.diffusion import LatentDiffusion
    from training.diffusion_trainer import DiffusionTrainer
    from training.vae_trainer import VAETrainer

    # ── Dataset ───────────────────────────────────────────────

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    class CaptionDataset(Dataset):
        def __init__(self, image_dir, captions, transform):
            self.paths    = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in EXTS)
            self.captions = captions
            self.transform = transform
            logger.info(f"Training dataset: {len(self.paths)} images")

        def __len__(self): return len(self.paths)

        def __getitem__(self, i):
            path = self.paths[i]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (512, 512))
            caption = self.captions.get(path.name, self.captions.get(str(path), ""))
            return self.transform(img), caption

    transform = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.LANCZOS),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    # Load captions
    captions = {}
    cap_path = args.captions_json or os.path.join(cfg.dataset.root, cfg.dataset.captions_file)
    if os.path.exists(cap_path):
        with open(cap_path) as f:
            captions = json.load(f)
        logger.info(f"Loaded {len(captions)} captions from {cap_path}")

    img_dir = args.image_dir or os.path.join(cfg.dataset.root, cfg.dataset.images_dir)
    bs = cfg.training.batch_size
    ds = CaptionDataset(img_dir, captions, transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=cfg.dataset.num_workers, pin_memory=True)

    # ── Build models ──────────────────────────────────────────

    vae = VAE(cfg)

    # Load VAE weights
    vae_path = args.vae_checkpoint
    if os.path.exists(vae_path):
        VAETrainer.load_vae_weights(vae, vae_path, use_ema=True)
    else:
        logger.warning(
            f"VAE checkpoint not found: {vae_path}. "
            f"Run scripts/train_vae.py first, or set --pretrained_vae."
        )

    unet = UNet(
        in_channels=cfg.diffusion.in_channels,
        out_channels=cfg.diffusion.out_channels,
        base_channels=cfg.diffusion.base_channels,
        channel_multipliers=list(cfg.diffusion.channel_multipliers),
        num_res_blocks=cfg.diffusion.num_res_blocks,
        context_dim=cfg.diffusion.context_dim,
    )

    text_encoder = TextEncoder(cfg)
    text_encoder.load()

    scheduler = NoiseScheduler(
        timesteps=cfg.diffusion.timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        prediction_type=cfg.diffusion.prediction_type,
    )

    model = LatentDiffusion(cfg, vae, unet, text_encoder, scheduler)

    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"UNet parameters: {unet_params:,}")

    # ── Train ─────────────────────────────────────────────────
    trainer = DiffusionTrainer(cfg, model, loader)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()
    logger.info("Diffusion training complete!")
    logger.info("Generate images with: python scripts/generate.py --checkpoint outputs/checkpoints/...")


if __name__ == "__main__":
    main()
