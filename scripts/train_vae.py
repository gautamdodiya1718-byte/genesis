#!/usr/bin/env python3
"""
scripts/train_vae.py
---------------------
Train the Genesis VAE on a custom image dataset.

This is Phase 1 of the training process. Run this BEFORE train.py.

Examples:
    # Train from scratch on your images
    python scripts/train_vae.py --image_dir /path/to/images

    # Fine-tune from a checkpoint
    python scripts/train_vae.py --image_dir /path/to/images --resume outputs/vae_training/vae_epoch_010.pt

    # Quick test (tiny model, few steps)
    python scripts/train_vae.py --image_dir /path/to/images vae_training.epochs=2 vae_training.batch_size=2
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
import logging


def main():
    parser = argparse.ArgumentParser(description="Train Genesis VAE")
    parser.add_argument("--config",      default="configs/base.yaml")
    parser.add_argument("--image_dir",   required=True,  help="Directory of training images")
    parser.add_argument("--val_dir",     default=None,   help="Validation images (optional)")
    parser.add_argument("--resume",      default=None,   help="Resume from VAE checkpoint")
    parser.add_argument("--epochs",      type=int,       default=None)
    parser.add_argument("--batch_size",  type=int,       default=None)
    parser.add_argument("--lr",          type=float,     default=None)
    parser.add_argument("--no_perceptual", action="store_true",  help="Disable perceptual loss")
    parser.add_argument("--adversarial",   action="store_true",  help="Enable PatchGAN discriminator")
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    overrides = parse_cli_overrides(args.overrides)
    if args.epochs:     overrides["vae_training.epochs"] = args.epochs
    if args.batch_size: overrides["vae_training.batch_size"] = args.batch_size
    if args.lr:         overrides["vae_training.learning_rate"] = args.lr
    if args.no_perceptual: overrides["vae_training.use_perceptual_loss"] = False
    if args.adversarial:   overrides["vae_training.use_adversarial_loss"] = True

    cfg = load_config(args.config, overrides=overrides)
    setup_logging(cfg.system.log_level)
    logger = logging.getLogger(__name__)

    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from pathlib import Path

    from models.vae.vae import VAE
    from training.vae_trainer import VAETrainer

    # ── Dataset ───────────────────────────────────────────────
    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    class ImageFolderDS(Dataset):
        def __init__(self, root, transform):
            self.paths = sorted(
                p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS
            )
            self.transform = transform
            logger.info(f"Dataset: {len(self.paths)} images from {root}")

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                img = Image.open(self.paths[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (512, 512))
            return self.transform(img), ""

    transform = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.LANCZOS),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    bs = cfg.vae_training.get("batch_size", 8)
    train_ds = ImageFolderDS(args.image_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=cfg.dataset.num_workers, pin_memory=True)

    val_loader = None
    if args.val_dir:
        val_ds = ImageFolderDS(args.val_dir, transform)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # ── Build VAE ─────────────────────────────────────────────
    vae = VAE(cfg)
    param_count = sum(p.numel() for p in vae.parameters())
    logger.info(f"VAE parameters: {param_count:,}")

    # ── Train ─────────────────────────────────────────────────
    trainer = VAETrainer(cfg, vae, train_loader, val_loader)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        vae.load_state_dict(state.get("vae", state))
        logger.info(f"Resumed VAE from: {args.resume}")

    trainer.train()
    logger.info("VAE training complete! Check outputs/vae_training/vae_final.pt")


if __name__ == "__main__":
    main()
