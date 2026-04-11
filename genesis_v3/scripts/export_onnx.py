#!/usr/bin/env python3
"""
scripts/export_onnx.py — Export Genesis models to ONNX for 2-3× CPU speedup

NEW in Genesis v0.2. Export once, benefit on every inference run.

Examples:
    # Export U-Net from a trained checkpoint
    python scripts/export_onnx.py --checkpoint outputs/checkpoints/step_00010000

    # Export default SD 1.5 UNet (no training needed)
    python scripts/export_onnx.py --pretrained

    # Benchmark PyTorch vs ONNX after export
    python scripts/export_onnx.py --checkpoint ... --benchmark

    # Force re-export even if ONNX already exists
    python scripts/export_onnx.py --checkpoint ... --force
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
from core.model_manager import ModelManager
from inference.onnx_exporter import ONNXExporter, ONNXUNet, benchmark_onnx_vs_pytorch
import logging


def main():
    parser = argparse.ArgumentParser(description="Export Genesis models to ONNX")
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Genesis checkpoint directory")
    parser.add_argument("--pretrained", action="store_true", help="Export pretrained SD UNet")
    parser.add_argument("--onnx_dir",   type=str, default="model_cache/onnx")
    parser.add_argument("--benchmark",  action="store_true", help="Run speed comparison after export")
    parser.add_argument("--force",      action="store_true", help="Re-export even if ONNX exists")
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config, overrides=parse_cli_overrides(args.overrides))
    setup_logging(cfg.system.log_level)
    logger = logging.getLogger(__name__)

    import torch
    exporter = ONNXExporter(onnx_dir=args.onnx_dir)

    if args.checkpoint:
        logger.info(f"Exporting UNet from checkpoint: {args.checkpoint}")
        from models.vae.vae import VAE
        from models.diffusion.unet import UNet
        from training.vae_trainer import VAETrainer

        unet = UNet(
            in_channels=cfg.diffusion.in_channels,
            out_channels=cfg.diffusion.out_channels,
            base_channels=cfg.diffusion.base_channels,
            channel_multipliers=list(cfg.diffusion.channel_multipliers),
            context_dim=cfg.diffusion.context_dim,
        )

        ckpt_path = os.path.join(args.checkpoint, "training_state.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            key = "ema" if "ema" in state else "unet"
            unet.load_state_dict(state[key])
            logger.info(f"Loaded {'EMA' if key=='ema' else 'raw'} UNet weights")

    elif args.pretrained:
        logger.info("Exporting pretrained SD 1.5 UNet")
        from diffusers import StableDiffusionPipeline
        mm = ModelManager(cfg.model_cache_dir)
        mm.ensure(cfg.generation.model_id, model_type="diffusion")
        local = mm.get_path(cfg.generation.model_id) or cfg.generation.model_id
        pipe = StableDiffusionPipeline.from_pretrained(
            local, torch_dtype=torch.float32, safety_checker=None
        )
        unet = pipe.unet
        logger.info("Pretrained SD UNet loaded")
    else:
        parser.error("Provide --checkpoint or --pretrained")

    # Export
    onnx_path = exporter.export_unet(
        unet=unet,
        context_dim=cfg.diffusion.context_dim,
        force=args.force,
    )
    print(f"\n✓ U-Net exported → {onnx_path}")

    # Optional benchmark
    if args.benchmark:
        logger.info("Running benchmark: PyTorch vs ONNX...")
        onnx_unet = ONNXUNet(onnx_path)
        unet.eval()
        result = benchmark_onnx_vs_pytorch(unet, onnx_unet)
        print(f"\n{'='*40}")
        print(f"  PyTorch : {result['pytorch_ms']} ms/step")
        print(f"  ONNX    : {result['onnx_ms']} ms/step")
        print(f"  Speedup : {result['speedup']}×")
        print(f"{'='*40}")
        print(f"\nAt 50 steps/image:")
        print(f"  PyTorch : {result['pytorch_ms']*50/1000:.1f}s")
        print(f"  ONNX    : {result['onnx_ms']*50/1000:.1f}s")

    print(f"\nTo use ONNX in generation: set generation.use_onnx=true in config.yaml")
    print(f"Or: python scripts/generate.py --checkpoint ... generation.use_onnx=true")


if __name__ == "__main__":
    main()
