#!/usr/bin/env python3
"""
scripts/generate.py
--------------------
Image generation CLI for Genesis.

Supports:
  - txt2img (SD or LCM, CPU or GPU)
  - img2img with controllable strength
  - Batch generation from prompt categories
  - Trained Genesis model checkpoints

Examples:
    # Basic txt2img (auto LCM on CPU)
    python scripts/generate.py --prompt "mountain at sunset, oil painting"

    # Category batch
    python scripts/generate.py --category landscape --count 5

    # Force LCM for speed
    python scripts/generate.py --prompt "..." --lcm --steps 4

    # img2img
    python scripts/generate.py --init_image photo.jpg --prompt "same but in winter" --strength 0.7

    # Use trained Genesis checkpoint
    python scripts/generate.py --checkpoint outputs/checkpoints/step_00010000 --prompt "..."

    # Add to dataset automatically
    python scripts/generate.py --category nature --count 10 --add_to_dataset
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
from core.image_utils import save_image
from models.generation.prompt_engine import PromptEngine
from core.model_manager import ModelManager
import logging


def main():
    parser = argparse.ArgumentParser(description="Genesis image generator")
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--prompt",     type=str, default=None)
    parser.add_argument("--category",   type=str, default=None)
    parser.add_argument("--count",      type=int, default=1)
    parser.add_argument("--init_image", type=str, default=None, help="img2img source")
    parser.add_argument("--strength",   type=float, default=0.75)
    parser.add_argument("--steps",      type=int,   default=None)
    parser.add_argument("--guidance",   type=float, default=None)
    parser.add_argument("--width",      type=int,   default=None)
    parser.add_argument("--height",     type=int,   default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--lcm",        action="store_true")
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Path to Genesis training checkpoint (trained model)")
    parser.add_argument("--vae_ckpt",   type=str,   default=None)
    parser.add_argument("--output_dir", type=str,   default="outputs/generated_images")
    parser.add_argument("--add_to_dataset", action="store_true")
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    overrides = parse_cli_overrides(args.overrides)
    if args.steps:    overrides["generation.num_inference_steps"] = args.steps
    if args.guidance: overrides["inference.guidance_scale"] = args.guidance
    if args.width:    overrides["generation.width"] = args.width
    if args.height:   overrides["generation.height"] = args.height
    if args.lcm:      overrides["lcm.enabled"] = True

    cfg = load_config(args.config, overrides=overrides)
    setup_logging(cfg.system.log_level)
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine generation mode
    if args.checkpoint:
        logger.info(f"Using trained Genesis checkpoint: {args.checkpoint}")
        from inference.pipeline import CustomPipeline
        pipe = CustomPipeline.from_checkpoint(
            checkpoint_dir=args.checkpoint,
            cfg=cfg,
            vae_checkpoint=args.vae_ckpt,
            use_onnx=cfg.generation.get("use_onnx", False),
        )
        prompts_list = [args.prompt or "a beautiful landscape"]
        images = pipe.txt2img(
            prompts=prompts_list * args.count,
            num_steps=args.steps or cfg.inference.num_inference_steps,
            guidance_scale=args.guidance or cfg.inference.guidance_scale,
            seed=args.seed,
        )
    else:
        mm = ModelManager(cfg.model_cache_dir)
        from models.generation.lcm_generator import AdaptiveGenerator
        gen = AdaptiveGenerator(cfg, mm)

        # Resolve prompts
        prompt_engine = PromptEngine(cfg.prompts.get("templates_file", "configs/prompt_templates.yaml"))
        if args.prompt:
            prompt_obj = prompt_engine.from_raw(args.prompt)
            text_prompts = [args.prompt] * args.count
            neg_prompts  = [prompt_obj.negative] * args.count
        else:
            cat = args.category or "landscape"
            p_list = prompt_engine.generate(category=cat, count=args.count, seed=args.seed)
            text_prompts = [p.text for p in p_list]
            neg_prompts  = [p.negative for p in p_list]

        images = []
        for i, (tp, np_) in enumerate(zip(text_prompts, neg_prompts)):
            logger.info(f"Generating {i+1}/{len(text_prompts)}: '{tp[:60]}'")
            if args.init_image:
                imgs = gen.generate_img2img(
                    prompt=tp, init_image=args.init_image,
                    strength=args.strength, seed=args.seed,
                )
            else:
                imgs = gen.generate(
                    prompt=tp, negative_prompt=np_,
                    steps=args.steps, guidance_scale=args.guidance,
                    width=args.width, height=args.height,
                    seed=args.seed, num_images=1,
                )
            images.extend(imgs)

    # Save all images
    saved = []
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f"output_{i:04d}.png")
        save_image(img, path)
        saved.append(path)
        logger.info(f"Saved: {path}")

    # Optionally add to dataset
    if args.add_to_dataset:
        from dataset.builder import DatasetBuilder
        ds = DatasetBuilder(cfg)
        for path, img in zip(saved, images):
            caption = args.prompt or "generated image"
            ds.add_image(img, caption=caption, source="generated", prompt=caption)
        logger.info(f"Added {len(saved)} images to dataset")

    print(f"\nDone! Generated {len(saved)} images → {args.output_dir}")


if __name__ == "__main__":
    main()
