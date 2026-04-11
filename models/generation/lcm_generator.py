"""
models/generation/lcm_generator.py
-------------------------------------
LCM (Latent Consistency Model) Generator — NEW in Genesis v0.2.

LCM is the #1 CPU speed improvement from the roadmap:
  - 4 steps vs 20-50 for SD 1.5  →  5-10× faster
  - Quality comparable to SD 1.5 at 20 steps for most subjects
  - Same pipeline API as the SD generator (drop-in replacement)

How LCM works:
  LCMs are distilled from diffusion models using Consistency Distillation.
  Instead of predicting noise (epsilon), the model learns a direct mapping
  from any noisy latent z_t → clean latent z_0 in very few steps.
  This collapses the 50-step denoising chain into 4-8 steps.

Model: SimianLuo/LCM_Dreamshaper_v7
  ~3.8GB, compatible with standard SD 1.5 checkpoint format.
  Uses LCMScheduler from diffusers (not DDIM/DDPM).
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from core.model_manager import ModelManager
from core.image_utils import save_image, load_image, resize_center_crop

logger = logging.getLogger(__name__)


class LCMGenerator:
    """
    CPU-optimized image generator using Latent Consistency Models.

    Generates images in 4-8 steps (vs 20-50 for standard SD),
    making it dramatically faster on CPU — the primary deployment target
    for the AutoDiff autonomous pipeline.

    Drop-in replacement for ImageGenerator — identical API.
    """

    def __init__(self, cfg, model_manager: ModelManager):
        self.cfg = cfg
        self.model_manager = model_manager
        self.pipe = None
        self._loaded = False
        self.device = "cpu"

        # LCM-specific settings
        self.lcm_cfg = cfg.get("lcm", {})
        self.model_id = self.lcm_cfg.get("model_id", "SimianLuo/LCM_Dreamshaper_v7")
        self.default_steps = self.lcm_cfg.get("num_inference_steps", 4)
        # LCM uses very low guidance scale (1.0-2.0) — high scale degrades quality
        self.default_guidance = self.lcm_cfg.get("guidance_scale", 1.0)

    def load(self) -> None:
        if self._loaded:
            return

        logger.info(f"Loading LCM model: {self.model_id}")

        try:
            from diffusers import DiffusionPipeline, LCMScheduler
        except ImportError:
            raise ImportError(
                "diffusers >= 0.21.0 required for LCM support. "
                "Run: pip install -U diffusers"
            )

        self.model_manager.ensure(self.model_id, model_type="diffusion")
        local_path = self.model_manager.get_path(self.model_id)
        load_from = local_path or self.model_id

        self.pipe = DiffusionPipeline.from_pretrained(
            load_from,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Swap in LCM scheduler — this is what makes 4-step generation work
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # CPU memory optimizations
        try:
            self.pipe.enable_attention_slicing(1)
        except Exception:
            pass
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

        self._loaded = True
        logger.info(
            f"LCM loaded | {self.default_steps} steps | "
            f"guidance={self.default_guidance} | device=cpu"
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images using LCM (4-8 steps, CPU-optimized).

        NOTE: guidance_scale for LCM should be 1.0-2.0.
        Values > 2.0 will degrade output quality (unlike standard SD).
        """
        if not self._loaded:
            self.load()

        steps    = steps    or self.default_steps
        guidance = guidance_scale or self.default_guidance
        width    = width    or self.cfg.generation.get("width", 512)
        height   = height   or self.cfg.generation.get("height", 512)

        # Warn if guidance is set too high (common mistake from SD users)
        if guidance > 3.0:
            logger.warning(
                f"LCM guidance_scale={guidance} is high. "
                f"Recommend 1.0-2.0 for LCM. High values degrade quality."
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            f"LCM generating {num_images}× | '{prompt[:60]}' | "
            f"{steps} steps | {width}×{height}"
        )
        start = time.time()

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )

        elapsed = time.time() - start
        logger.info(f"LCM done in {elapsed:.1f}s ({elapsed/num_images:.1f}s/img)")

        return result.images

    def generate_img2img(
        self,
        prompt: str,
        init_image: Union[str, Path, Image.Image],
        strength: float = 0.6,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """LCM image-to-image — use lower strength than SD (0.4-0.7 typical)."""
        if not self._loaded:
            self.load()

        from diffusers import LCMScheduler

        # Need img2img pipeline
        try:
            from diffusers import AutoPipelineForImage2Image
            pipe_i2i = AutoPipelineForImage2Image.from_pipe(self.pipe)
            pipe_i2i.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            logger.warning("LCM img2img pipeline unavailable, falling back to txt2img")
            return self.generate(prompt, steps=steps, guidance_scale=guidance_scale, seed=seed)

        if isinstance(init_image, (str, Path)):
            init_image = load_image(init_image)

        w = self.cfg.generation.get("width", 512)
        h = self.cfg.generation.get("height", 512)
        init_image = resize_center_crop(init_image, (w, h))

        generator = torch.Generator(device="cpu").manual_seed(seed) if seed else None

        result = pipe_i2i(
            prompt=prompt,
            image=init_image,
            num_inference_steps=steps or self.default_steps,
            guidance_scale=guidance_scale or self.default_guidance,
            strength=strength,
            generator=generator,
        )
        return result.images

    def save_images(
        self, images: List[Image.Image],
        output_dir: str, prefix: str = "lcm", start_idx: int = 0
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"{prefix}_{start_idx + i:06d}.png")
            save_image(img, path)
            paths.append(path)
        return paths

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._loaded = False
        logger.info("LCM model unloaded")


class AdaptiveGenerator:
    """
    Smart generator that automatically selects LCM or SD based on config
    and available resources.

    Decision logic:
      - If lcm.enabled=true → use LCM (4 steps, fast)
      - If device=cpu → prefer LCM (much faster)
      - If device=cuda → use SD (higher quality at 50 steps)
      - Falls back gracefully if LCM model unavailable
    """

    def __init__(self, cfg, model_manager: ModelManager):
        self.cfg = cfg
        self.model_manager = model_manager
        self._generator = None
        self._mode = None

    def _init_generator(self):
        if self._generator is not None:
            return

        use_lcm = (
            self.cfg.get("lcm", {}).get("enabled", False)
            or self.cfg.system.device == "cpu"
        )

        if use_lcm:
            try:
                self._generator = LCMGenerator(self.cfg, self.model_manager)
                self._mode = "lcm"
                logger.info("AdaptiveGenerator: using LCM (fast CPU mode)")
                return
            except Exception as e:
                logger.warning(f"LCM unavailable ({e}), falling back to SD")

        # Fall back to standard SD
        from models.generation.sd_generator import SDGenerator
        self._generator = SDGenerator(self.cfg, self.model_manager)
        self._mode = "sd"
        logger.info("AdaptiveGenerator: using SD 1.5")

    def generate(self, *args, **kwargs) -> List[Image.Image]:
        self._init_generator()
        return self._generator.generate(*args, **kwargs)

    def generate_img2img(self, *args, **kwargs) -> List[Image.Image]:
        self._init_generator()
        return self._generator.generate_img2img(*args, **kwargs)

    def save_images(self, *args, **kwargs) -> List[str]:
        return self._generator.save_images(*args, **kwargs)

    def unload(self) -> None:
        if self._generator:
            self._generator.unload()

    @property
    def mode(self) -> str:
        return self._mode or "uninitialized"
