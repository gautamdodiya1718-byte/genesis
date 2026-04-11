"""
models/generation/sd_generator.py
-----------------------------------
Standard Stable Diffusion generator (CPU-optimized).
Merged and updated from AutoDiff generator.py.
Use LCMGenerator for faster CPU generation (4 steps).
"""

from __future__ import annotations
import logging, os, time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from core.model_manager import ModelManager
from core.image_utils import save_image, load_image, resize_center_crop

logger = logging.getLogger(__name__)


class SDGenerator:
    """Standard SD 1.5 generator — solid quality, slow on CPU."""

    def __init__(self, cfg, model_manager: ModelManager):
        self.cfg = cfg
        self.model_manager = model_manager
        self.pipe_txt2img = None
        self.pipe_img2img = None
        self._loaded = False
        self.device = cfg.system.device if hasattr(cfg.system, "device") else "cpu"

    def load(self) -> None:
        if self._loaded:
            return
        model_id = self.cfg.generation.model_id
        logger.info(f"Loading SD model: {model_id}")
        self.model_manager.ensure(model_id, model_type="diffusion")
        local = self.model_manager.get_path(model_id) or model_id

        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        self.pipe_txt2img = StableDiffusionPipeline.from_pretrained(
            local, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False,
        ).to(self.device)

        try: self.pipe_txt2img.enable_attention_slicing(1)
        except Exception: pass
        try: self.pipe_txt2img.enable_vae_slicing()
        except Exception: pass

        self.pipe_img2img = StableDiffusionImg2ImgPipeline(
            vae=self.pipe_txt2img.vae,
            text_encoder=self.pipe_txt2img.text_encoder,
            tokenizer=self.pipe_txt2img.tokenizer,
            unet=self.pipe_txt2img.unet,
            scheduler=self.pipe_txt2img.scheduler,
            safety_checker=None,
            feature_extractor=getattr(self.pipe_txt2img, "feature_extractor", None),
            requires_safety_checker=False,
        ).to(self.device)

        self._loaded = True
        logger.info(f"SD loaded | device={self.device}")

    def generate(
        self, prompt: str, negative_prompt: Optional[str] = None,
        num_images: int = 1, steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None, height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        if not self._loaded: self.load()

        steps    = steps    or self.cfg.generation.num_inference_steps
        guidance = guidance_scale or self.cfg.generation.guidance_scale
        width    = width    or self.cfg.generation.width
        height   = height   or self.cfg.generation.height
        neg      = negative_prompt or "blurry, low quality, distorted, watermark"
        gen      = torch.Generator(self.device).manual_seed(seed) if seed else None

        logger.info(f"SD generating '{prompt[:60]}' | {steps}steps {width}×{height}")
        start = time.time()
        result = self.pipe_txt2img(
            prompt=prompt, negative_prompt=neg,
            num_images_per_prompt=num_images,
            num_inference_steps=steps, guidance_scale=guidance,
            width=width, height=height, generator=gen,
        )
        logger.info(f"SD done in {time.time()-start:.1f}s")
        return result.images

    def generate_img2img(
        self, prompt: str, init_image: Union[str, Path, Image.Image],
        negative_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        steps: Optional[int] = None, guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        if not self._loaded: self.load()
        if isinstance(init_image, (str, Path)):
            init_image = load_image(init_image)
        w = self.cfg.generation.width
        h = self.cfg.generation.height
        init_image = resize_center_crop(init_image, (w, h))
        gen = torch.Generator(self.device).manual_seed(seed) if seed else None
        result = self.pipe_img2img(
            prompt=prompt, image=init_image,
            negative_prompt=negative_prompt or "blurry, low quality",
            strength=strength or self.cfg.generation.img2img_strength,
            num_inference_steps=steps or self.cfg.generation.num_inference_steps,
            guidance_scale=guidance_scale or self.cfg.generation.guidance_scale,
            generator=gen,
        )
        return result.images

    def save_images(self, images, output_dir, prefix="gen", start_idx=0):
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            p = os.path.join(output_dir, f"{prefix}_{start_idx+i:06d}.png")
            save_image(img, p)
            paths.append(p)
        return paths

    def unload(self) -> None:
        for attr in ("pipe_txt2img", "pipe_img2img"):
            if getattr(self, attr): delattr(self, attr)
        self._loaded = False
        logger.info("SD unloaded")
