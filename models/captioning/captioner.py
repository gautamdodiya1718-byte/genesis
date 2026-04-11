"""
models/captioning/captioner.py
--------------------------------
BLIP / ViT-GPT2 image captioner. Merged from AutoDiff captioner.py
with updated imports for the Genesis unified system.
"""

from __future__ import annotations
import logging, time
from pathlib import Path
from typing import List, Optional, Union, Dict

from PIL import Image

from core.model_manager import ModelManager
from core.image_utils import load_image

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """Batch image captioner using BLIP (default) or ViT-GPT2."""

    def __init__(self, cfg, model_manager: ModelManager):
        self.cfg = cfg
        self.model_manager = model_manager
        self.processor = None
        self.model = None
        self._loaded = False
        self._type = None

    def load(self) -> None:
        if self._loaded:
            return
        model_id = self.cfg.captioning.model_id
        logger.info(f"Loading captioner: {model_id}")

        if "blip" in model_id.lower():
            self._type = "blip2" if "blip2" in model_id.lower() else "blip"
            self.model_manager.ensure(model_id, model_type=self._type)
        else:
            self._type = "vit_gpt2"
            self.model_manager.ensure(model_id, model_type="vit_gpt2")

        local = self.model_manager.get_path(model_id) or model_id

        try:
            if self._type == "blip2":
                self._load_blip2(local, model_id)
            elif self._type == "blip":
                self._load_blip(local, model_id)
            else:
                self._load_vit_gpt2(local, model_id)
            self._loaded = True
            logger.info(f"Captioner loaded: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load captioner: {e}")
            raise

    def _load_blip(self, local: str, fallback: str) -> None:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        try:
            self.processor = BlipProcessor.from_pretrained(local)
            self.model = BlipForConditionalGeneration.from_pretrained(local, torch_dtype=torch.float32)
        except Exception:
            self.processor = BlipProcessor.from_pretrained(fallback)
            self.model = BlipForConditionalGeneration.from_pretrained(fallback, torch_dtype=torch.float32)
        self.model.eval()

    def _load_blip2(self, local: str, fallback: str) -> None:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        try:
            self.processor = Blip2Processor.from_pretrained(local)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                local, torch_dtype=torch.float16, device_map="auto"
            )
        except Exception:
            self.processor = Blip2Processor.from_pretrained(fallback)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                fallback, torch_dtype=torch.float16, device_map="auto"
            )
        self.model.eval()

    def _load_vit_gpt2(self, local: str, fallback: str) -> None:
        from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
        try:
            self.processor = ViTFeatureExtractor.from_pretrained(local)
            self.model = VisionEncoderDecoderModel.from_pretrained(local)
            self.tokenizer = AutoTokenizer.from_pretrained(local)
        except Exception:
            self.processor = ViTFeatureExtractor.from_pretrained(fallback)
            self.model = VisionEncoderDecoderModel.from_pretrained(fallback)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model.eval()

    # ── Caption generation ─────────────────────────────────────

    def caption(self, image: Union[str, Path, Image.Image], prompt: str = "") -> str:
        if not self._loaded:
            self.load()
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if image is None:
            return ""
        try:
            if self._type in ("blip", "blip2"):
                return self._caption_blip(image, prompt)
            else:
                return self._caption_vit(image)
        except Exception as e:
            logger.warning(f"Caption failed: {e}")
            return ""

    def _caption_blip(self, image: Image.Image, prompt: str = "") -> str:
        import torch
        inputs = self.processor(image, text=prompt or None, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.captioning.max_new_tokens,
                num_beams=self.cfg.captioning.num_beams,
            )
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def _caption_vit(self, image: Image.Image) -> str:
        import torch
        px = self.processor(images=[image], return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = self.model.generate(px, max_new_tokens=self.cfg.captioning.max_new_tokens)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def caption_batch(
        self, images: List[Union[str, Path, Image.Image]],
        show_progress: bool = True,
    ) -> List[Dict]:
        if not self._loaded:
            self.load()
        results = []
        for i, img in enumerate(images):
            path_str = str(img) if isinstance(img, (str, Path)) else f"image_{i}"
            if show_progress:
                print(f"\rCaptioning {i+1}/{len(images)}: {Path(path_str).name[:40]}", end="", flush=True)
            t = time.time()
            cap = self.caption(img)
            results.append({"path": path_str, "caption": cap, "time_s": round(time.time()-t, 2)})
        if show_progress:
            print()
        logger.info(f"Captioned {len(images)} images in {sum(r['time_s'] for r in results):.1f}s")
        return results

    def caption_directory(self, d: Union[str, Path], exts=(".jpg",".jpeg",".png",".webp")) -> List[Dict]:
        paths = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in exts)
        logger.info(f"Found {len(paths)} images in {d}")
        return self.caption_batch(paths)

    def unload(self) -> None:
        for attr in ("model", "processor"):
            if hasattr(self, attr): setattr(self, attr, None)
        self._loaded = False
        logger.info("Captioner unloaded")
