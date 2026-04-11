"""
dataset/intelligence/quality_filter.py
----------------------------------------
Multi-criterion image quality filter.
Cheapest filters run first; stops early on first failure.

Filters (in execution order):
  1. ResolutionFilter   — pixel dimensions
  2. AspectRatioFilter  — reject extreme crops
  3. BlurFilter         — Laplacian variance sharpness
  4. CaptionQualityFilter — length + garbage detection
  5. NSFWFilter         — CLIP zero-shot (optional, expensive)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from core.image_utils import load_image

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    passed: bool
    filter_name: str
    reason: str = ""
    score: Optional[float] = None


@dataclass
class FilterDecision:
    accepted: bool
    path: str
    caption: str = ""
    rejection_reason: str = ""
    results: List[FilterResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "path": self.path,
            "rejection_reason": self.rejection_reason,
            "filters": [{"name": r.filter_name, "passed": r.passed,
                          "reason": r.reason, "score": r.score}
                         for r in self.results],
        }


# ── individual filters ────────────────────────────────────────

class ResolutionFilter:
    def __init__(self, min_width: int = 256, min_height: int = 256,
                 max_width: int = 8192, max_height: int = 8192):
        self.min_w = min_width; self.min_h = min_height
        self.max_w = max_width; self.max_h = max_height

    def check(self, img: Image.Image) -> FilterResult:
        w, h = img.size
        if w < self.min_w or h < self.min_h:
            return FilterResult(False, "resolution",
                f"{w}x{h} below min {self.min_w}x{self.min_h}", float(min(w, h)))
        if w > self.max_w or h > self.max_h:
            return FilterResult(False, "resolution",
                f"{w}x{h} above max {self.max_w}x{self.max_h}", float(max(w, h)))
        return FilterResult(True, "resolution", f"{w}x{h}", float(min(w, h)))


class AspectRatioFilter:
    def __init__(self, max_ratio: float = 3.0):
        self.max_ratio = max_ratio

    def check(self, img: Image.Image) -> FilterResult:
        w, h = img.size
        ratio = max(w / h, h / w)
        if ratio > self.max_ratio:
            return FilterResult(False, "aspect_ratio",
                f"ratio={ratio:.2f} > {self.max_ratio}", ratio)
        return FilterResult(True, "aspect_ratio", f"ratio={ratio:.2f}", ratio)


class BlurFilter:
    """Laplacian variance sharpness. Low var = blurry."""

    def __init__(self, min_laplacian_variance: float = 50.0):
        self.threshold = min_laplacian_variance

    def check(self, img: Image.Image) -> FilterResult:
        try:
            arr = np.array(img.convert("L"), dtype=np.float32)
            lap = (arr[:-2, 1:-1] + arr[2:, 1:-1]
                   + arr[1:-1, :-2] + arr[1:-1, 2:]
                   - 4 * arr[1:-1, 1:-1])
            var = float(lap.var())
        except Exception:
            return FilterResult(True, "blur", "compute_failed", None)
        if var < self.threshold:
            return FilterResult(False, "blur",
                f"laplacian_var={var:.1f} < {self.threshold}", var)
        return FilterResult(True, "blur", f"laplacian_var={var:.1f}", var)


class CaptionQualityFilter:
    """Heuristic caption quality: length, uniqueness, garbage detection."""

    _GARBAGE_PREFIXES = ("image of a", "picture of a picture", "a image")
    _GARBAGE_TOKENS   = {"untitled", "dsc_", "img_", "screenshot", "photo_"}

    def __init__(self, min_words: int = 3, max_words: int = 75):
        self.min_words = min_words
        self.max_words = max_words

    def check(self, caption: str) -> FilterResult:
        cap = caption.strip().lower()
        if not cap:
            return FilterResult(False, "caption_quality", "empty", 0.0)
        words = cap.split()
        nw = len(words)
        if nw < self.min_words:
            return FilterResult(False, "caption_quality",
                f"too_short ({nw} words, min {self.min_words})",
                float(nw) / self.min_words)
        if nw > self.max_words:
            return FilterResult(False, "caption_quality",
                f"too_long ({nw} words, max {self.max_words})", 1.0)
        for pfx in self._GARBAGE_PREFIXES:
            if cap.startswith(pfx):
                return FilterResult(False, "caption_quality",
                    f"generic prefix: '{pfx}'", 0.3)
        for tok in self._GARBAGE_TOKENS:
            if any(tok in w for w in words):
                return FilterResult(False, "caption_quality",
                    f"garbage token '{tok}'", 0.2)
        score = min(1.0, len(set(words)) / max(nw, 1) * min(nw, 20) / 20)
        return FilterResult(True, "caption_quality", f"{nw} words", score)


class NSFWFilter:
    """CLIP zero-shot NSFW classifier. CPU-compatible. Loads on first use."""

    _SAFE = ["a safe family-friendly image", "a professional photograph",
             "a landscape photo", "a nature photograph", "a portrait photo"]
    _UNSAFE = ["explicit sexual content", "nudity", "graphic violence",
               "disturbing imagery", "adult content"]

    def __init__(self, threshold: float = 0.3,
                 model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu"):
        self.threshold = threshold
        self.model_id = model_id
        self.device = device
        self._clip = self._proc = self._text_feats = None

    def _load(self) -> None:
        if self._clip is not None:
            return
        import torch
        from transformers import CLIPProcessor, CLIPModel
        logger.info("Loading CLIP for NSFW filter")
        self._proc = CLIPProcessor.from_pretrained(self.model_id)
        self._clip = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._clip.eval()
        labels = self._SAFE + self._UNSAFE
        inp = {k: v.to(self.device)
               for k, v in self._proc(text=labels, return_tensors="pt",
                                       padding=True).items()}
        with torch.no_grad():
            tf = self._clip.get_text_features(**inp)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        self._text_feats = tf
        self._n_safe = len(self._SAFE)

    def check(self, img: Image.Image) -> FilterResult:
        try:
            self._load()
        except Exception as e:
            logger.warning(f"NSFW filter load failed: {e}")
            return FilterResult(True, "nsfw", "unavailable", None)
        import torch
        inp = {k: v.to(self.device)
               for k, v in self._proc(images=img, return_tensors="pt").items()}
        with torch.no_grad():
            f = self._clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (f @ self._text_feats.T).squeeze(0).softmax(0).cpu().numpy()
        nsfw_prob = float(probs[self._n_safe:].sum())
        if nsfw_prob > self.threshold:
            return FilterResult(False, "nsfw",
                f"nsfw_prob={nsfw_prob:.3f} > {self.threshold}", nsfw_prob)
        return FilterResult(True, "nsfw",
            f"safe_prob={float(probs[:self._n_safe].sum()):.3f}", nsfw_prob)


# ── Unified QualityFilter ─────────────────────────────────────

class QualityFilter:
    """
    Aggregates all filters. Runs cheapest-first, stops on first failure.

    Usage:
        qf = QualityFilter.from_config(cfg)
        decision = qf.check("image.jpg", caption="a mountain at sunset")
    """

    def __init__(
        self,
        resolution: Optional[ResolutionFilter] = None,
        aspect_ratio: Optional[AspectRatioFilter] = None,
        blur: Optional[BlurFilter] = None,
        caption: Optional[CaptionQualityFilter] = None,
        nsfw: Optional[NSFWFilter] = None,
        stop_on_first_fail: bool = True,
    ):
        # ordered cheapest → most expensive
        self._image_filters = [f for f in [resolution, aspect_ratio, blur, nsfw] if f]
        self._caption_filter = caption
        self.stop_on_first_fail = stop_on_first_fail

    @classmethod
    def from_config(cls, cfg) -> "QualityFilter":
        qf = cfg.get("quality_filter", {})
        res = qf.get("resolution", {})
        return cls(
            resolution=ResolutionFilter(
                min_width=res.get("min_width", 256),
                min_height=res.get("min_height", 256)),
            aspect_ratio=AspectRatioFilter(
                max_ratio=qf.get("max_aspect_ratio", 3.0)),
            blur=(BlurFilter(min_laplacian_variance=qf.get("min_sharpness", 50.0))
                  if qf.get("check_blur", True) else None),
            caption=CaptionQualityFilter(
                min_words=qf.get("min_caption_words", 3)),
            nsfw=(NSFWFilter(threshold=qf.get("nsfw_threshold", 0.3),
                             device=cfg.system.get("device", "cpu"))
                  if qf.get("check_nsfw", False) else None),
        )

    @classmethod
    def default(cls) -> "QualityFilter":
        return cls(
            resolution=ResolutionFilter(),
            aspect_ratio=AspectRatioFilter(),
            blur=BlurFilter(),
            caption=CaptionQualityFilter(),
            nsfw=None,  # disabled by default (slow)
        )

    def check(self, path: str, caption: str = "",
              img: Optional[Image.Image] = None) -> FilterDecision:
        decision = FilterDecision(accepted=True, path=path, caption=caption)
        pil = img or load_image(path)
        if pil is None:
            decision.accepted = False
            decision.rejection_reason = "load_failed"
            decision.results.append(FilterResult(False, "load", "image load failed"))
            return decision

        for filt in self._image_filters:
            result = filt.check(pil)
            decision.results.append(result)
            if not result.passed:
                decision.accepted = False
                decision.rejection_reason = f"{result.filter_name}: {result.reason}"
                if self.stop_on_first_fail:
                    return decision

        if self._caption_filter is not None:
            result = self._caption_filter.check(caption)
            decision.results.append(result)
            if not result.passed:
                decision.accepted = False
                decision.rejection_reason = f"{result.filter_name}: {result.reason}"

        return decision

    def filter_batch(
        self, items: List[Dict],
        show_progress: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter list of dicts with 'path' and optional 'caption' keys."""
        accepted, rejected = [], []
        for i, item in enumerate(items):
            if show_progress and i % 100 == 0:
                print(f"\rQuality filter {i}/{len(items)}", end="", flush=True)
            d = self.check(item["path"], item.get("caption", ""))
            item["filter_decision"] = d.to_dict()
            (accepted if d.accepted else rejected).append(item)
            if not d.accepted:
                item["rejection_reason"] = d.rejection_reason
        if show_progress:
            print()
        logger.info(
            f"Quality filter: {len(accepted)} accepted, {len(rejected)} rejected"
        )
        return accepted, rejected

    def rejection_stats(self, decisions: List[FilterDecision]) -> Dict:
        by_filter: Dict[str, int] = {}
        for d in decisions:
            if not d.accepted:
                key = d.rejection_reason.split(":")[0].strip()
                by_filter[key] = by_filter.get(key, 0) + 1
        return {
            "total": len(decisions),
            "accepted": sum(1 for d in decisions if d.accepted),
            "rejected": sum(1 for d in decisions if not d.accepted),
            "by_filter": by_filter,
        }
