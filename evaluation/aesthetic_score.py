"""
evaluation/aesthetic_score.py
-------------------------------
Aesthetic quality scorer using a pretrained LAION aesthetic predictor.

Model: LAION-Aesthetics-Predictor V2 (MLP on top of CLIP ViT-L/14).
Score range: [0, 10]. Human-rated aesthetics.
  < 4.5 = poor,  4.5–6 = acceptable,  > 6 = good,  > 7 = excellent.

Falls back to a lightweight proxy scorer if LAION model is unavailable.
CPU-compatible (~750MB for ViT-L/14, ~15MB MLP head).
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from core.image_utils import load_image

logger = logging.getLogger(__name__)
_LAION_URL = ("https://github.com/christophschuhmann/"
              "improved-aesthetic-predictor/raw/main/"
              "sac+logos+ava1-l14-linearMSE.pth")


# ── LAION MLP head ────────────────────────────────────────────

class AestheticMLP(nn.Module):
    """Simple MLP head from LAION improved aesthetic predictor."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(1024, 128),       nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(128, 64),         nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(64, 16),          nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ── Proxy scorer (no downloads) ───────────────────────────────

class ProxyAestheticScorer:
    """
    Lightweight aesthetic proxy using image statistics.
    No model download required. Less accurate than LAION but always available.

    Signals used:
      - Colorfulness (Shannon entropy of HSV histogram)
      - Contrast (RMS pixel standard deviation)
      - Sharpness (Laplacian variance)
      - Saturation distribution
    Combined into a heuristic score in [0, 10].
    """

    def score(self, img: Image.Image) -> float:
        try:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            # Sharpness
            grey = arr.mean(-1)
            lap = (grey[:-2, 1:-1] + grey[2:, 1:-1]
                   + grey[1:-1, :-2] + grey[1:-1, 2:]
                   - 4 * grey[1:-1, 1:-1])
            sharpness = min(float(lap.var()) / 0.02, 1.0)   # normalise
            # Colorfulness: std of color channels
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            rg = r - g; yb = 0.5*(r+g) - b
            colorfulness = min(
                (float(rg.std())**2 + float(yb.std())**2)**0.5 / 0.3, 1.0)
            # Contrast: global std
            contrast = min(float(arr.std()) / 0.25, 1.0)
            # Saturation (HSV)
            hsv = np.array(img.convert("HSV") if hasattr(img, "convert") else img)
            saturation = min(float(hsv[:,:,1].mean()) / 200.0
                             if hsv.ndim == 3 else 0.5, 1.0)
            # Combine
            raw = (0.35 * sharpness + 0.30 * colorfulness
                   + 0.20 * contrast + 0.15 * saturation)
            # Map to [0,10]; realistic range ~3-8 for natural images
            return float(np.clip(raw * 10.0 * 0.6 + 3.5, 0.0, 10.0))
        except Exception:
            return 5.0


# ── Main AestheticScorer ──────────────────────────────────────

class AestheticScorer:
    """
    Aesthetic quality scorer.

    Tries to load the LAION aesthetic predictor (CLIP ViT-L/14 + MLP).
    Falls back to ProxyAestheticScorer if unavailable.

    Usage:
        scorer = AestheticScorer(device="cpu", cache_dir="model_cache")
        score = scorer.score(image)           # float in [0, 10]
        scores = scorer.score_batch(images)   # List[float]
    """

    def __init__(self, device: str = "cpu", batch_size: int = 16,
                 cache_dir: str = "model_cache/aesthetic",
                 use_laion: bool = True):
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.use_laion = use_laion
        self._mlp: Optional[AestheticMLP] = None
        self._clip = self._proc = None
        self._proxy = ProxyAestheticScorer()
        self._laion_loaded = False
        self._laion_failed = False

    def _load_laion(self) -> bool:
        """Try loading LAION model. Returns True on success."""
        if self._laion_loaded:
            return True
        if self._laion_failed:
            return False
        try:
            from transformers import CLIPProcessor, CLIPModel
            import urllib.request

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            mlp_path = self.cache_dir / "sac_logos_ava1_l14_linearMSE.pth"

            if not mlp_path.exists():
                logger.info("Downloading LAION aesthetic MLP weights (~15MB)...")
                urllib.request.urlretrieve(_LAION_URL, str(mlp_path))

            clip_id = "openai/clip-vit-large-patch14"
            logger.info(f"Loading CLIP ViT-L/14 for aesthetic scoring")
            self._proc = CLIPProcessor.from_pretrained(clip_id)
            self._clip = CLIPModel.from_pretrained(clip_id).to(self.device)
            self._clip.eval()

            self._mlp = AestheticMLP(input_dim=768)
            state = torch.load(str(mlp_path), map_location="cpu")
            self._mlp.load_state_dict(state)
            self._mlp.to(self.device).eval()

            self._laion_loaded = True
            logger.info("LAION aesthetic predictor loaded")
            return True
        except Exception as e:
            logger.warning(f"LAION aesthetic model unavailable: {e}. Using proxy scorer.")
            self._laion_failed = True
            return False

    def _clip_embed(self, images: List[Image.Image]) -> np.ndarray:
        inp = {k: v.to(self.device) for k, v in
               self._proc(images=images, return_tensors="pt",
                          padding=True).items()}
        with torch.no_grad():
            f = self._clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy()

    def score(self, image: Union[str, Path, Image.Image]) -> float:
        """Score a single image. Returns float in [0, 10]."""
        if isinstance(image, (str, Path)):
            pil = load_image(str(image))
            if pil is None:
                return 0.0
        else:
            pil = image

        if self.use_laion and self._load_laion():
            emb = self._clip_embed([pil])  # (1, 768)
            t = torch.from_numpy(emb).to(self.device)
            with torch.no_grad():
                s = self._mlp(t).item()
            return float(np.clip(s, 0.0, 10.0))
        else:
            return self._proxy.score(pil)

    def score_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
    ) -> List[float]:
        """Score multiple images."""
        pils, valid_idx = [], []
        for i, img in enumerate(images):
            pil = load_image(str(img)) if isinstance(img, (str, Path)) else img
            if pil is not None:
                pils.append(pil); valid_idx.append(i)

        scores = [0.0] * len(images)
        if not pils:
            return scores

        if self.use_laion and self._load_laion():
            all_embs = []
            for i in range(0, len(pils), self.batch_size):
                all_embs.append(self._clip_embed(pils[i:i + self.batch_size]))
            embs = np.vstack(all_embs)
            t = torch.from_numpy(embs).to(self.device)
            with torch.no_grad():
                s = self._mlp(t).squeeze(-1).cpu().numpy()
            for rank, idx in enumerate(valid_idx):
                scores[idx] = float(np.clip(s[rank], 0.0, 10.0))
        else:
            for rank, idx in enumerate(valid_idx):
                scores[idx] = self._proxy.score(pils[rank])

        return scores

    def score_directory(
        self,
        image_dir: str,
        max_images: Optional[int] = None,
    ) -> Dict:
        """Score all images in a directory."""
        _EXT = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(p for p in Path(image_dir).iterdir()
                       if p.suffix.lower() in _EXT)
        if max_images:
            paths = paths[:max_images]
        scores = self.score_batch([str(p) for p in paths])
        return {
            "n_images": len(paths),
            "mean": float(np.mean(scores)) if scores else 0.0,
            "std":  float(np.std(scores))  if scores else 0.0,
            "min":  float(np.min(scores))  if scores else 0.0,
            "max":  float(np.max(scores))  if scores else 0.0,
            "pct_above_5": sum(1 for s in scores if s > 5.0) / max(len(scores),1),
            "pct_above_6": sum(1 for s in scores if s > 6.0) / max(len(scores),1),
            "scores": [{"path": str(p), "score": s}
                       for p, s in zip(paths, scores)],
        }
