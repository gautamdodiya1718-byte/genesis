"""
evaluation/fid_score.py
------------------------
Fréchet Inception Distance (FID) between real and generated image sets.

Lower FID = distributions closer = better quality.
SD 1.5 reference ≈ 12-15 FID on COCO. Good custom model target < 30.

CPU-compatible. Real statistics are cached to disk (compute once).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)
_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 truncated at pool_3 (2048-dim). Standard FID backbone."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.device = device
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()
        self.net.aux_logits = False
        self.net.AuxLogits = None
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 2048)


def _preprocess(img: Image.Image, device: str) -> torch.Tensor:
    import torchvision.transforms.functional as F
    img = img.convert("RGB").resize((299, 299), Image.LANCZOS)
    t = F.to_tensor(img) * 2 - 1  # [0,1] → [-1,1]
    return t.to(device)


def _load_dir(d: str, max_n: Optional[int]) -> List[Image.Image]:
    paths = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in _EXT)
    if max_n:
        paths = paths[:max_n]
    return [img for p in paths
            if (img := _open(p)) is not None]


def _open(p: Path) -> Optional[Image.Image]:
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def _compute_stats(images: List[Image.Image],
                   model: InceptionFeatureExtractor,
                   batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([_preprocess(im, model.device)
                              for im in images[i:i + batch_size]])
        with torch.no_grad():
            f = model(batch).cpu().float().numpy()
        all_feats.append(f)
    feats = np.concatenate(all_feats, 0)  # (N, 2048)
    return feats.mean(0), np.cov(feats, rowvar=False)


def _frechet(mu1, s1, mu2, s2, eps: float = 1e-6) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        offset = eps * np.eye(s1.shape[0])
        covmean, _ = sqrtm((s1 + offset) @ (s2 + offset), disp=False)
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


class FIDScorer:
    """
    FID between real and generated sets.
    Real statistics are cached to avoid recomputation.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32,
                 cache_dir: str = "outputs/eval_cache"):
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[InceptionFeatureExtractor] = None

    def _load_model(self) -> InceptionFeatureExtractor:
        if self._model is None:
            logger.info("Loading InceptionV3 for FID")
            self._model = InceptionFeatureExtractor(self.device)
        return self._model

    def _cache_path(self, image_dir: str, max_n: Optional[int]) -> Path:
        key = f"{image_dir}_{max_n}"
        return self.cache_dir / f"fid_{hash(key) & 0xffffffff:08x}.npz"

    def compute_stats(self, image_dir: str, max_images: Optional[int] = None,
                      use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        cp = self._cache_path(image_dir, max_images)
        if use_cache and cp.exists():
            data = np.load(str(cp))
            logger.info(f"Loaded cached FID stats: {cp}")
            return data["mu"], data["sigma"]
        imgs = _load_dir(image_dir, max_images)
        if len(imgs) < 2:
            raise ValueError(f"Need ≥2 images, got {len(imgs)}")
        logger.info(f"Computing Inception features for {len(imgs)} images")
        mu, sigma = _compute_stats(imgs, self._load_model(), self.batch_size)
        if use_cache:
            np.savez(str(cp), mu=mu, sigma=sigma)
        return mu, sigma

    def compute(self, real_dir: str, generated_dir: str,
                max_real: Optional[int] = None,
                max_generated: Optional[int] = None,
                use_real_cache: bool = True) -> float:
        mu_r, s_r = self.compute_stats(real_dir, max_real, use_real_cache)
        imgs_g = _load_dir(generated_dir, max_generated)
        if len(imgs_g) < 2:
            raise ValueError(f"Need ≥2 generated images, got {len(imgs_g)}")
        mu_g, s_g = _compute_stats(imgs_g, self._load_model(), self.batch_size)
        fid = _frechet(mu_r, s_r, mu_g, s_g)
        logger.info(f"FID = {fid:.4f}")
        return fid

    def compute_from_images(self, real: List[Image.Image],
                             generated: List[Image.Image]) -> float:
        m = self._load_model()
        mu_r, s_r = _compute_stats(real, m, self.batch_size)
        mu_g, s_g = _compute_stats(generated, m, self.batch_size)
        return _frechet(mu_r, s_r, mu_g, s_g)
