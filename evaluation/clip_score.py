"""
evaluation/clip_score.py
--------------------------
CLIP similarity score between prompts and generated images.

CLIPScore measures how well a generated image matches its conditioning text.
Range: [0, 100] approximately. Higher = better prompt adherence.

Reference: Hessel et al., 2021 — CLIPScore: A Reference-Free Evaluation
Metric for Image Captioning.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from core.image_utils import load_image

logger = logging.getLogger(__name__)


class CLIPScorer:
    """
    CLIP-based prompt-image alignment scorer.

    Usage:
        scorer = CLIPScorer(device="cpu")
        score = scorer.score("a mountain at sunset", image)
        scores = scorer.score_batch(prompts, images)
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu", batch_size: int = 32):
        self.model_id  = model_id
        self.device    = device
        self.batch_size = batch_size
        self._clip = self._proc = None

    def _load(self) -> None:
        if self._clip is not None:
            return
        from transformers import CLIPProcessor, CLIPModel
        logger.info(f"Loading CLIP scorer: {self.model_id}")
        self._proc = CLIPProcessor.from_pretrained(self.model_id)
        self._clip = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._clip.eval()

    def _embed_images(self, images: List[Image.Image]) -> np.ndarray:
        self._load()
        inp = {k: v.to(self.device) for k, v in
               self._proc(images=images, return_tensors="pt",
                          padding=True).items()}
        with torch.no_grad():
            f = self._clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        self._load()
        inp = {k: v.to(self.device) for k, v in
               self._proc(text=texts, return_tensors="pt",
                          padding=True, truncation=True).items()}
        with torch.no_grad():
            f = self._clip.get_text_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy()

    def score(self, prompt: str,
              image: Union[str, Path, Image.Image]) -> float:
        """
        Single prompt-image CLIP score.
        Returns cosine similarity in [0, 1] scaled to [0, 100].
        """
        if isinstance(image, (str, Path)):
            pil = load_image(str(image))
            if pil is None:
                return 0.0
        else:
            pil = image
        img_emb = self._embed_images([pil])    # (1, D)
        txt_emb = self._embed_texts([prompt])  # (1, D)
        sim = float(np.dot(img_emb[0], txt_emb[0]))
        # Scale cosine [-1,1] → [0,100], clamp to [0,100]
        return float(np.clip(sim * 100.0, 0.0, 100.0))

    def score_batch(
        self,
        prompts: List[str],
        images: List[Union[str, Path, Image.Image]],
    ) -> List[float]:
        """
        Batch scoring. prompts[i] paired with images[i].
        Returns list of scores in same order.
        """
        assert len(prompts) == len(images), "prompts and images must be same length"
        pils = []
        valid_idx = []
        for i, img in enumerate(images):
            if isinstance(img, (str, Path)):
                pil = load_image(str(img))
            else:
                pil = img
            if pil is not None:
                pils.append(pil)
                valid_idx.append(i)

        scores = [0.0] * len(prompts)
        if not pils:
            return scores

        # Process in batches
        all_img_embs = []
        for i in range(0, len(pils), self.batch_size):
            all_img_embs.append(self._embed_images(pils[i:i + self.batch_size]))
        img_embs = np.vstack(all_img_embs)  # (N_valid, D)

        valid_prompts = [prompts[i] for i in valid_idx]
        all_txt_embs = []
        for i in range(0, len(valid_prompts), self.batch_size):
            all_txt_embs.append(self._embed_texts(valid_prompts[i:i + self.batch_size]))
        txt_embs = np.vstack(all_txt_embs)  # (N_valid, D)

        # Pairwise diagonal cosine similarity
        sims = np.einsum("nd,nd->n", img_embs, txt_embs)  # (N_valid,)
        for rank, idx in enumerate(valid_idx):
            scores[idx] = float(np.clip(sims[rank] * 100.0, 0.0, 100.0))

        return scores

    def score_image_dir(
        self,
        prompt: str,
        image_dir: str,
        max_images: Optional[int] = None,
    ) -> Dict:
        """Score all images in a directory against a single prompt."""
        _EXT = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(p for p in Path(image_dir).iterdir()
                       if p.suffix.lower() in _EXT)
        if max_images:
            paths = paths[:max_images]
        prompts_rep = [prompt] * len(paths)
        scores = self.score_batch(prompts_rep, [str(p) for p in paths])
        return {
            "prompt": prompt,
            "n_images": len(paths),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score":  float(np.std(scores))  if scores else 0.0,
            "min_score":  float(np.min(scores))  if scores else 0.0,
            "max_score":  float(np.max(scores))  if scores else 0.0,
            "scores": [{"path": str(p), "score": s}
                       for p, s in zip(paths, scores)],
        }
