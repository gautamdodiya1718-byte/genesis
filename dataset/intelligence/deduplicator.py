"""
dataset/intelligence/deduplicator.py
--------------------------------------
Two-tier deduplication: pHash (fast) → CLIP+FAISS (semantic).
Tier 1 runs first to reduce set size before expensive Tier 2.
"""
from __future__ import annotations
import json, logging, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from core.image_utils import load_image, perceptual_hash, hash_distance

logger = logging.getLogger(__name__)
_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class DuplicateGroup:
    keep: str
    duplicates: List[str] = field(default_factory=list)
    method: str = "phash"
    similarity: float = 1.0


@dataclass
class DeduplicationReport:
    total_scanned: int = 0
    kept: int = 0
    removed: int = 0
    tier1_removed: int = 0
    tier2_removed: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Dedup | scanned={self.total_scanned} kept={self.kept} "
            f"removed={self.removed} (t1={self.tier1_removed} t2={self.tier2_removed}) "
            f"groups={len(self.duplicate_groups)} {self.duration_seconds:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "total_scanned": self.total_scanned, "kept": self.kept,
            "removed": self.removed, "tier1_removed": self.tier1_removed,
            "tier2_removed": self.tier2_removed,
            "n_groups": len(self.duplicate_groups),
            "duration_seconds": round(self.duration_seconds, 2),
            "groups": [
                {"keep": g.keep, "duplicates": g.duplicates,
                 "method": g.method, "similarity": round(g.similarity, 4)}
                for g in self.duplicate_groups
            ],
        }


class PHashDeduplicator:
    def __init__(self, threshold: float = 0.95):
        self._max_dist = 1.0 - threshold

    def compute_hashes(self, paths: List[str]) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for i, p in enumerate(paths):
            if i and i % 500 == 0:
                logger.debug(f"  pHash {i}/{len(paths)}")
            img = load_image(p)
            if img is not None:
                hashes[p] = perceptual_hash(img)
        return hashes

    def find_duplicates(self, hashes: Dict[str, str]) -> List[DuplicateGroup]:
        paths = list(hashes.keys())
        removed: Set[str] = set()
        groups: List[DuplicateGroup] = []
        for i, pi in enumerate(paths):
            if pi in removed:
                continue
            dups = [pj for pj in paths[i + 1:]
                    if pj not in removed
                    and hash_distance(hashes[pi], hashes[pj]) <= self._max_dist]
            for pj in dups:
                removed.add(pj)
            if dups:
                groups.append(DuplicateGroup(pi, dups, "phash", 1.0 - self._max_dist))
        return groups


class EmbeddingDeduplicator:
    """CLIP ViT-B/32 + FAISS flat IP for semantic duplicate detection."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 similarity_threshold: float = 0.97, batch_size: int = 32,
                 device: str = "cpu"):
        self.model_id = model_id
        self.threshold = similarity_threshold
        self.batch_size = batch_size
        self.device = device
        self._clip = self._proc = None

    def _load(self) -> None:
        if self._clip is not None:
            return
        from transformers import CLIPProcessor, CLIPModel
        logger.info(f"Loading CLIP: {self.model_id}")
        self._proc = CLIPProcessor.from_pretrained(self.model_id)
        self._clip = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._clip.eval()

    def embed(self, paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Returns (N, D) L2-normalised embeddings + valid paths."""
        self._load()
        import torch
        all_embs, valid = [], []
        for i in range(0, len(paths), self.batch_size):
            batch = paths[i:i + self.batch_size]
            imgs, bv = [], []
            for p in batch:
                img = load_image(p)
                if img is not None:
                    imgs.append(img); bv.append(p)
            if not imgs:
                continue
            inp = {k: v.to(self.device)
                   for k, v in self._proc(images=imgs, return_tensors="pt",
                                          padding=True).items()}
            with torch.no_grad():
                f = self._clip.get_image_features(**inp)
                f = f / f.norm(dim=-1, keepdim=True)
            all_embs.append(f.cpu().float().numpy())
            valid.extend(bv)
        return (np.vstack(all_embs) if all_embs
                else np.zeros((0, 512), dtype=np.float32)), valid

    def find_duplicates(self, embeddings: np.ndarray,
                        paths: List[str], k: int = 10) -> List[DuplicateGroup]:
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu")
        n, d = embeddings.shape
        if n < 2:
            return []
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        sims, idxs = index.search(embeddings, min(k + 1, n))
        removed: Set[int] = set()
        groups: List[DuplicateGroup] = []
        for i in range(n):
            if i in removed:
                continue
            dups = [(int(j), float(s)) for s, j in zip(sims[i], idxs[i])
                    if j != i and j not in removed and float(s) >= self.threshold]
            if dups:
                for j, _ in dups:
                    removed.add(j)
                groups.append(DuplicateGroup(
                    keep=paths[i], duplicates=[paths[j] for j, _ in dups],
                    method="embedding",
                    similarity=sum(s for _, s in dups) / len(dups)))
        return groups


class Deduplicator:
    """Unified two-tier deduplicator integrating pHash + CLIP+FAISS."""

    def __init__(self, phash_threshold: float = 0.95,
                 embedding_threshold: float = 0.97,
                 use_embedding_tier: bool = True,
                 device: str = "cpu", dry_run: bool = False):
        self.dry_run = dry_run
        self._t1 = PHashDeduplicator(phash_threshold)
        self._t2 = (EmbeddingDeduplicator(
            similarity_threshold=embedding_threshold, device=device)
            if use_embedding_tier else None)

    def run(self, image_dir: str,
            report_path: Optional[str] = None) -> DeduplicationReport:
        t0 = time.time()
        report = DeduplicationReport()
        paths = sorted(str(p) for p in Path(image_dir).iterdir()
                       if p.suffix.lower() in _EXT)
        report.total_scanned = len(paths)
        logger.info(f"Dedup: {len(paths)} images in {image_dir}")
        if len(paths) < 2:
            report.kept = len(paths)
            report.duration_seconds = time.time() - t0
            return report

        # Tier 1
        hashes = self._t1.compute_hashes(paths)
        t1_groups = self._t1.find_duplicates(hashes)
        t1_removed = {p for g in t1_groups for p in g.duplicates}
        report.duplicate_groups.extend(t1_groups)
        report.tier1_removed = len(t1_removed)
        logger.info(f"  pHash: {len(t1_removed)} dups")

        remaining = [p for p in paths if p not in t1_removed]

        # Tier 2
        t2_removed: Set[str] = set()
        if self._t2 and len(remaining) > 1:
            try:
                embs, valid = self._t2.embed(remaining)
                if len(valid) > 1:
                    t2_groups = self._t2.find_duplicates(embs, valid)
                    t2_removed = {p for g in t2_groups for p in g.duplicates}
                    report.duplicate_groups.extend(t2_groups)
                    report.tier2_removed = len(t2_removed)
                    logger.info(f"  Embedding: {len(t2_removed)} semantic dups")
            except ImportError as e:
                logger.warning(f"Tier 2 skipped: {e}")

        all_removed = t1_removed | t2_removed
        report.removed = len(all_removed)
        report.kept = report.total_scanned - report.removed
        report.duration_seconds = time.time() - t0

        if not self.dry_run:
            for p in all_removed:
                try:
                    os.remove(p)
                except OSError:
                    pass

        if report_path:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

        logger.info(report.summary())
        return report
