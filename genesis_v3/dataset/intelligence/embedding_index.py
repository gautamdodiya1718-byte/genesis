"""
dataset/intelligence/embedding_index.py
-----------------------------------------
CLIP image/text embedding index backed by FAISS.

Capabilities:
  - Batch embed images (CLIP ViT-B/32, CPU-compatible)
  - Persist to disk: FAISS binary + JSON metadata sidecar
  - Image → nearest image search
  - Text → nearest image search (cross-modal)
  - Semantic K-means clustering
  - Duplicate pair detection
  - Incremental add without full rebuild

On-disk layout:
  <index_dir>/
    embeddings.index  — FAISS IndexFlatIP
    id_map.json       — {faiss_pos: image_id}
    metadata.json     — {image_id: {...}}
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from core.image_utils import load_image

logger = logging.getLogger(__name__)
_DIM = 512  # CLIP ViT-B/32 output dimension


class EmbeddingIndex:
    """
    Persistent FAISS embedding index with image metadata.
    Uses IndexFlatIP (exact cosine search on L2-normalised vectors).
    """

    def __init__(self, index_dir: str,
                 model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu", batch_size: int = 32):
        self.index_dir  = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_id   = model_id
        self.device     = device
        self.batch_size = batch_size

        self._idx_path  = self.index_dir / "embeddings.index"
        self._map_path  = self.index_dir / "id_map.json"
        self._meta_path = self.index_dir / "metadata.json"

        self._faiss_idx = None
        self._id_map: Dict[int, str] = {}       # pos → image_id
        self._id_inv: Dict[str, int] = {}       # image_id → pos
        self._meta:   Dict[str, dict] = {}

        self._clip = self._proc = None
        self._load_from_disk()

    # ── CLIP ──────────────────────────────────────────────────

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        from transformers import CLIPProcessor, CLIPModel
        logger.info(f"Loading CLIP: {self.model_id}")
        self._proc = CLIPProcessor.from_pretrained(self.model_id)
        self._clip = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._clip.eval()

    def _embed_pil(self, images: list) -> np.ndarray:
        import torch
        self._load_clip()
        inp = {k: v.to(self.device)
               for k, v in self._proc(images=images, return_tensors="pt",
                                       padding=True).items()}
        with torch.no_grad():
            f = self._clip.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy()

    def _embed_text_batch(self, texts: List[str]) -> np.ndarray:
        import torch
        self._load_clip()
        inp = {k: v.to(self.device)
               for k, v in self._proc(text=texts, return_tensors="pt",
                                       padding=True, truncation=True).items()}
        with torch.no_grad():
            f = self._clip.get_text_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy()

    # ── FAISS ─────────────────────────────────────────────────

    def _init_faiss(self) -> None:
        if self._faiss_idx is None:
            try:
                import faiss
                self._faiss_idx = faiss.IndexFlatIP(_DIM)
            except ImportError:
                raise ImportError("pip install faiss-cpu")

    def _load_from_disk(self) -> None:
        try:
            import faiss
        except ImportError:
            logger.warning("faiss not available; index will be in-memory")
            return
        if self._idx_path.exists():
            self._faiss_idx = faiss.read_index(str(self._idx_path))
            logger.info(f"Loaded FAISS index: {self._faiss_idx.ntotal} vectors")
        if self._map_path.exists():
            raw = json.loads(self._map_path.read_text())
            self._id_map = {int(k): v for k, v in raw.items()}
            self._id_inv = {v: int(k) for k, v in self._id_map.items()}
        if self._meta_path.exists():
            self._meta = json.loads(self._meta_path.read_text())

    def _save_to_disk(self) -> None:
        import faiss
        if self._faiss_idx is not None:
            faiss.write_index(self._faiss_idx, str(self._idx_path))
        self._map_path.write_text(json.dumps(self._id_map, indent=2))
        self._meta_path.write_text(json.dumps(self._meta, indent=2))

    def _add_embedding(self, image_id: str, emb: np.ndarray,
                       metadata: Optional[dict]) -> None:
        self._init_faiss()
        pos = self._faiss_idx.ntotal
        self._faiss_idx.add(emb.reshape(1, -1))
        self._id_map[pos] = image_id
        self._id_inv[image_id] = pos
        if metadata:
            self._meta[image_id] = metadata

    # ── Public API ────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._faiss_idx.ntotal if self._faiss_idx else 0

    def add(self, image_id: str,
            image: Union[str, "Image.Image"],
            metadata: Optional[dict] = None) -> bool:
        """Add single image. Returns False if already present."""
        if image_id in self._id_inv:
            return False
        if isinstance(image, (str, Path)):
            pil = load_image(str(image))
            if pil is None:
                return False
        else:
            pil = image
        emb = self._embed_pil([pil])
        self._add_embedding(image_id, emb[0], metadata)
        return True

    def add_batch(self, items: List[Dict],
                  save_every: int = 500) -> Tuple[int, int]:
        """
        Each item: {"id": str, "image": path|PIL, "metadata": dict}.
        Returns (added, skipped).
        """
        self._init_faiss()
        added = skipped = 0
        buf_imgs, buf_ids, buf_meta = [], [], []

        def flush(buf_imgs, buf_ids, buf_meta):
            nonlocal added
            embs = self._embed_pil(buf_imgs)
            for iid, emb, meta in zip(buf_ids, embs, buf_meta):
                self._add_embedding(iid, emb, meta)
                added += 1

        for i, item in enumerate(items):
            iid = item["id"]
            if iid in self._id_inv:
                skipped += 1
                continue
            img = item["image"]
            pil = load_image(str(img)) if isinstance(img, (str, Path)) else img
            if pil is None:
                skipped += 1
                continue
            buf_imgs.append(pil)
            buf_ids.append(iid)
            buf_meta.append(item.get("metadata", {}))
            if len(buf_imgs) >= self.batch_size:
                flush(buf_imgs, buf_ids, buf_meta)
                buf_imgs, buf_ids, buf_meta = [], [], []
                if (i + 1) % save_every == 0:
                    self._save_to_disk()
                    logger.info(f"  Index: {added} added, total={self.size}")

        if buf_imgs:
            flush(buf_imgs, buf_ids, buf_meta)
        self._save_to_disk()
        logger.info(f"Batch add: +{added} (skipped={skipped}) total={self.size}")
        return added, skipped

    def search_by_image(self, image: Union[str, "Image.Image"],
                        k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """Find k nearest images by visual similarity."""
        pil = load_image(str(image)) if isinstance(image, (str, Path)) else image
        if pil is None:
            return []
        emb = self._embed_pil([pil])
        return self._search(emb, k, threshold)

    def search_by_text(self, text: str,
                       k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """Cross-modal: find k nearest images to a text query."""
        emb = self._embed_text_batch([text])
        return self._search(emb, k, threshold)

    def _search(self, emb: np.ndarray, k: int,
                threshold: float) -> List[Dict]:
        if not self._faiss_idx or self._faiss_idx.ntotal == 0:
            return []
        sims, idxs = self._faiss_idx.search(
            emb.reshape(1, -1), min(k, self._faiss_idx.ntotal))
        results = []
        for sim, pos in zip(sims[0], idxs[0]):
            sim = float(sim)
            if sim < threshold:
                continue
            iid = self._id_map.get(int(pos))
            if iid:
                results.append({"image_id": iid, "similarity": sim,
                                 "metadata": self._meta.get(iid, {})})
        return results

    def find_duplicates(self, similarity_threshold: float = 0.97,
                        k: int = 10) -> List[Tuple[str, str, float]]:
        """Return (id_a, id_b, similarity) pairs within the index."""
        if not self._faiss_idx or self._faiss_idx.ntotal < 2:
            return []
        n = self._faiss_idx.ntotal
        try:
            all_embs = np.zeros((n, _DIM), dtype=np.float32)
            for i in range(n):
                self._faiss_idx.reconstruct(i, all_embs[i])
        except Exception:
            logger.warning("Cannot reconstruct embeddings for dup detection")
            return []
        sims, idxs = self._faiss_idx.search(all_embs, min(k + 1, n))
        seen: set = set()
        pairs = []
        for i in range(n):
            for sim, j in zip(sims[i][1:], idxs[i][1:]):
                if float(sim) < similarity_threshold:
                    break
                key = tuple(sorted([i, int(j)]))
                if key not in seen:
                    seen.add(key)
                    ia, ib = self._id_map.get(i), self._id_map.get(int(j))
                    if ia and ib:
                        pairs.append((ia, ib, float(sim)))
        return pairs

    def cluster(self, n_clusters: int = 20) -> Dict[int, List[str]]:
        """K-means clustering. Returns {cluster_id: [image_ids]}."""
        import faiss
        n = self.size
        if n < n_clusters:
            return {0: list(self._id_map.values())}
        all_embs = np.zeros((n, _DIM), dtype=np.float32)
        for i in range(n):
            try:
                self._faiss_idx.reconstruct(i, all_embs[i])
            except Exception:
                pass
        km = faiss.Kmeans(_DIM, n_clusters, niter=20, verbose=False)
        km.train(all_embs)
        _, assignments = km.index.search(all_embs, 1)
        assignments = assignments.flatten()
        clusters: Dict[int, List[str]] = {c: [] for c in range(n_clusters)}
        for pos, cid in enumerate(assignments):
            iid = self._id_map.get(pos)
            if iid:
                clusters[int(cid)].append(iid)
        return clusters

    def stats(self) -> dict:
        return {"total_vectors": self.size, "index_dir": str(self.index_dir),
                "model_id": self.model_id, "dim": _DIM}
