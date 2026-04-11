"""
core/model_manager.py
----------------------
Production model manager for the Genesis AI Studio.

Responsibilities:
  - Download pretrained models from HuggingFace Hub
  - Verify model integrity via SHA-256 checksums
  - Manage local model cache with disk usage tracking
  - Load models dynamically with in-process caching
  - Track download status, model metadata, last-used timestamps
  - Auto-evict least-recently-used models from memory on RAM pressure

Supported models:
  Stable Diffusion 1.5       runwayml/stable-diffusion-v1-5
  Stable Diffusion XL Base   stabilityai/stable-diffusion-xl-base-1.0
  LCM Dreamshaper v7         SimianLuo/LCM_Dreamshaper_v7
  CLIP ViT-L/14              openai/clip-vit-large-patch14
  CLIP ViT-B/32              openai/clip-vit-base-patch32
  BLIP                       Salesforce/blip-image-captioning-base
  VAE ft-mse                 stabilityai/sd-vae-ft-mse

Cache layout:
  model_cache/
    registry.json           -- download records + hashes
    usage.json              -- access timestamps + hit counts
    runwayml--stable-diffusion-v1-5/
      model_index.json      -- diffusers pipeline config
      unet/  vae/  ...
    openai--clip-vit-large-patch14/
      ...
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Model catalog ─────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Descriptor for a downloadable pretrained model."""
    hub_id:     str
    model_type: str          # diffusion | sdxl | lcm | clip | blip | vae | snapshot
    size_gb:    float
    description: str = ""
    sha256_hint: Optional[str] = None   # optional partial hash for quick verify
    requires_accept_tos: bool = False    # some models need HF login
    min_ram_gb: float = 0.0
    tags: List[str] = field(default_factory=list)

    @property
    def safe_name(self) -> str:
        return self.hub_id.replace("/", "--").replace(":", "_")


_CATALOG: Dict[str, ModelSpec] = {
    # ── Diffusion pipelines ────────────────────────────────────
    "sd15": ModelSpec(
        hub_id="runwayml/stable-diffusion-v1-5",
        model_type="diffusion",
        size_gb=4.2,
        min_ram_gb=4.0,
        description="Stable Diffusion 1.5 — general purpose text-to-image",
        tags=["txt2img", "img2img", "production"],
    ),
    "sdxl": ModelSpec(
        hub_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="sdxl",
        size_gb=13.5,
        min_ram_gb=12.0,
        description="Stable Diffusion XL — high quality, high RAM",
        tags=["txt2img", "high_quality"],
    ),
    "lcm": ModelSpec(
        hub_id="SimianLuo/LCM_Dreamshaper_v7",
        model_type="lcm",
        size_gb=3.8,
        min_ram_gb=4.0,
        description="LCM Dreamshaper v7 — 4-step fast CPU generation",
        tags=["txt2img", "img2img", "fast", "cpu_optimized"],
    ),
    "lcm_sdxl": ModelSpec(
        hub_id="latent-consistency/lcm-sdxl",
        model_type="lcm",
        size_gb=6.7,
        min_ram_gb=10.0,
        description="LCM-LoRA for SDXL — fast high-quality generation",
        tags=["txt2img", "fast", "high_quality"],
    ),
    # ── VAE ───────────────────────────────────────────────────
    "vae_ft_mse": ModelSpec(
        hub_id="stabilityai/sd-vae-ft-mse",
        model_type="vae",
        size_gb=0.3,
        description="Fine-tuned VAE for SD — improved reconstruction quality",
        tags=["vae", "component"],
    ),
    # ── CLIP encoders ─────────────────────────────────────────
    "clip_vit_l": ModelSpec(
        hub_id="openai/clip-vit-large-patch14",
        model_type="clip",
        size_gb=0.9,
        description="CLIP ViT-L/14 — used for aesthetic scoring and embeddings",
        tags=["clip", "embedding", "scoring"],
    ),
    "clip_vit_b": ModelSpec(
        hub_id="openai/clip-vit-base-patch32",
        model_type="clip",
        size_gb=0.35,
        description="CLIP ViT-B/32 — lightweight CLIP for dedup and CLIP score",
        tags=["clip", "embedding", "fast"],
    ),
    # ── Captioning ────────────────────────────────────────────
    "blip_base": ModelSpec(
        hub_id="Salesforce/blip-image-captioning-base",
        model_type="blip",
        size_gb=0.9,
        description="BLIP base — fast image captioning for dataset building",
        tags=["captioning"],
    ),
    "blip_large": ModelSpec(
        hub_id="Salesforce/blip-image-captioning-large",
        model_type="blip",
        size_gb=1.8,
        description="BLIP large — higher quality captioning",
        tags=["captioning", "high_quality"],
    ),
    "vit_gpt2": ModelSpec(
        hub_id="nlpconnect/vit-gpt2-image-captioning",
        model_type="blip",
        size_gb=0.4,
        description="ViT-GPT2 — smallest captioning model",
        tags=["captioning", "fast"],
    ),
}


# ── Download record ───────────────────────────────────────────

@dataclass
class DownloadRecord:
    model_key:    str
    hub_id:       str
    model_type:   str
    local_path:   str
    downloaded_at: float
    size_bytes:   int = 0
    sha256:       Optional[str] = None
    verified:     bool = False
    download_secs: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "DownloadRecord":
        return cls(**{k: d.get(k) for k in
                      ("model_key","hub_id","model_type","local_path",
                       "downloaded_at","size_bytes","sha256","verified","download_secs")})


# ── Usage tracking ────────────────────────────────────────────

@dataclass
class UsageRecord:
    model_key:  str
    last_used:  float = 0.0
    hit_count:  int = 0
    load_count: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── In-process loaded model cache ────────────────────────────

class _LoadedModelCache:
    """LRU in-process cache for loaded PyTorch/diffusers models."""

    def __init__(self, max_models: int = 2, max_ram_gb: float = 12.0):
        self.max_models  = max_models
        self.max_ram_gb  = max_ram_gb
        self._cache: Dict[str, Tuple[Any, float, float]] = {}  # key→(model, loaded_at, last_used)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                model, loaded_at, _ = self._cache[key]
                self._cache[key] = (model, loaded_at, time.time())
                return model
        return None

    def put(self, key: str, model: Any, size_gb: float = 1.0) -> None:
        with self._lock:
            # Evict if over capacity
            while len(self._cache) >= self.max_models and key not in self._cache:
                self._evict_lru()
            self._cache[key] = (model, time.time(), time.time())

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k][2])
        logger.info(f"Evicting model from cache: {lru_key}")
        del self._cache[lru_key]
        gc.collect()

    def evict(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                gc.collect()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            gc.collect()

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())

    def loaded_count(self) -> int:
        with self._lock:
            return len(self._cache)


# ── ModelManager ──────────────────────────────────────────────

class ModelManager:
    """
    Production model manager for Genesis AI Studio.

    Handles download, verification, disk tracking, and in-process caching
    for all pretrained models used across the platform.

    Usage:
        mm = ModelManager("model_cache")

        # Download + cache to disk
        path = mm.ensure("sd15")

        # Load into memory (returns diffusers pipeline / transformers model)
        pipe = mm.load("sd15")

        # Unload from memory
        mm.unload("sd15")

        # Check what's installed
        mm.print_status()
    """

    def __init__(
        self,
        cache_dir: str = "model_cache",
        max_loaded_models: int = 2,
        max_ram_gb: float = 12.0,
    ):
        self.cache_dir   = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._reg_path   = self.cache_dir / "registry.json"
        self._usage_path = self.cache_dir / "usage.json"
        self._registry:  Dict[str, DownloadRecord] = {}
        self._usage:     Dict[str, UsageRecord]    = {}
        self._model_cache = _LoadedModelCache(max_loaded_models, max_ram_gb)
        self._dl_lock    = threading.Lock()

        self._load_registry()
        self._load_usage()

    # ── Persistence ───────────────────────────────────────────

    def _load_registry(self) -> None:
        if self._reg_path.exists():
            try:
                data = json.loads(self._reg_path.read_text())
                for k, v in data.items():
                    self._registry[k] = DownloadRecord.from_dict(v)
            except Exception as e:
                logger.warning(f"Registry load failed: {e}")

    def _save_registry(self) -> None:
        self._reg_path.write_text(
            json.dumps({k: v.to_dict() for k, v in self._registry.items()}, indent=2)
        )

    def _load_usage(self) -> None:
        if self._usage_path.exists():
            try:
                data = json.loads(self._usage_path.read_text())
                for k, v in data.items():
                    self._usage[k] = UsageRecord(**v)
            except Exception:
                pass

    def _save_usage(self) -> None:
        self._usage_path.write_text(
            json.dumps({k: v.to_dict() for k, v in self._usage.items()}, indent=2)
        )

    def _touch_usage(self, key: str) -> None:
        u = self._usage.setdefault(key, UsageRecord(model_key=key))
        u.last_used = time.time()
        u.hit_count += 1
        self._save_usage()

    # ── Catalog ───────────────────────────────────────────────

    def catalog(self) -> Dict[str, ModelSpec]:
        return _CATALOG.copy()

    def get_spec(self, model_key: str) -> Optional[ModelSpec]:
        return _CATALOG.get(model_key)

    def register_custom(self, key: str, spec: ModelSpec) -> None:
        """Register a custom model not in the built-in catalog."""
        _CATALOG[key] = spec
        logger.info(f"Custom model registered: {key} ({spec.hub_id})")

    # ── Download ──────────────────────────────────────────────

    def is_cached(self, model_key: str) -> bool:
        """True if model is downloaded to local disk."""
        if model_key in self._registry:
            p = Path(self._registry[model_key].local_path)
            return p.exists() and any(p.iterdir())
        spec = _CATALOG.get(model_key)
        if spec:
            p = self.cache_dir / spec.safe_name
            return p.exists() and any(p.iterdir())
        return False

    def ensure(
        self,
        model_key: str,
        force: bool = False,
        verify_after: bool = True,
    ) -> str:
        """
        Ensure model is downloaded to disk.
        Returns local directory path.

        Args:
            model_key:    Key from catalog (e.g. "sd15") OR raw hub_id
            force:        Re-download even if cached
            verify_after: Compute + store directory hash after download
        """
        # Allow raw hub_id as model_key
        spec = _CATALOG.get(model_key)
        if spec is None:
            # treat as direct hub_id
            spec = ModelSpec(
                hub_id=model_key, model_type="snapshot",
                size_gb=1.0, description="custom model"
            )
            effective_key = model_key.replace("/", "--")
        else:
            effective_key = model_key

        local_dir = self.cache_dir / spec.safe_name

        if not force and self.is_cached(effective_key):
            self._touch_usage(effective_key)
            return str(local_dir)

        # Disk space check
        if not self._check_disk(spec.size_gb * 1.1):
            raise RuntimeError(
                f"Insufficient disk space for {model_key} "
                f"({spec.size_gb:.1f}GB needed)"
            )

        with self._dl_lock:
            # Double-check after acquiring lock
            if not force and self.is_cached(effective_key):
                return str(local_dir)

            local_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            logger.info(f"Downloading {model_key} ({spec.hub_id}) → {local_dir}")

            try:
                self._download(spec, str(local_dir))
            except Exception as e:
                shutil.rmtree(str(local_dir), ignore_errors=True)
                logger.error(f"Download failed [{model_key}]: {e}")
                raise

            elapsed   = time.time() - t0
            size_bytes = self._dir_size(local_dir)
            sha256     = self._hash_dir(local_dir) if verify_after else None

            rec = DownloadRecord(
                model_key=effective_key,
                hub_id=spec.hub_id,
                model_type=spec.model_type,
                local_path=str(local_dir),
                downloaded_at=time.time(),
                size_bytes=size_bytes,
                sha256=sha256,
                verified=verify_after,
                download_secs=elapsed,
            )
            self._registry[effective_key] = rec
            self._save_registry()
            logger.info(
                f"Downloaded {model_key} in {elapsed:.0f}s "
                f"({size_bytes/1024**3:.2f}GB)"
            )

        return str(local_dir)

    def _download(self, spec: ModelSpec, local_dir: str) -> None:
        """Dispatch to appropriate downloader based on model type."""
        import torch

        if spec.model_type in ("diffusion", "lcm"):
            self._dl_diffusion(spec.hub_id, local_dir)
        elif spec.model_type == "sdxl":
            self._dl_sdxl(spec.hub_id, local_dir)
        elif spec.model_type == "vae":
            self._dl_vae(spec.hub_id, local_dir)
        elif spec.model_type == "clip":
            self._dl_clip(spec.hub_id, local_dir)
        elif spec.model_type == "blip":
            self._dl_blip(spec.hub_id, local_dir)
        else:
            self._dl_snapshot(spec.hub_id, local_dir)

    def _dl_diffusion(self, hub_id: str, local: str) -> None:
        import torch
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            hub_id, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False,
        )
        pipe.save_pretrained(local)
        del pipe
        gc.collect()

    def _dl_sdxl(self, hub_id: str, local: str) -> None:
        import torch
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            hub_id, torch_dtype=torch.float32, use_safetensors=True,
        )
        pipe.save_pretrained(local)
        del pipe
        gc.collect()

    def _dl_vae(self, hub_id: str, local: str) -> None:
        import torch
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(hub_id, torch_dtype=torch.float32)
        vae.save_pretrained(local)
        del vae

    def _dl_clip(self, hub_id: str, local: str) -> None:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(hub_id)
        proc  = CLIPProcessor.from_pretrained(hub_id)
        model.save_pretrained(local)
        proc.save_pretrained(local)
        del model

    def _dl_blip(self, hub_id: str, local: str) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        try:
            model = BlipForConditionalGeneration.from_pretrained(hub_id)
            proc  = BlipProcessor.from_pretrained(hub_id)
            model.save_pretrained(local)
            proc.save_pretrained(local)
            del model
        except Exception:
            self._dl_snapshot(hub_id, local)

    def _dl_snapshot(self, hub_id: str, local: str) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=hub_id, local_dir=local)

    # ── Verification ──────────────────────────────────────────

    def verify(self, model_key: str) -> bool:
        """
        Verify model integrity by recomputing directory hash
        and comparing against stored hash.
        Returns True if verified or no prior hash stored.
        """
        rec = self._registry.get(model_key)
        if rec is None:
            logger.warning(f"verify: {model_key} not in registry")
            return False

        local = Path(rec.local_path)
        if not local.exists():
            logger.error(f"verify: {model_key} path missing: {local}")
            return False

        current_hash = self._hash_dir(local)
        if rec.sha256 is None:
            rec.sha256   = current_hash
            rec.verified = True
            self._save_registry()
            return True

        ok = current_hash == rec.sha256
        if ok:
            rec.verified = True
            logger.info(f"verify: {model_key} ✓")
        else:
            logger.error(
                f"verify: {model_key} MISMATCH "
                f"stored={rec.sha256[:16]} current={current_hash[:16]}"
            )
        self._save_registry()
        return ok

    def _hash_dir(self, path: Path) -> str:
        """Compute SHA-256 of all files in directory (sorted for determinism)."""
        h = hashlib.sha256()
        for fp in sorted(path.rglob("*")):
            if fp.is_file():
                h.update(fp.name.encode())
                with open(fp, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
        return h.hexdigest()

    # ── Load into memory ──────────────────────────────────────

    def load(
        self,
        model_key: str,
        device: str = "cpu",
        dtype_fp16: bool = False,
        force_reload: bool = False,
    ) -> Any:
        """
        Load model into memory, using in-process cache.
        Downloads first if not on disk.

        Returns the loaded pipeline / model object.
        """
        if not force_reload:
            cached = self._model_cache.get(model_key)
            if cached is not None:
                self._touch_usage(model_key)
                return cached

        local_path = self.ensure(model_key)
        spec = _CATALOG.get(model_key)
        model_type = spec.model_type if spec else "snapshot"

        logger.info(f"Loading {model_key} into memory ({device})")
        t0 = time.time()

        model = self._load_model(model_key, local_path, model_type, device, dtype_fp16)
        self._model_cache.put(model_key, model,
                               size_gb=spec.size_gb if spec else 1.0)

        u = self._usage.setdefault(model_key, UsageRecord(model_key=model_key))
        u.load_count += 1
        u.last_used = time.time()
        self._save_usage()

        logger.info(f"Loaded {model_key} in {time.time()-t0:.1f}s")
        return model

    def _load_model(
        self,
        key: str,
        local_path: str,
        model_type: str,
        device: str,
        dtype_fp16: bool,
    ) -> Any:
        import torch
        dtype = torch.float16 if (dtype_fp16 and device != "cpu") else torch.float32

        if model_type in ("diffusion", "lcm"):
            from diffusers import DiffusionPipeline, LCMScheduler
            pipe = DiffusionPipeline.from_pretrained(
                local_path, torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False,
            ).to(device)
            if model_type == "lcm":
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            # CPU optimizations
            try: pipe.enable_attention_slicing(1)
            except Exception: pass
            try: pipe.enable_vae_slicing()
            except Exception: pass
            return pipe

        elif model_type == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                local_path, torch_dtype=dtype, use_safetensors=True,
            ).to(device)
            try: pipe.enable_attention_slicing(1)
            except Exception: pass
            return pipe

        elif model_type == "vae":
            from diffusers import AutoencoderKL
            return AutoencoderKL.from_pretrained(local_path, torch_dtype=dtype).to(device)

        elif model_type == "clip":
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(local_path).to(device)
            proc  = CLIPProcessor.from_pretrained(local_path)
            return {"model": model, "processor": proc}

        elif model_type == "blip":
            from transformers import BlipForConditionalGeneration, BlipProcessor
            model = BlipForConditionalGeneration.from_pretrained(local_path).to(device)
            proc  = BlipProcessor.from_pretrained(local_path)
            return {"model": model, "processor": proc}

        else:
            # Generic: return path so caller can load themselves
            return local_path

    def unload(self, model_key: str) -> None:
        """Remove model from in-process cache, freeing RAM."""
        self._model_cache.evict(model_key)
        logger.info(f"Unloaded {model_key} from memory")

    def unload_all(self) -> None:
        self._model_cache.clear()

    # ── Disk management ───────────────────────────────────────

    def get_path(self, model_key: str) -> Optional[str]:
        """Return local path if model is on disk, else None."""
        if model_key in self._registry:
            p = self._registry[model_key].local_path
            return p if Path(p).exists() else None
        spec = _CATALOG.get(model_key)
        if spec:
            p = self.cache_dir / spec.safe_name
            return str(p) if (p.exists() and any(p.iterdir())) else None
        return None

    def delete(self, model_key: str, confirm: bool = False) -> bool:
        """
        Delete model from disk. Requires confirm=True.
        Also unloads from memory if loaded.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to delete a model")
        self.unload(model_key)
        rec = self._registry.get(model_key)
        if rec:
            p = Path(rec.local_path)
            if p.exists():
                shutil.rmtree(str(p))
                logger.info(f"Deleted {model_key} ({rec.size_bytes/1024**3:.2f}GB freed)")
            del self._registry[model_key]
            self._save_registry()
            return True
        return False

    def disk_usage(self) -> Dict[str, Any]:
        """Return disk usage summary for model cache."""
        total_bytes = 0
        per_model   = {}
        for key, rec in self._registry.items():
            p = Path(rec.local_path)
            if p.exists():
                sz = self._dir_size(p)
                per_model[key] = {
                    "size_gb": round(sz / 1024**3, 2),
                    "path": rec.local_path,
                    "downloaded_at": rec.downloaded_at,
                    "verified": rec.verified,
                }
                total_bytes += sz

        free  = shutil.disk_usage(self.cache_dir).free
        total = shutil.disk_usage(self.cache_dir).total
        return {
            "cache_dir":      str(self.cache_dir),
            "total_used_gb":  round(total_bytes / 1024**3, 2),
            "disk_free_gb":   round(free / 1024**3, 2),
            "disk_total_gb":  round(total / 1024**3, 2),
            "n_models":       len(per_model),
            "models":         per_model,
        }

    def _dir_size(self, path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _check_disk(self, required_gb: float) -> bool:
        free_gb = shutil.disk_usage(self.cache_dir).free / 1024**3
        if free_gb < required_gb:
            logger.warning(f"Low disk: {free_gb:.1f}GB free, {required_gb:.1f}GB needed")
            return False
        return True

    # ── Status / reporting ────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        loaded  = self._model_cache.keys()
        on_disk = [k for k in self._registry if self.is_cached(k)]
        return {
            "loaded_in_memory": loaded,
            "cached_on_disk":   on_disk,
            "n_loaded":         len(loaded),
            "n_on_disk":        len(on_disk),
            "disk_usage":       self.disk_usage(),
        }

    def print_status(self) -> None:
        disk = self.disk_usage()
        loaded = set(self._model_cache.keys())
        print(f"\n{'='*65}")
        print(f"  Genesis Model Manager  |  cache: {self.cache_dir}")
        print(f"  Disk: {disk['total_used_gb']:.1f}GB used  "
              f"{disk['disk_free_gb']:.1f}GB free")
        print(f"{'='*65}")
        print(f"  {'KEY':<18} {'STATUS':<12} {'SIZE':>6}  DESCRIPTION")
        print(f"  {'-'*60}")
        for key, spec in _CATALOG.items():
            on_disk  = self.is_cached(key)
            in_mem   = key in loaded
            status   = "MEM+DISK" if (on_disk and in_mem) else \
                       "DISK"     if on_disk else \
                       "IN MEM"   if in_mem  else "not downloaded"
            disk_sz  = disk["models"].get(key, {}).get("size_gb", spec.size_gb)
            print(f"  {key:<18} {status:<12} {disk_sz:>5.1f}G  {spec.description[:35]}")
        print()

    # ── Convenience context manager ───────────────────────────

    @contextmanager
    def loaded(self, model_key: str, device: str = "cpu"):
        """Context manager that loads model, yields it, then unloads."""
        model = self.load(model_key, device=device)
        try:
            yield model
        finally:
            self.unload(model_key)

    # ── Bulk operations ───────────────────────────────────────

    def download_all(self, tags: Optional[List[str]] = None) -> List[str]:
        """
        Download all catalog models (or filtered by tags).
        Returns list of successfully downloaded keys.
        """
        targets = [
            k for k, s in _CATALOG.items()
            if tags is None or any(t in s.tags for t in tags)
        ]
        succeeded = []
        for key in targets:
            try:
                self.ensure(key)
                succeeded.append(key)
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
        return succeeded

    def verify_all(self) -> Dict[str, bool]:
        """Verify all downloaded models. Returns {key: ok} dict."""
        results = {}
        for key in list(self._registry.keys()):
            results[key] = self.verify(key)
        return results

    def cleanup_unregistered(self) -> int:
        """
        Remove directories in cache_dir that are not in registry.
        Returns number of directories removed.
        """
        known_paths = {Path(r.local_path) for r in self._registry.values()}
        removed = 0
        for d in self.cache_dir.iterdir():
            if d.is_dir() and d not in known_paths and d.name not in (".", ".."):
                if d.suffix not in (".json", ".lock"):
                    logger.warning(f"Removing unregistered cache dir: {d}")
                    shutil.rmtree(str(d))
                    removed += 1
        return removed
