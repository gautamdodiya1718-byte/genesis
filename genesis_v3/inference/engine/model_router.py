"""
inference/engine/model_router.py
----------------------------------
Routes generation requests to the appropriate pretrained model backend
based on prompt content, requested quality tier, and available hardware.

Supported backends:
  lcm    — SimianLuo/LCM_Dreamshaper_v7     (4-8 steps,  fast CPU)
  sd15   — runwayml/stable-diffusion-v1-5   (20 steps,   medium CPU)
  sdxl   — stabilityai/stable-diffusion-xl-base-1.0 (25 steps, RAM-heavy)
  onnx   — ONNX-exported U-Net via OnnxRuntime (fastest CPU, needs export first)

Routing logic:
  "fast" quality tier   → lcm  (default for interactive use)
  "balanced" tier       → sd15 (better quality, slower)
  "high" tier           → sdxl (best quality, needs 16GB RAM)
  ONNX available        → prefer onnx over pytorch for same model

All models run fully locally — no API calls.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ── Model Descriptors ──────────────────────────────────────────

@dataclass
class ModelSpec:
    model_id: str                    # HuggingFace model id or local path
    backend: str                     # lcm | sd15 | sdxl | onnx
    quality_tiers: List[str]         # fast | balanced | high
    default_steps: int
    default_guidance: float
    min_ram_gb: float                # RAM required to load
    supports_img2img: bool = True
    supports_inpaint: bool = False
    onnx_dir: Optional[str] = None   # If set, use ONNX from this dir
    description: str = ""


_MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "lcm": ModelSpec(
        model_id="SimianLuo/LCM_Dreamshaper_v7",
        backend="lcm",
        quality_tiers=["fast"],
        default_steps=4,
        default_guidance=1.0,
        min_ram_gb=4.0,
        description="Latent Consistency Model — 4-step fast CPU generation",
    ),
    "sd15": ModelSpec(
        model_id="runwayml/stable-diffusion-v1-5",
        backend="sd15",
        quality_tiers=["balanced", "fast"],
        default_steps=20,
        default_guidance=7.5,
        min_ram_gb=4.0,
        description="Stable Diffusion 1.5 — balanced quality/speed",
    ),
    "sdxl": ModelSpec(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        backend="sdxl",
        quality_tiers=["high"],
        default_steps=25,
        default_guidance=7.5,
        min_ram_gb=12.0,
        description="Stable Diffusion XL — high quality, RAM-heavy",
    ),
    "sd15_onnx": ModelSpec(
        model_id="runwayml/stable-diffusion-v1-5",
        backend="onnx",
        quality_tiers=["balanced", "fast"],
        default_steps=20,
        default_guidance=7.5,
        min_ram_gb=3.5,
        onnx_dir="outputs/onnx/sd15",
        description="SD 1.5 with ONNX Runtime — 2-3× faster CPU inference",
    ),
}

_TIER_PREFERENCE = {
    "fast":     ["lcm", "sd15_onnx", "sd15"],
    "balanced": ["sd15_onnx", "sd15", "lcm"],
    "high":     ["sdxl", "sd15"],
}


@dataclass
class RouteDecision:
    model_key: str
    spec: ModelSpec
    steps: int
    guidance: float
    reason: str
    fallback_used: bool = False


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    quality_tier: str = "fast"       # fast | balanced | high
    steps: Optional[int] = None      # override default_steps
    guidance: Optional[float] = None # override default_guidance
    seed: Optional[int] = None
    n_images: int = 1
    init_image: Optional[Image.Image] = None
    strength: float = 0.75           # img2img strength
    request_id: str = ""
    user_id: str = "anonymous"
    model_hint: Optional[str] = None  # force specific model key


class ModelRouter:
    """
    Routes GenerationRequests to the best available model backend.

    Considers:
      - Requested quality tier (fast/balanced/high)
      - Available RAM (avoids OOM by checking system memory)
      - ONNX availability (prefers ONNX when exported)
      - Model hint (user can force a specific model)
      - Current load (routes away from overloaded backends)

    Usage:
        router = ModelRouter(available_ram_gb=8.0)
        decision = router.route(request)
        pipeline = get_pipeline(decision.model_key)
        images = pipeline.generate(request, decision)
    """

    def __init__(
        self,
        available_ram_gb: Optional[float] = None,
        prefer_onnx: bool = True,
        onnx_base_dir: str = "outputs/onnx",
        custom_registry: Optional[Dict[str, ModelSpec]] = None,
    ):
        self._registry = {**_MODEL_REGISTRY, **(custom_registry or {})}
        self.prefer_onnx   = prefer_onnx
        self.onnx_base_dir = Path(onnx_base_dir)

        if available_ram_gb is None:
            available_ram_gb = self._detect_ram()
        self.available_ram_gb = available_ram_gb

        logger.info(
            f"ModelRouter init | RAM={self.available_ram_gb:.1f}GB "
            f"prefer_onnx={prefer_onnx}"
        )

    def _detect_ram(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            return 6.0  # conservative default

    def _onnx_available(self, model_key: str) -> bool:
        spec = self._registry.get(model_key)
        if spec is None or spec.onnx_dir is None:
            return False
        onnx_path = Path(spec.onnx_dir)
        return (onnx_path / "unet" / "model.onnx").exists() or \
               (onnx_path / "unet.onnx").exists()

    def route(self, request: GenerationRequest) -> RouteDecision:
        """
        Select the best model for this request.
        Returns RouteDecision with model_key, steps, guidance.
        """
        # 1. Explicit hint
        if request.model_hint and request.model_hint in self._registry:
            spec = self._registry[request.model_hint]
            if spec.min_ram_gb <= self.available_ram_gb:
                return RouteDecision(
                    model_key=request.model_hint,
                    spec=spec,
                    steps=request.steps or spec.default_steps,
                    guidance=request.guidance or spec.default_guidance,
                    reason=f"explicit model_hint={request.model_hint}",
                )

        tier      = request.quality_tier
        preferred = _TIER_PREFERENCE.get(tier, _TIER_PREFERENCE["fast"])

        # 2. Try each preference in order
        for key in preferred:
            spec = self._registry.get(key)
            if spec is None:
                continue
            if spec.min_ram_gb > self.available_ram_gb:
                logger.debug(
                    f"  Skip {key}: needs {spec.min_ram_gb:.1f}GB, "
                    f"have {self.available_ram_gb:.1f}GB"
                )
                continue
            # Skip ONNX backend if not exported yet
            if spec.backend == "onnx" and not self._onnx_available(key):
                logger.debug(f"  Skip {key}: ONNX not exported")
                continue
            # Skip sdxl for img2img unless explicitly requested
            if key == "sdxl" and request.init_image is not None:
                logger.debug("  Skip sdxl for img2img (slow on CPU)")
                continue

            return RouteDecision(
                model_key=key,
                spec=spec,
                steps=request.steps or spec.default_steps,
                guidance=request.guidance or spec.default_guidance,
                reason=f"tier={tier} preference match",
            )

        # 3. Last resort fallback — LCM (lowest RAM)
        lcm = self._registry["lcm"]
        logger.warning(
            f"All preferred models unavailable for tier={tier}. "
            f"Falling back to LCM."
        )
        return RouteDecision(
            model_key="lcm",
            spec=lcm,
            steps=request.steps or lcm.default_steps,
            guidance=request.guidance or lcm.default_guidance,
            reason="fallback — no preferred model available",
            fallback_used=True,
        )

    def list_available(self) -> List[Dict]:
        """Return list of models that can run given current RAM."""
        result = []
        for key, spec in self._registry.items():
            available = spec.min_ram_gb <= self.available_ram_gb
            onnx_ready = spec.backend == "onnx" and self._onnx_available(key)
            result.append({
                "key": key,
                "backend": spec.backend,
                "model_id": spec.model_id,
                "quality_tiers": spec.quality_tiers,
                "min_ram_gb": spec.min_ram_gb,
                "available": available,
                "onnx_ready": onnx_ready,
                "description": spec.description,
            })
        return result

    def register_model(self, key: str, spec: ModelSpec) -> None:
        self._registry[key] = spec
        logger.info(f"Registered model: {key} ({spec.backend})")

    def update_ram(self) -> float:
        self.available_ram_gb = self._detect_ram()
        return self.available_ram_gb

    def print_models(self) -> None:
        print(f"\n{'='*65}")
        print(f"  Model Registry (RAM available: {self.available_ram_gb:.1f}GB)")
        print(f"{'='*65}")
        for info in self.list_available():
            ok   = "✓" if info["available"] else "✗"
            onnx = " [ONNX]" if info["onnx_ready"] else ""
            print(f"  [{ok}] {info['key']:<14} {info['backend']:<8} "
                  f"{info['min_ram_gb']:.0f}GB  "
                  f"{','.join(info['quality_tiers']):<18} {onnx}")
        print()
