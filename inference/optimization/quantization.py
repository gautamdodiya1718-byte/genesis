"""
inference/optimization/quantization.py
-----------------------------------------
Post-training quantization for Genesis models.

Reduces model size and improves CPU inference speed by converting
FP32 weights to INT8 or FP16.

Strategies:
  fp16_cast     FP32 → FP16 weight cast (2× smaller, ~same speed on CPU, 2× GPU)
  dynamic_int8  FP32 → INT8 weights with FP32 activations (no calibration needed)
  static_int8   FP32 → INT8 weights + INT8 activations (best speedup, needs data)
  onnx_int8     Post-quantize an ONNX model via ORT quantization tools

Typical results on SD 1.5 U-Net:
  fp16_cast:    1.0× speed,   ~2× RAM reduction  (weights only)
  dynamic_int8: 1.5–2×  speed, ~4× RAM reduction  (CPU)
  static_int8:  2–3×  speed,  ~4× RAM reduction  (CPU, needs calibration)

Usage:
    q = Quantizer()
    # Dynamic INT8 for a PyTorch module
    q_model = q.quantize_dynamic(model, dtype="int8")

    # ONNX dynamic quantization
    q.quantize_onnx_dynamic("model.onnx", "model_int8.onnx")

    # FP16 cast (for GPU or fast RAM reads on CPU)
    fp16_model = q.cast_fp16(model)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Quantization profiles ─────────────────────────────────────

# Layer types to include in dynamic quantization
_INT8_TARGETS = {
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.LSTM,
    torch.nn.GRU,
}


class QuantizationProfile:
    """
    Configuration for a quantization run.
    Controls which layers to quantize and how.
    """

    def __init__(
        self,
        mode: str = "dynamic_int8",        # fp16_cast | dynamic_int8 | static_int8
        targets: Optional[Set[type]] = None,
        skip_layers: Optional[List[str]] = None,  # layer name substrings to skip
        per_channel: bool = True,
        reduce_range: bool = False,
    ):
        self.mode         = mode
        self.targets      = targets or _INT8_TARGETS
        self.skip_layers  = skip_layers or ["norm", "embed", "pos_embed"]
        self.per_channel  = per_channel
        self.reduce_range = reduce_range

    @classmethod
    def for_unet(cls) -> "QuantizationProfile":
        """Recommended profile for SD U-Net quantization."""
        return cls(
            mode="dynamic_int8",
            skip_layers=["norm", "embed", "time_embed", "pos"],
            per_channel=True,
            reduce_range=False,
        )

    @classmethod
    def for_text_encoder(cls) -> "QuantizationProfile":
        """Recommended profile for CLIP text encoder quantization."""
        return cls(
            mode="dynamic_int8",
            targets={torch.nn.Linear},
            skip_layers=["LayerNorm", "embedding"],
            per_channel=False,
        )

    @classmethod
    def for_vae(cls) -> "QuantizationProfile":
        """Recommended profile for VAE decoder quantization."""
        return cls(
            mode="dynamic_int8",
            targets={torch.nn.Conv2d},
            skip_layers=["norm", "attn"],
            per_channel=True,
        )


class Quantizer:
    """
    Post-training quantizer for Genesis model components.

    Applies dynamic or static quantization to PyTorch models
    and ONNX models without requiring retraining.
    """

    def __init__(self, output_dir: str = "outputs/quantized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── FP16 cast ─────────────────────────────────────────────

    def cast_fp16(
        self,
        model: nn.Module,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Cast all floating point parameters to FP16.
        Reduces memory by ~50%. Works well on GPU.
        On CPU: may be slower unless using AVX-512 FP16 (rare).

        Returns original model (inplace) or a new model (inplace=False).
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        model.half()
        logger.info(
            f"Cast to FP16 | params={self._count_params(model):,} "
            f"size={self._model_size_mb(model):.0f}MB"
        )
        return model

    def cast_fp32(self, model: nn.Module, inplace: bool = False) -> nn.Module:
        """Upcast FP16 model back to FP32 (for CPU inference compatibility)."""
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        model.float()
        return model

    # ── Dynamic INT8 ──────────────────────────────────────────

    def quantize_dynamic(
        self,
        model: nn.Module,
        profile: Optional[QuantizationProfile] = None,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply PyTorch dynamic INT8 quantization.

        Dynamic quantization quantizes weights to INT8 at load time
        and activations on-the-fly during inference.
        No calibration data needed.

        Best for: Linear-heavy models (text encoders, U-Net attention layers).
        Speedup:  1.5–2× on CPU for linear layers.
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)

        profile = profile or QuantizationProfile()
        t0 = time.time()

        size_before = self._model_size_mb(model)
        model_q = torch.quantization.quantize_dynamic(
            model,
            profile.targets,
            dtype=torch.qint8,
        )
        size_after = self._model_size_mb(model_q)
        elapsed = time.time() - t0

        logger.info(
            f"Dynamic INT8 quantization | "
            f"{size_before:.0f}MB → {size_after:.0f}MB "
            f"({(1-size_after/size_before)*100:.0f}% reduction) "
            f"in {elapsed:.1f}s"
        )
        return model_q

    # ── Static INT8 ───────────────────────────────────────────

    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: List[Dict],
        profile: Optional[QuantizationProfile] = None,
    ) -> nn.Module:
        """
        Apply static INT8 quantization with calibration.

        Static quantization pre-computes activation scales using
        representative calibration data. Better accuracy than dynamic
        at the cost of requiring calibration samples.

        Args:
            calibration_data:  List of input dicts, e.g.
                               [{"sample": tensor, "timestep": tensor, ...}]
                               10-100 samples is typically sufficient.
        """
        import copy
        profile = profile or QuantizationProfile()

        model_q = copy.deepcopy(model)
        model_q.eval()

        # Prepare quantization config
        model_q.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # Skip specified layers
        if profile.skip_layers:
            self._set_skip_layers(model_q, profile.skip_layers)

        torch.quantization.prepare(model_q, inplace=True)

        # Calibration pass
        logger.info(f"Calibrating with {len(calibration_data)} samples...")
        t0 = time.time()
        with torch.no_grad():
            for sample in calibration_data:
                try:
                    if isinstance(sample, dict):
                        model_q(**sample)
                    elif isinstance(sample, (list, tuple)):
                        model_q(*sample)
                    else:
                        model_q(sample)
                except Exception as e:
                    logger.debug(f"Calibration sample failed: {e}")

        torch.quantization.convert(model_q, inplace=True)
        logger.info(
            f"Static INT8 done | "
            f"{self._model_size_mb(model):.0f}MB → {self._model_size_mb(model_q):.0f}MB "
            f"in {time.time()-t0:.1f}s"
        )
        return model_q

    def _set_skip_layers(self, model: nn.Module, skip_names: List[str]) -> None:
        """Mark layers as not-to-quantize based on name substrings."""
        for name, module in model.named_modules():
            if any(skip in name for skip in skip_names):
                module.qconfig = None

    # ── ONNX quantization ─────────────────────────────────────

    def quantize_onnx_dynamic(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        weight_type: str = "QInt8",        # QInt8 | QUInt8
        per_channel: bool = False,
    ) -> str:
        """
        Apply ONNX Runtime dynamic quantization to an ONNX model.

        Quantizes weight tensors to INT8 without calibration data.
        Fast to apply, works on most models.

        Returns path to quantized ONNX file.
        """
        try:
            from onnxruntime.quantization import (
                quantize_dynamic, QuantType, QuantizationMode
            )
        except ImportError:
            raise RuntimeError(
                "onnxruntime.quantization not available. "
                "Install: pip install onnxruntime-tools"
            )

        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_int8{p.suffix}")

        wtype = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8
        t0 = time.time()
        logger.info(f"ONNX dynamic INT8: {Path(input_path).name} → {Path(output_path).name}")

        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=wtype,
            per_channel=per_channel,
            reduce_range=False,
        )

        in_mb  = Path(input_path).stat().st_size / 1024**2
        out_mb = Path(output_path).stat().st_size / 1024**2
        logger.info(
            f"ONNX INT8: {in_mb:.0f}MB → {out_mb:.0f}MB "
            f"({(1-out_mb/in_mb)*100:.0f}% reduction) in {time.time()-t0:.1f}s"
        )
        return output_path

    def quantize_onnx_static(
        self,
        input_path: str,
        calibration_data_reader,   # ort.quantization.CalibrationDataReader
        output_path: Optional[str] = None,
        quant_format: str = "QOperator",
    ) -> str:
        """
        Apply ONNX Runtime static quantization with calibration.

        Requires a CalibrationDataReader that yields input dicts.
        Better accuracy than dynamic for compute-heavy models.

        Returns path to quantized ONNX file.
        """
        try:
            from onnxruntime.quantization import (
                quantize_static, QuantFormat, QuantType,
                CalibrationMethod
            )
        except ImportError:
            raise RuntimeError("onnxruntime.quantization not available")

        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_static_int8{p.suffix}")

        fmt = QuantFormat.QOperator if quant_format == "QOperator" else QuantFormat.QDQ

        t0 = time.time()
        logger.info(f"ONNX static INT8 with calibration: {Path(input_path).name}")

        quantize_static(
            model_input=input_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=fmt,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )

        in_mb  = Path(input_path).stat().st_size / 1024**2
        out_mb = Path(output_path).stat().st_size / 1024**2
        logger.info(
            f"ONNX static INT8: {in_mb:.0f}MB → {out_mb:.0f}MB in {time.time()-t0:.1f}s"
        )
        return output_path

    # ── Pipeline-level helpers ────────────────────────────────

    def quantize_diffusion_pipeline(
        self,
        pipeline,
        strategy: str = "dynamic_int8",
        components: Optional[List[str]] = None,
    ) -> dict:
        """
        Apply quantization to multiple components of a diffusers pipeline.

        Args:
            pipeline:   Loaded diffusers DiffusionPipeline
            strategy:   "fp16_cast" | "dynamic_int8"
            components: ["unet", "text_encoder", "vae"] (default: all)

        Returns:
            Dict of component → quantized module
        """
        if components is None:
            components = ["unet", "text_encoder"]  # VAE is already small

        results = {}
        for comp_name in components:
            module = getattr(pipeline, comp_name, None)
            if module is None:
                logger.warning(f"Component {comp_name} not found on pipeline")
                continue

            t0 = time.time()
            try:
                if strategy == "fp16_cast":
                    q_module = self.cast_fp16(module, inplace=False)
                elif strategy == "dynamic_int8":
                    profile_map = {
                        "unet": QuantizationProfile.for_unet(),
                        "text_encoder": QuantizationProfile.for_text_encoder(),
                        "vae": QuantizationProfile.for_vae(),
                    }
                    profile = profile_map.get(comp_name, QuantizationProfile())
                    q_module = self.quantize_dynamic(module, profile=profile)
                else:
                    logger.warning(f"Unknown strategy: {strategy}")
                    continue

                # Replace component on pipeline
                setattr(pipeline, comp_name, q_module)
                results[comp_name] = q_module
                logger.info(
                    f"Quantized pipeline.{comp_name} [{strategy}] "
                    f"in {time.time()-t0:.1f}s"
                )
            except Exception as e:
                logger.error(f"Quantization failed for {comp_name}: {e}")

        return results

    # ── Save / Load ───────────────────────────────────────────

    def save(self, model: nn.Module, name: str) -> str:
        """Save a quantized model to disk."""
        path = self.output_dir / f"{name}.pt"
        torch.save(model.state_dict(), str(path))
        logger.info(f"Saved quantized model: {path}")
        return str(path)

    def load_dynamic(
        self, model: nn.Module, path: str,
        profile: Optional[QuantizationProfile] = None,
    ) -> nn.Module:
        """Load weights into a dynamically-quantized version of model."""
        profile = profile or QuantizationProfile()
        q_model = torch.quantization.quantize_dynamic(
            model, profile.targets, dtype=torch.qint8
        )
        q_model.load_state_dict(torch.load(path, map_location="cpu"))
        return q_model

    # ── Utilities ─────────────────────────────────────────────

    def _model_size_mb(self, model: nn.Module) -> float:
        import io
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell() / 1024**2

    def _count_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def benchmark(
        self,
        original: nn.Module,
        quantized: nn.Module,
        dummy_input,
        n_runs: int = 20,
    ) -> dict:
        """
        Compare latency of original vs quantized model.
        Returns speedup ratio and size reduction.
        """
        def _time_model(m, inp, n):
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    if isinstance(inp, dict):
                        m(**inp)
                    else:
                        m(*inp) if isinstance(inp, tuple) else m(inp)
                times = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    if isinstance(inp, dict):
                        m(**inp)
                    else:
                        m(*inp) if isinstance(inp, tuple) else m(inp)
                    times.append((time.perf_counter() - t0) * 1000)
            return sum(times) / len(times)

        orig_ms  = _time_model(original,  dummy_input, n_runs)
        quant_ms = _time_model(quantized, dummy_input, n_runs)
        speedup  = orig_ms / max(quant_ms, 0.01)
        size_reduction = 1 - (self._model_size_mb(quantized) /
                               max(self._model_size_mb(original), 0.01))

        logger.info(
            f"Quantization benchmark: "
            f"orig={orig_ms:.1f}ms quant={quant_ms:.1f}ms "
            f"speedup={speedup:.2f}× size_reduction={size_reduction*100:.0f}%"
        )
        return {
            "original_ms":     round(orig_ms, 2),
            "quantized_ms":    round(quant_ms, 2),
            "speedup":         round(speedup, 3),
            "size_reduction":  round(size_reduction, 3),
            "n_runs":          n_runs,
        }
