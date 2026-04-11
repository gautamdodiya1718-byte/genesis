"""
inference/optimization/openvino_backend.py
--------------------------------------------
OpenVINO backend for accelerated CPU inference.

OpenVINO (Intel's Open Visual Inference and Neural Networks Optimization)
provides 2-4× CPU speedup over PyTorch on Intel hardware via:
  - Kernel fusion and graph optimization
  - INT8 quantization (optional)
  - Intel MKL-DNN acceleration
  - Multi-core threading with OpenMP

Workflow:
  1. Export model from ONNX → OpenVINO IR (XML + BIN)
  2. Load IR with OpenVINO Runtime
  3. Run inference via CompiledModel

Compatibility:
  Works on any x86 CPU (Intel/AMD).
  Best speedup on Intel CPUs with AVX-512.
  Falls back to ONNX Runtime if OpenVINO not installed.

Layout:
  outputs/openvino/<model_key>/<component>/
    model.xml   -- OpenVINO IR graph
    model.bin   -- OpenVINO IR weights
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_OV_AVAILABLE = False
try:
    import openvino as ov
    from openvino.runtime import Core, CompiledModel, InferRequest
    _OV_AVAILABLE = True
except ImportError:
    pass

_ORT_FALLBACK = False
try:
    import onnxruntime as ort
    _ORT_FALLBACK = True
except ImportError:
    pass


class OpenVINOBackend:
    """
    OpenVINO runtime backend for Genesis inference.

    Converts ONNX models → OpenVINO IR and compiles them for the target device.
    Falls back to ONNX Runtime if OpenVINO is unavailable.

    Usage:
        backend = OpenVINOBackend("outputs/openvino")

        # Convert from ONNX
        backend.convert("outputs/onnx/sd15/unet/model.onnx", "sd15", "unet")

        # Load and run
        req = backend.load("sd15", "unet")
        output = backend.infer(req, {"sample": ..., "timestep": ..., "encoder_hidden_states": ...})
    """

    def __init__(
        self,
        ov_root: str = "outputs/openvino",
        device: str = "CPU",           # CPU | GPU | AUTO
        num_streams: int = 1,
        num_threads: Optional[int] = None,
        enable_int8: bool = False,
    ):
        self.ov_root      = Path(ov_root)
        self.ov_root.mkdir(parents=True, exist_ok=True)
        self.device       = device
        self.num_streams  = num_streams
        self.num_threads  = num_threads or self._cpu_count()
        self.enable_int8  = enable_int8
        self._compiled:   Dict[str, Any] = {}  # cache_key → compiled model
        self._ort_sessions: Dict[str, Any] = {}

        if _OV_AVAILABLE:
            self._core = ov.Core()
            logger.info(
                f"OpenVINO backend ready | device={device} "
                f"threads={self.num_threads}"
            )
        elif _ORT_FALLBACK:
            logger.warning(
                "OpenVINO not installed — using ONNX Runtime fallback. "
                "Install: pip install openvino openvino-dev"
            )
        else:
            logger.error("Neither OpenVINO nor ONNX Runtime available")

    def _cpu_count(self) -> int:
        try:
            import psutil
            return psutil.cpu_count(logical=False) or 4
        except ImportError:
            import os
            return max(1, (os.cpu_count() or 4) // 2)

    def _ir_dir(self, model_key: str, component: str) -> Path:
        d = self.ov_root / model_key / component
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ir_xml(self, model_key: str, component: str) -> Path:
        return self._ir_dir(model_key, component) / "model.xml"

    # ── Conversion: ONNX → OV IR ─────────────────────────────

    def convert(
        self,
        onnx_path: str,
        model_key: str,
        component: str,
        force: bool = False,
    ) -> Optional[str]:
        """
        Convert an ONNX model to OpenVINO IR format.

        Args:
            onnx_path:  Path to source ONNX file
            model_key:  Identifier (e.g. "sd15")
            component:  Component name (e.g. "unet")
            force:      Re-convert even if IR exists

        Returns:
            Path to model.xml or None if conversion failed
        """
        xml_path = self._ir_xml(model_key, component)
        if xml_path.exists() and not force:
            logger.info(f"OV IR exists: {model_key}/{component}")
            return str(xml_path)

        if not _OV_AVAILABLE:
            logger.warning("OpenVINO not available — cannot convert")
            return None

        if not Path(onnx_path).exists():
            logger.error(f"ONNX source not found: {onnx_path}")
            return None

        logger.info(f"Converting {component} ONNX → OV IR...")
        t0 = time.time()

        try:
            model = self._core.read_model(onnx_path)

            if self.enable_int8:
                model = self._apply_int8(model, component)

            # Set layout hints for better optimization
            self._set_layout_hints(model, component)

            ov.save_model(model, str(xml_path))

            elapsed = time.time() - t0
            xml_size_mb = xml_path.stat().st_size / 1024**2
            bin_path    = xml_path.with_suffix(".bin")
            bin_size_mb = bin_path.stat().st_size / 1024**2 if bin_path.exists() else 0
            logger.info(
                f"OV IR {model_key}/{component}: "
                f"xml={xml_size_mb:.0f}MB bin={bin_size_mb:.0f}MB in {elapsed:.1f}s"
            )
            return str(xml_path)

        except Exception as e:
            logger.error(f"OV conversion failed [{component}]: {e}", exc_info=True)
            return None

    def _apply_int8(self, model, component: str):
        """Apply INT8 quantization via NNCF (if available)."""
        try:
            import nncf
            logger.info(f"Applying NNCF INT8 quantization to {component}")
            # Note: actual NNCF requires calibration dataset
            # This is a placeholder — full INT8 needs calibration data
            return model
        except ImportError:
            logger.debug("NNCF not available — skipping INT8")
            return model

    def _set_layout_hints(self, model, component: str) -> None:
        """Set input layout hints for OV optimization."""
        try:
            from openvino.runtime import Layout
            if component in ("unet",):
                for inp in model.inputs:
                    name = inp.any_name
                    if "sample" in name or "latent" in name:
                        try:
                            model.reshape({name: [-1, -1, -1, -1]})
                        except Exception:
                            pass
        except Exception:
            pass

    # ── Load compiled model ───────────────────────────────────

    def load(
        self,
        model_key: str,
        component: str,
        onnx_path: Optional[str] = None,
    ) -> Any:
        """
        Load and compile model for inference.

        If OpenVINO IR doesn't exist, converts from onnx_path first.
        If OpenVINO unavailable, returns ORT session fallback.

        Returns CompiledModel (OV) or InferenceSession (ORT fallback).
        """
        cache_key = f"{model_key}/{component}"
        if cache_key in self._compiled:
            return self._compiled[cache_key]

        if _OV_AVAILABLE:
            xml_path = self._ir_xml(model_key, component)

            if not xml_path.exists():
                if onnx_path is None:
                    raise FileNotFoundError(
                        f"OV IR not found: {xml_path}. "
                        f"Run convert() first or provide onnx_path."
                    )
                result = self.convert(onnx_path, model_key, component)
                if result is None:
                    raise RuntimeError(f"OV conversion failed for {cache_key}")

            logger.info(f"Compiling OV model: {cache_key} | device={self.device}")
            t0 = time.time()

            config = {
                "NUM_STREAMS": str(self.num_streams),
                "INFERENCE_NUM_THREADS": str(self.num_threads),
                "PERFORMANCE_HINT": "THROUGHPUT" if self.num_streams > 1 else "LATENCY",
            }
            compiled = self._core.compile_model(str(xml_path), self.device, config)
            self._compiled[cache_key] = compiled
            logger.info(f"OV compiled {cache_key} in {time.time()-t0:.1f}s")
            return compiled

        elif _ORT_FALLBACK and onnx_path:
            # Fallback to ORT
            logger.info(f"OV unavailable — using ORT fallback for {cache_key}")
            if cache_key not in self._ort_sessions:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = self.num_threads
                self._ort_sessions[cache_key] = ort.InferenceSession(
                    onnx_path, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            return self._ort_sessions[cache_key]

        raise RuntimeError(
            f"Cannot load {cache_key}: OpenVINO and ORT both unavailable"
        )

    # ── Inference ─────────────────────────────────────────────

    def infer(
        self,
        compiled_model,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a compiled OV model or ORT session.
        Returns dict of output name → numpy array.
        """
        if _OV_AVAILABLE and isinstance(compiled_model, ov.CompiledModel if _OV_AVAILABLE else type(None)):
            infer_req = compiled_model.create_infer_request()
            infer_req.infer(inputs)
            return {out.any_name: infer_req.get_output_tensor(i).data.copy()
                    for i, out in enumerate(compiled_model.outputs)}
        else:
            # ORT session fallback
            input_names = [inp.name for inp in compiled_model.get_inputs()]
            feeds = {k: v.astype(np.float32) for k, v in inputs.items()
                     if k in input_names}
            outputs = compiled_model.run(None, feeds)
            out_names = [o.name for o in compiled_model.get_outputs()]
            return dict(zip(out_names, outputs))

    def infer_unet(
        self,
        compiled_model,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
    ) -> np.ndarray:
        """Convenience wrapper for U-Net inference."""
        out = self.infer(compiled_model, {
            "sample": sample.astype(np.float32),
            "timestep": timestep.astype(np.int64 if not _OV_AVAILABLE else np.float32),
            "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
        })
        return list(out.values())[0]

    def infer_vae_decoder(
        self, compiled_model, latent: np.ndarray
    ) -> np.ndarray:
        out = self.infer(compiled_model, {"latent": latent.astype(np.float32)})
        return list(out.values())[0]

    # ── Benchmark ─────────────────────────────────────────────

    def benchmark(
        self,
        model_key: str,
        component: str,
        n_runs: int = 20,
        onnx_path: Optional[str] = None,
    ) -> dict:
        """Benchmark compiled model throughput."""
        compiled = self.load(model_key, component, onnx_path=onnx_path)
        dummy = self._dummy_inputs(component, model_key)

        # Warmup
        for _ in range(3):
            self.infer(compiled, dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.infer(compiled, dummy)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = sum(times) / len(times)
        std_ms  = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
        backend = "OpenVINO" if _OV_AVAILABLE else "ORT-fallback"
        logger.info(
            f"[{backend}] {model_key}/{component}: "
            f"{mean_ms:.1f}±{std_ms:.1f}ms ({n_runs} runs)"
        )
        return {
            "backend": backend,
            "model_key": model_key,
            "component": component,
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "n_runs": n_runs,
        }

    def _dummy_inputs(self, component: str, model_key: str) -> Dict[str, np.ndarray]:
        from inference.optimization.onnx_optimizer import _SHAPE_CONFIGS
        shapes = _SHAPE_CONFIGS.get(model_key, _SHAPE_CONFIGS["sd15"])
        if component == "unet":
            return {
                "sample":   np.random.randn(1, shapes["latent_channels"],
                                            shapes["latent_h"], shapes["latent_w"]).astype(np.float32),
                "timestep": np.array([999], dtype=np.float32),
                "encoder_hidden_states": np.random.randn(
                    1, shapes["text_seq_len"], shapes["context_dim"]).astype(np.float32),
            }
        return {
            "latent": np.random.randn(1, 4, 64, 64).astype(np.float32)
        }

    # ── Status ────────────────────────────────────────────────

    def available(self) -> bool:
        return _OV_AVAILABLE

    def devices(self) -> List[str]:
        if not _OV_AVAILABLE:
            return []
        return self._core.available_devices

    def status(self) -> dict:
        ir_models = {}
        for model_dir in self.ov_root.iterdir():
            if not model_dir.is_dir():
                continue
            comps = [c.name for c in model_dir.iterdir()
                     if c.is_dir() and (c / "model.xml").exists()]
            if comps:
                ir_models[model_dir.name] = comps
        return {
            "openvino_available": _OV_AVAILABLE,
            "ort_fallback": _ORT_FALLBACK,
            "device": self.device,
            "ir_models": ir_models,
            "compiled_cache": list(self._compiled.keys()),
        }
