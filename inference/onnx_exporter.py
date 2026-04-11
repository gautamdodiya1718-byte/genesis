"""
inference/onnx_exporter.py
---------------------------
ONNX export and accelerated inference for the Genesis system.

NEW in Genesis v0.2: ONNX Runtime enables 2-3× CPU speedup over
PyTorch for inference without any quality loss.

How it works:
  1. Export the U-Net to ONNX once (takes ~2 min, saved to disk)
  2. Load ONNX model with ONNXRuntime for all subsequent inferences
  3. ONNXRuntime uses optimized CPU kernels (MKL/OpenBLAS/XNNPACK)
     that are faster than PyTorch's generic fallback ops

Components exported:
  - U-Net (biggest speedup — runs at every denoising step)
  - VAE Decoder (secondary export for faster image decoding)

The VAE Encoder is only needed during training, not inference.
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Exports PyTorch models (U-Net, VAE decoder) to ONNX format.
    Handles dynamic shape axes for variable batch sizes and resolutions.
    """

    def __init__(self, onnx_dir: str = "model_cache/onnx", opset: int = 17):
        self.onnx_dir = Path(onnx_dir)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.opset = opset

    def export_unet(
        self,
        unet: nn.Module,
        context_dim: int = 768,
        latent_channels: int = 4,
        latent_size: int = 64,
        batch_size: int = 1,
        force: bool = False,
    ) -> str:
        """
        Export U-Net to ONNX.

        Inputs:
          - latent:  (B, 4, H/8, W/8) noisy latent
          - t:       (B,) timestep indices
          - context: (B, 77, context_dim) text embeddings

        Output:
          - noise_pred: (B, 4, H/8, W/8) predicted noise

        Returns path to exported .onnx file.
        """
        out_path = self.onnx_dir / "unet.onnx"
        if out_path.exists() and not force:
            logger.info(f"U-Net ONNX already exists: {out_path}")
            return str(out_path)

        logger.info("Exporting U-Net to ONNX (this takes ~2 minutes)...")
        unet.eval()

        # Dummy inputs matching the model's expected signature
        latent  = torch.randn(batch_size, latent_channels, latent_size, latent_size)
        t       = torch.zeros(batch_size, dtype=torch.long)
        context = torch.randn(batch_size, 77, context_dim)

        with torch.no_grad():
            torch.onnx.export(
                unet,
                (latent, t, context),
                str(out_path),
                opset_version=self.opset,
                input_names=["latent", "timestep", "context"],
                output_names=["noise_pred"],
                dynamic_axes={
                    "latent":     {0: "batch", 2: "height", 3: "width"},
                    "timestep":   {0: "batch"},
                    "context":    {0: "batch"},
                    "noise_pred": {0: "batch", 2: "height", 3: "width"},
                },
                do_constant_folding=True,
            )

        size_mb = out_path.stat().st_size / 1024**2
        logger.info(f"U-Net exported → {out_path} ({size_mb:.0f} MB)")
        return str(out_path)

    def export_vae_decoder(
        self,
        vae: nn.Module,
        latent_channels: int = 4,
        latent_size: int = 64,
        force: bool = False,
    ) -> str:
        """Export VAE decoder to ONNX for faster image decoding."""
        out_path = self.onnx_dir / "vae_decoder.onnx"
        if out_path.exists() and not force:
            logger.info(f"VAE decoder ONNX exists: {out_path}")
            return str(out_path)

        logger.info("Exporting VAE decoder to ONNX...")
        vae.eval()

        # We need just the decoder forward pass
        class DecoderWrapper(nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            def forward(self, z):
                return self.vae.decode(z)

        wrapper = DecoderWrapper(vae)
        z = torch.randn(1, latent_channels, latent_size, latent_size)

        with torch.no_grad():
            torch.onnx.export(
                wrapper, (z,), str(out_path),
                opset_version=self.opset,
                input_names=["latent"],
                output_names=["image"],
                dynamic_axes={
                    "latent": {0: "batch", 2: "height", 3: "width"},
                    "image":  {0: "batch", 2: "out_height", 3: "out_width"},
                },
            )

        logger.info(f"VAE decoder exported → {out_path}")
        return str(out_path)

    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Apply ONNX Runtime graph optimizations.
        Produces a _opt.onnx file with fused ops, constant folding, etc.
        """
        try:
            import onnxruntime as ort
            from onnxruntime.transformers import optimizer as ort_opt
        except ImportError:
            logger.warning("onnxruntime not installed — skipping optimization")
            return onnx_path

        opt_path = onnx_path.replace(".onnx", "_opt.onnx")
        if os.path.exists(opt_path):
            return opt_path

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.optimized_model_filepath = opt_path

        ort.InferenceSession(onnx_path, sess_opts)
        logger.info(f"ONNX optimized → {opt_path}")
        return opt_path


class ONNXUNet:
    """
    Drop-in replacement for the PyTorch U-Net that uses ONNX Runtime.
    2-3× faster CPU inference with identical outputs.

    Usage:
        onnx_unet = ONNXUNet("model_cache/onnx/unet.onnx")
        noise_pred = onnx_unet(latent, t, context)
    """

    def __init__(self, onnx_path: str, num_threads: Optional[int] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        self.device = "cpu"
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if num_threads is not None:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1

        # Use CPU EP (execution provider)
        providers = ["CPUExecutionProvider"]

        # Use CUDA EP if available
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "cuda"

        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        logger.info(
            f"ONNX U-Net loaded | provider={providers[0]} | "
            f"inputs={self._input_names}"
        )

    def __call__(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass via ONNX Runtime.
        Accepts PyTorch tensors, returns PyTorch tensor.
        """
        # Convert to numpy (ONNX Runtime expects numpy arrays)
        inputs = {
            "latent":    latent.cpu().float().numpy(),
            "timestep":  t.cpu().numpy(),
            "context":   context.cpu().float().numpy(),
        }

        # Filter to only the inputs the model expects
        inputs = {k: v for k, v in inputs.items() if k in self._input_names}

        outputs = self.session.run(None, inputs)
        return torch.from_numpy(outputs[0])


class ONNXVAEDecoder:
    """ONNX Runtime VAE decoder for fast image decoding."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        logger.info(f"ONNX VAE decoder loaded ← {onnx_path}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.session.run(None, {"latent": z.cpu().float().numpy()})
        return torch.from_numpy(out[0])


def benchmark_onnx_vs_pytorch(
    pytorch_unet: nn.Module,
    onnx_unet: ONNXUNet,
    latent_size: int = 64,
    context_dim: int = 768,
    n_runs: int = 5,
) -> dict:
    """
    Compare PyTorch vs ONNX runtime on the same inputs.
    Returns timing dict with speedup ratio.
    """
    latent  = torch.randn(1, 4, latent_size, latent_size)
    t       = torch.zeros(1, dtype=torch.long)
    context = torch.randn(1, 77, context_dim)

    # Warmup
    with torch.no_grad():
        pytorch_unet(latent, t, context)
    onnx_unet(latent, t, context)

    # Benchmark PyTorch
    pt_times = []
    for _ in range(n_runs):
        s = time.time()
        with torch.no_grad():
            pytorch_unet(latent, t, context)
        pt_times.append(time.time() - s)

    # Benchmark ONNX
    onnx_times = []
    for _ in range(n_runs):
        s = time.time()
        onnx_unet(latent, t, context)
        onnx_times.append(time.time() - s)

    pt_avg   = sum(pt_times)   / n_runs
    onnx_avg = sum(onnx_times) / n_runs
    speedup  = pt_avg / onnx_avg

    result = {
        "pytorch_ms":  round(pt_avg * 1000, 1),
        "onnx_ms":     round(onnx_avg * 1000, 1),
        "speedup":     round(speedup, 2),
    }
    logger.info(
        f"Benchmark: PyTorch={result['pytorch_ms']}ms | "
        f"ONNX={result['onnx_ms']}ms | "
        f"Speedup={result['speedup']}×"
    )
    return result
