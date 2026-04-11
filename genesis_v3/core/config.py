"""
core/config.py
--------------
Unified configuration system for the Genesis platform.

Merges the typed dataclass approach (LocalDiffusion) with the
flexible dot-notation ConfigNode approach (AutoDiff).

Features:
  - YAML loading with inheritance (base + override files)
  - Dot-notation attribute access
  - CLI key=value overrides with auto type-coercion
  - Device auto-detection (cuda > mps > cpu)
  - Config validation
"""

from __future__ import annotations
import os
import yaml
import logging
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigNode(dict):
    """Dict with attribute-style access. Nested dicts become ConfigNodes."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            return ConfigNode(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default=None) -> Any:
        val = super().get(key, default)
        return ConfigNode(val) if isinstance(val, dict) else val

    def get_nested(self, dot_path: str, default=None) -> Any:
        parts = dot_path.split(".")
        node = self
        for p in parts:
            if not isinstance(node, (dict, ConfigNode)) or p not in node:
                return default
            node = node[p]
        return node

    def set_nested(self, dot_path: str, value: Any) -> None:
        parts = dot_path.split(".")
        node = self
        for p in parts[:-1]:
            if p not in node:
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        """Convert back to plain nested dict."""
        result = {}
        for k, v in self.items():
            result[k] = v.to_dict() if isinstance(v, ConfigNode) else v
        return result


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def load_config(
    config_path: str = "configs/base.yaml",
    override_paths: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigNode:
    """
    Load YAML config with optional additional override files and
    dot-notation key=value overrides.

    Args:
        config_path:    Base YAML config file
        override_paths: List of additional YAML files to merge on top
        overrides:      Dict of dot-notation overrides {"training.lr": 1e-5}

    Returns:
        ConfigNode with dot-notation access
    """
    # Load base config
    raw = _load_yaml(config_path)

    # Merge override files
    for op in (override_paths or []):
        extra = _load_yaml(op)
        raw = _deep_merge(raw, extra)

    cfg = ConfigNode(raw)

    # Apply CLI/programmatic overrides
    for key, value in (overrides or {}).items():
        cfg.set_nested(key, value)
        logger.debug(f"Config override: {key} = {value}")

    # Auto-detect device
    cfg = _resolve_device(cfg)

    return cfg


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Config file not found: {path}, using empty dict")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_device(cfg: ConfigNode) -> ConfigNode:
    """Replace 'auto' device with actual available device."""
    device = cfg.get_nested("system.device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        cfg.set_nested("system.device", device)
        logger.info(f"Auto-detected device: {device}")
    return cfg


# ─────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────

def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    Parse key=value CLI args with type coercion.
    Example: ["training.lr=5e-5", "dataset.batch_size=16"]
    """
    overrides = {}
    for item in args:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        overrides[key] = _coerce(raw)
    return overrides


def _coerce(raw: str) -> Any:
    """Coerce a string value to the most appropriate Python type."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def validate_config(cfg: ConfigNode) -> List[str]:
    """
    Validate config for common mistakes.
    Returns list of warning strings (empty = all good).
    """
    warnings = []

    device = cfg.get_nested("system.device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        warnings.append("device=cuda but CUDA is not available; will fall back to CPU")

    batch = cfg.get_nested("training.batch_size", 4)
    accum = cfg.get_nested("training.gradient_accumulation_steps", 1)
    eff = batch * accum
    if eff < 16:
        warnings.append(f"Effective batch size={eff} is small; consider increasing batch_size or gradient_accumulation_steps")

    if cfg.get_nested("diffusion.use_flash_attention", False) and not torch.cuda.is_available():
        warnings.append("use_flash_attention=true but no GPU found; flash attention requires CUDA")

    return warnings
