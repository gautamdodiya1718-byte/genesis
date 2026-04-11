"""
core/image_utils.py
--------------------
Shared image utilities: loading, saving, resizing, validation,
perceptual hashing, and deduplication helpers.
"""

from __future__ import annotations
import hashlib, io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Load / Save ────────────────────────────────────────────────

def load_image(path: Union[str, Path]) -> Optional[Image.Image]:
    try:
        img = Image.open(str(path))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def save_image(
    image: Image.Image, path: Union[str, Path],
    fmt: str = "PNG", quality: int = 95
) -> bool:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(str(path), format=fmt, quality=quality)
        return True
    except Exception:
        return False


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


# ── Validation ─────────────────────────────────────────────────

def is_valid_image(path: Union[str, Path], min_size: int = 64) -> bool:
    path = Path(path)
    if path.suffix.lower() not in VALID_EXT:
        return False
    img = load_image(path)
    if img is None:
        return False
    w, h = img.size
    return w >= min_size and h >= min_size


def validate_image_bytes(data: bytes, min_size: int = 64) -> bool:
    if len(data) < 100:
        return False
    img = bytes_to_image(data)
    if img is None:
        return False
    w, h = img.size
    return w >= min_size and h >= min_size


# ── Resize ──────────────────────────────────────────────────────

def resize_center_crop(
    image: Image.Image, target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    tw, th = target_size
    ow, oh = image.size
    scale = max(tw / ow, th / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    img = image.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def resize_with_padding(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    tw, th = target_size
    ow, oh = image.size
    scale = min(tw / ow, th / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = image.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", target_size, fill)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


# ── Hashing & Deduplication ─────────────────────────────────────

def md5_hash(path: Union[str, Path]) -> str:
    h = hashlib.md5()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(image: Image.Image, size: int = 16) -> str:
    img = image.resize((size, size), Image.LANCZOS).convert("L")
    pixels = np.array(img, dtype=np.float32)
    bits = (pixels > pixels.mean()).flatten()
    return np.packbits(bits).tobytes().hex()


def hash_distance(h1: str, h2: str) -> float:
    b1, b2 = bytes.fromhex(h1), bytes.fromhex(h2)
    if len(b1) != len(b2):
        return 1.0
    total = len(b1) * 8
    diff = sum(bin(a ^ b).count("1") for a, b in zip(b1, b2))
    return diff / total


def are_duplicates(h1: str, h2: str, threshold: float = 0.05) -> bool:
    return hash_distance(h1, h2) <= threshold
