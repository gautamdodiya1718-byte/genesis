"""
core/logger.py  —  Centralized logging with console + rotating file.
core/image_utils.py  —  PIL utilities, pHash, validation.
core/model_manager.py  —  HuggingFace auto-download + local cache.

All three modules unified into this file for import convenience.
Import via:
    from core.logger import setup_logging, get_logger
    from core.image_utils import load_image, perceptual_hash, ...
    from core.model_manager import ModelManager
"""

# ================================================================
# logger.py
# ================================================================
import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "outputs/logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "genesis.log"),
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
