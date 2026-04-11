"""
models/text_encoder/encoder.py
--------------------------------
Unified text encoder supporting CLIP and T5 backends.
Merged from LocalDiffusion with updated imports for Genesis.
"""

from __future__ import annotations
import logging
from typing import List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    Wraps CLIP or T5 text encoder for conditioning the diffusion U-Net.

    CLIP (default): (B, 77, 768)  — SD 1.x compatible
    T5:             (B, L,  D)    — SD3/FLUX style, richer semantics

    Typically frozen during diffusion training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.model_id  = cfg.text_encoder.model_id
        self.max_length = cfg.text_encoder.get("max_length", 77)
        self.freeze     = cfg.text_encoder.get("freeze", True)
        self._backend   = "clip" if "clip" in self.model_id.lower() else "t5"

        self.tokenizer = None
        self.model     = None
        self._null_emb: Optional[torch.Tensor] = None
        self._loaded   = False

    def load(self) -> None:
        if self._loaded:
            return
        if self._backend == "clip":
            from transformers import CLIPTokenizer, CLIPTextModel
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
            self.model     = CLIPTextModel.from_pretrained(self.model_id)
        else:
            from transformers import T5Tokenizer, T5EncoderModel
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
            self.model     = T5EncoderModel.from_pretrained(self.model_id)

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        self._loaded = True
        logger.info(f"TextEncoder loaded: {self.model_id} ({self._backend})")

    def forward(self, texts: List[str]) -> torch.Tensor:
        if not self._loaded:
            self.load()

        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.set_grad_enabled(not self.freeze):
            if self._backend == "clip":
                out = self.model(**tokens)
                return out.last_hidden_state   # (B, 77, 768)
            else:
                out = self.model(**tokens)
                return out.last_hidden_state   # (B, L, D)

    def get_null_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """Cached empty-string embedding for CFG unconditional branch."""
        if self._null_emb is None:
            self._null_emb = self.forward([""])
        return self._null_emb.expand(batch_size, -1, -1)

    def encode_for_guidance(
        self, prompts: List[str], negative_prompts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Build (2B, L, D) combined batch: [uncond; cond] for single U-Net pass."""
        neg = negative_prompts or [""] * len(prompts)
        cond   = self.forward(prompts)
        uncond = self.forward(neg)
        return torch.cat([uncond, cond], dim=0)
