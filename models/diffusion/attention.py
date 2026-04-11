"""
models/diffusion/attention.py
------------------------------
Attention modules for the diffusion U-Net.

NEW in Genesis v0.2:
  - Flash Attention via torch.nn.functional.scaled_dot_product_attention
    (PyTorch 2.0+). Enabled when use_flash_attention=true in config.
    Falls back gracefully to standard attention if unavailable.
  - ~2-4× speedup and ~50% VRAM reduction at 512px+ resolution.

Modules:
  SinusoidalTimestepEmbedding — timestep → learned embedding
  FlashMultiHeadAttention     — unified self/cross attention with Flash Attn
  GEGLU                       — gated activation for FFN
  SpatialTransformerBlock     — pre-norm: SA → CA → FFN
  SpatialTransformer          — wraps blocks for 2D feature maps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Detect Flash Attention availability (PyTorch 2.0+)
_FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


# ─────────────────────────────────────────────────────────────
# Timestep Embedding
# ─────────────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep encoding → learned embedding via 2-layer MLP.
    Input:  (B,) float timesteps
    Output: (B, dim * 4) conditioning vector
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────
# Flash Attention — unified self / cross attention
# ─────────────────────────────────────────────────────────────

class FlashMultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional Flash Attention backend.

    When use_flash=True and PyTorch ≥ 2.0:
      - Uses F.scaled_dot_product_attention (FlashAttention-2 kernel on CUDA)
      - ~2-4× faster, ~50% less VRAM vs naive attention
      - Mathematically identical to standard attention

    When use_flash=False or PyTorch < 2.0:
      - Falls back to explicit QK^T V attention (always safe)

    For self-attention:   context=None  (Q, K, V all from x)
    For cross-attention:  context=text_embeddings (K, V from context)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.use_flash = use_flash and _FLASH_AVAILABLE

        ctx_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(ctx_dim,   self.inner_dim, bias=False)
        self.to_v = nn.Linear(ctx_dim,   self.inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, H*D) → (B, H, L, D)"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx = context if context is not None else x

        q = self._reshape(self.to_q(x))    # (B, H, Lq, D)
        k = self._reshape(self.to_k(ctx))  # (B, H, Lk, D)
        v = self._reshape(self.to_v(ctx))  # (B, H, Lk, D)

        if self.use_flash:
            # PyTorch 2.0+ — dispatches to FlashAttention-2 on CUDA,
            # or optimized CPU kernel on CPU. No manual softmax needed.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            # Standard explicit attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        # (B, H, L, D) → (B, L, H*D)
        B, H, L, D = out.shape
        out = out.transpose(1, 2).reshape(B, L, H * D)
        return self.to_out(out)


# ─────────────────────────────────────────────────────────────
# FFN with GEGLU activation
# ─────────────────────────────────────────────────────────────

class GEGLU(nn.Module):
    """Gated GeLU — better than ReLU/GELU for transformer FFNs."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            GEGLU(dim, dim * mult),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Spatial Transformer Block
# ─────────────────────────────────────────────────────────────

class SpatialTransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
      LayerNorm → Self-Attention → LayerNorm → Cross-Attention → LayerNorm → FFN
    All residual connections included.
    """
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = FlashMultiHeadAttention(
            dim, context_dim=None, num_heads=num_heads,
            head_dim=head_dim, dropout=dropout, use_flash=use_flash,
        )
        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = FlashMultiHeadAttention(
            dim, context_dim=context_dim, num_heads=num_heads,
            head_dim=head_dim, dropout=dropout, use_flash=use_flash,
        )
        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))               # Self-attention
        x = x + self.attn2(self.norm2(x), context)      # Cross-attention
        x = x + self.ff(self.norm3(x))                  # FFN
        return x


class SpatialTransformer(nn.Module):
    """
    Wraps SpatialTransformerBlocks for 2D feature maps.

    Input:  (B, C, H, W) feature map
    Output: (B, C, H, W) same shape after attending to context

    Internally reshapes to (B, H*W, C), applies N transformer blocks,
    then reshapes back.
    """
    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1)

        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(
                in_channels, context_dim, num_heads, head_dim, dropout, use_flash
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
        # Zero-init: starts as identity → stable early training
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        x = self.proj_in(x)

        # (B, C, H, W) → (B, H*W, C) for transformer
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        for block in self.blocks:
            x = block(x, context=context)

        # (B, H*W, C) → (B, C, H, W)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)

        return x + residual
