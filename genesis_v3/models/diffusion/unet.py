"""
models/diffusion/unet.py
------------------------
U-Net backbone for latent diffusion.
Architecture: Encoder path → Bottleneck → Decoder path with skip connections.

Each resolution level contains:
  - ResBlocks with timestep conditioning (via AdaGN or addition)
  - SpatialTransformers with cross-attention (text/image conditioning)
  - Downsampling (encoder) / Upsampling (decoder)

This U-Net operates entirely in latent space (4-channel tensors from VAE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .attention import SpatialTransformer, SinusoidalTimestepEmbedding  # genesis flash-attention version


# ─────────────────────────────────────────────────────────────
# Residual Block with Timestep Conditioning
# ─────────────────────────────────────────────────────────────

class ResBlockTime(nn.Module):
    """
    ResBlock conditioned on timestep embedding.
    Timestep embedding is projected and added to the feature map
    between the two convolutions (temporal conditioning).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Project timestep embedding to channel dim
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C_in, H, W)
            t_emb: (B, time_emb_dim) timestep embedding

        Returns:
            out: (B, C_out, H, W)
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Inject timestep: add projected embedding (broadcast over H, W)
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t

        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────
# Down/Up sampling
# ─────────────────────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Latent Diffusion U-Net with:
    - Timestep conditioning via sinusoidal embeddings
    - Text/image cross-attention via SpatialTransformers
    - Skip connections between encoder and decoder
    - Flexible resolution-based attention placement

    Input:
        z_t:     (B, in_channels, H, W) noisy latent
        t:       (B,) integer timesteps
        context: (B, seq, context_dim) text/image embeddings  [optional]

    Output:
        pred: (B, out_channels, H, W) denoised prediction (epsilon or v)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 320,
        channel_multipliers: List[int] = None,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        context_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]
        if attention_resolutions is None:
            attention_resolutions = [4, 2, 1]

        self.base_channels = base_channels
        time_emb_dim = base_channels * 4

        # ── Timestep embedding ─────────────────────────────────
        self.time_embed = SinusoidalTimestepEmbedding(base_channels)
        # time_embed outputs base_channels * 4

        # ── Input projection ───────────────────────────────────
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Encoder (downsampling) path ────────────────────────
        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]  # Track channels for skip connections
        ch = base_channels
        ds = 1   # Current downsampling factor (1 = full res)

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            stage = nn.ModuleList()

            for _ in range(num_res_blocks):
                stage.append(ResBlockTime(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)

                # Add spatial transformer at attention resolutions
                # attention_resolutions = [4, 2, 1] means at 1/4, 1/2, 1/1 of base
                if ds in attention_resolutions:
                    stage.append(SpatialTransformer(ch, context_dim, num_heads, head_dim))
                else:
                    stage.append(nn.Identity())   # Placeholder for uniform indexing

            self.down_blocks.append(stage)

            if level < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                ds *= 2

        # ── Bottleneck ─────────────────────────────────────────
        self.mid_block1 = ResBlockTime(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SpatialTransformer(ch, context_dim, num_heads, head_dim)
        self.mid_block2 = ResBlockTime(ch, ch, time_emb_dim, dropout)

        # ── Decoder (upsampling) path ──────────────────────────
        self.up_blocks = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            stage = nn.ModuleList()

            for i in range(num_res_blocks + 1):
                # Input = current channels + skip connection channels
                in_ch = ch + skip_channels.pop()
                stage.append(ResBlockTime(in_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch

                if ds in attention_resolutions:
                    stage.append(SpatialTransformer(ch, context_dim, num_heads, head_dim))
                else:
                    stage.append(nn.Identity())

            self.up_blocks.append(stage)

            if level < len(channel_multipliers) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                ds //= 2

        # ── Output projection ──────────────────────────────────
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Zero-initialize output conv (improves training stability)
        nn.init.zeros_(self.conv_out.weight)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t:     (B, C, H, W) noisy latent at timestep t
            t:       (B,) integer timesteps
            context: (B, seq, context_dim) conditioning (text/image)

        Returns:
            pred: (B, C, H, W) denoised prediction
        """
        # Timestep embedding (B, time_emb_dim)
        t_emb = self.time_embed(t)

        # Entry
        h = self.conv_in(z_t)
        skips = [h]

        # Encoder path
        for block_group in self.down_blocks:
            for i, block in enumerate(block_group):
                if isinstance(block, ResBlockTime):
                    h = block(h, t_emb)
                    skips.append(h)
                elif isinstance(block, SpatialTransformer):
                    h = block(h, context)
                elif isinstance(block, Downsample):
                    h = block(h)
                    skips.append(h)
                else:
                    pass   # nn.Identity()

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        # Decoder path
        for block_group in self.up_blocks:
            for block in block_group:
                if isinstance(block, ResBlockTime):
                    # Concatenate skip connection
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, t_emb)
                elif isinstance(block, SpatialTransformer):
                    h = block(h, context)
                elif isinstance(block, Upsample):
                    h = block(h)
                else:
                    pass

        # Output
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
