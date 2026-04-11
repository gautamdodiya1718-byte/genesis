"""
models/vae/encoder.py
----------------------
VAE Encoder: images → (mean, log_var) latent distribution parameters.

Architecture:
  Conv2d input → [ResBlock + AttentionBlock]* × 4 levels (with 2× downsample) → conv_out
  Output: (B, 2*latent_channels, H/8, W/8) split into mean and log_var
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ResBlock(nn.Module):
    """Standard ResBlock with GroupNorm + SiLU (no timestep conditioning for VAE)."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention for VAE feature maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, -1).transpose(1, 2)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).transpose(1, 2)

        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class Encoder(nn.Module):
    """
    VAE Encoder: maps images to latent distribution parameters.

    Returns:
        mean:    (B, latent_channels, H/8, W/8)
        log_var: (B, latent_channels, H/8, W/8)
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: List[int] = None,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        channel_multipliers = channel_multipliers or [1, 2, 4, 4]
        attention_resolutions = attention_resolutions or [16]

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_multipliers]
        self.down_layers = nn.ModuleList()
        current_res = 256  # assumed input resolution

        ch = base_channels
        for i, out_ch in enumerate(channels):
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(ResBlock(ch, out_ch, dropout))
                ch = out_ch
                if current_res in attention_resolutions:
                    stage.append(AttentionBlock(ch))
                else:
                    stage.append(nn.Identity())
            self.down_layers.append(stage)

            if i < len(channels) - 1:
                self.down_layers.append(
                    nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)])
                )
                current_res //= 2

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, dropout)
        self.mid_attn   = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, dropout)

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, latent_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        for stage in self.down_layers:
            for layer in stage:
                h = layer(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        mean, log_var = h.chunk(2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        return mean, log_var
