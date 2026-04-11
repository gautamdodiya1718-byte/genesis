"""
models/vae/decoder.py
----------------------
VAE Decoder: latent → pixel-space images.
Mirror of encoder with upsampling instead of downsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .encoder import ResBlock, AttentionBlock


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv (avoids checkerboard artifacts vs transposed conv)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Decoder(nn.Module):
    """
    VAE Decoder: maps latent z back to pixel space.

    Input:  (B, latent_channels, H/8, W/8)
    Output: (B, out_channels, H, W) in [-1, 1]
    """
    def __init__(
        self,
        out_channels: int = 3,
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

        channels = [base_channels * m for m in reversed(channel_multipliers)]
        ch = channels[0]

        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, dropout)
        self.mid_attn   = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, dropout)

        self.up_layers = nn.ModuleList()
        current_res = 32  # Starting resolution after encoding 256→32

        for i, out_ch in enumerate(channels):
            stage = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(ResBlock(ch, out_ch, dropout))
                ch = out_ch
                if current_res in attention_resolutions:
                    stage.append(AttentionBlock(ch))
                else:
                    stage.append(nn.Identity())
            self.up_layers.append(stage)

            if i < len(channels) - 1:
                self.up_layers.append(nn.ModuleList([Upsample(ch)]))
                current_res *= 2

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for stage in self.up_layers:
            for layer in stage:
                h = layer(h)

        h = F.silu(self.norm_out(h))
        return torch.tanh(self.conv_out(h))
