"""
models/diffusion/scheduler.py
------------------------------
Noise schedulers for forward and reverse diffusion processes.

Implements:
  - DDPM (Denoising Diffusion Probabilistic Models, Ho et al. 2020)
  - DDIM (Denoising Diffusion Implicit Models, Song et al. 2020)

Both operate on the same forward process (adding noise to latents).
DDIM provides faster inference by skipping timesteps.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, Optional, List


class NoiseScheduler:
    """
    Beta schedule and diffusion math utilities.
    Shared by both DDPM and DDIM samplers.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
    ):
        self.timesteps = timesteps
        self.prediction_type = prediction_type

        # ── Compute betas ─────────────────────────────────────
        self.betas = self._make_beta_schedule(
            schedule=beta_schedule,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # ── Precompute cumulative products ────────────────────
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]], dim=0
        )

        # Forward process: q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Posterior: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ── Schedule constructors ──────────────────────────────────

    @staticmethod
    def _make_beta_schedule(
        schedule: str,
        timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64).float()

        elif schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal 2021) - better for images
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999).float()

        elif schedule == "scaled_linear":
            # Used in SD (sqrt of linear schedule)
            return torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, timesteps
            ).pow(2).float()

        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    # ── Forward process (q) ────────────────────────────────────

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        q(x_t | x_0): Add noise to clean latent at specified timesteps.

        Args:
            x_start:   (B, C, H, W) clean latent z_0
            noise:     (B, C, H, W) Gaussian noise eps ~ N(0, I)
            timesteps: (B,) integer timestep indices

        Returns:
            x_noisy: (B, C, H, W) noisy latent z_t
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )

        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def get_velocity(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target for v-prediction parameterization.
        v = sqrt(alpha) * eps - sqrt(1 - alpha) * x_0
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha * noise - sqrt_one_minus * x_start

    # ── Prediction target ──────────────────────────────────────

    def get_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the training target based on prediction_type."""
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "v_prediction":
            return self.get_velocity(x_start, noise, timesteps)
        elif self.prediction_type == "sample":
            return x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    # ── Reverse step: DDPM ─────────────────────────────────────

    def ddpm_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        p(x_{t-1} | x_t): One DDPM reverse step.
        Stochastic — adds noise at each step.

        Args:
            model_output: (B, C, H, W) U-Net prediction
            timestep:     current integer timestep
            x_t:          (B, C, H, W) noisy latent at time t

        Returns:
            x_prev: (B, C, H, W) denoised latent at t-1
        """
        t = timestep
        # Predict x_0 from model output
        x_0_pred = self._predict_x0(model_output, x_t, t)

        # Compute posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, torch.tensor([t]), x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, torch.tensor([t]), x_t.shape)
        mean = coef1 * x_0_pred + coef2 * x_t

        # Add noise except at t=0
        if t > 0:
            log_var = self._extract(
                self.posterior_log_variance_clipped, torch.tensor([t]), x_t.shape
            )
            noise = torch.randn_like(x_t)
            x_prev = mean + (0.5 * log_var).exp() * noise
        else:
            x_prev = mean

        return x_prev

    # ── Reverse step: DDIM ─────────────────────────────────────

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse step. Deterministic when eta=0, stochastic when eta>0.
        Allows large step skipping (much faster inference).

        Args:
            model_output:  (B, C, H, W) U-Net prediction
            timestep:      current timestep t
            prev_timestep: next (previous in reverse) timestep t'
            x_t:           (B, C, H, W) noisy latent at t
            eta:           stochasticity (0=DDIM, 1≈DDPM)
        """
        t = timestep
        t_prev = prev_timestep

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        )

        x_0_pred = self._predict_x0(model_output, x_t, t)

        # DDIM direction toward x_t
        sigma = (
            eta
            * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t))
            * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_prev)
        )
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        dir_xt = torch.sqrt(1 - alpha_prod_t_prev - sigma**2) * model_output
        x_prev = torch.sqrt(alpha_prod_t_prev) * x_0_pred + dir_xt + sigma * noise

        return x_prev

    # ── Utilities ──────────────────────────────────────────────

    def _predict_x0(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Predict x_0 from model output (handles all prediction types)."""
        if self.prediction_type == "epsilon":
            sqrt_alpha = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
            return (x_t - sqrt_one_minus * model_output) / sqrt_alpha

        elif self.prediction_type == "v_prediction":
            sqrt_alpha = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
            return sqrt_alpha * x_t - sqrt_one_minus * model_output

        elif self.prediction_type == "sample":
            return model_output

    @staticmethod
    def _extract(
        arr: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: tuple,
    ) -> torch.Tensor:
        """
        Extract values from 1-D array at specified timestep indices,
        then broadcast to spatial dimensions.
        """
        arr = arr.to(timesteps.device)
        out = arr[timesteps.long()]
        while out.ndim < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out.expand(broadcast_shape)

    def make_ddim_timesteps(self, num_inference_steps: int) -> List[int]:
        """
        Uniformly subsample timesteps for DDIM inference.
        Returns list of timesteps in reverse order (T → 0).
        """
        step_ratio = self.timesteps // num_inference_steps
        timesteps = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round().astype(int)[::-1].tolist()
        return timesteps
