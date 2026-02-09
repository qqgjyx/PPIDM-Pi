"""VP-SDE diffusion: cosine schedule, losses, and DDPM sampler.

Implements the forward noising process, Tweedie denoising,
score-matching and physics-informed losses, and the full
reverse DDPM sampling loop.
"""

import math

import torch
import torch.nn as nn
from tqdm import tqdm

from ppidm_pi.physics import physics_projection


# ---------------------------------------------------------------------------
# Cosine noise schedule
# ---------------------------------------------------------------------------
class CosineSchedule:
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    Provides α_t (signal coefficient), σ_t (noise coefficient), and β_t.
    All quantities are pre-computed and stored as 1-D tensors of length *T*.
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((t / T + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = f / f[0]

        # Clip β to avoid instability near t = T
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        beta = beta.clamp(max=0.999)

        alpha_bar = alpha_bar[1:]  # length T, index 0 → step 1
        self.alpha_bar = alpha_bar.float()
        self.sigma = torch.sqrt(1.0 - self.alpha_bar).float()
        self.alpha = torch.sqrt(self.alpha_bar).float()
        self.beta = beta.float()

    def to(self, device: torch.device) -> "CosineSchedule":
        if self.alpha_bar.device == device:
            return self
        self.alpha_bar = self.alpha_bar.to(device)
        self.sigma = self.sigma.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        return self


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------
def forward_process(
    x0: torch.Tensor, t: torch.Tensor, schedule: CosineSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add noise: x_t = α_t · x0 + σ_t · ε.

    Args:
        x0: Clean data ``[B, C, L]``.
        t: Timestep indices ``[B]`` (0-indexed, range [0, T-1]).
        schedule: Noise schedule.

    Returns:
        (x_t, ε) — noisy data and the sampled noise.
    """
    alpha = schedule.alpha[t][:, None, None]
    sigma = schedule.sigma[t][:, None, None]
    eps = torch.randn_like(x0)
    return alpha * x0 + sigma * eps, eps


# ---------------------------------------------------------------------------
# Tweedie one-step denoising
# ---------------------------------------------------------------------------
def tweedie_denoise(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    schedule: CosineSchedule,
) -> torch.Tensor:
    """Tweedie estimate: x̂₀ = (x_t − σ_t · ε_θ) / α_t."""
    alpha = schedule.alpha[t][:, None, None]
    sigma = schedule.sigma[t][:, None, None]
    return (xt - sigma * eps_pred) / alpha


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def score_matching_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    """Standard denoising score matching loss: ||ε_θ − ε||²."""
    return nn.functional.mse_loss(eps_pred, eps_true)


def physics_loss(
    x0_hat: torch.Tensor, x0: torch.Tensor, template: torch.Tensor
) -> torch.Tensor:
    """Physics projection loss: ||K(x̂₀; y) − K(x₀; y)||²."""
    return nn.functional.mse_loss(
        physics_projection(x0_hat, template),
        physics_projection(x0, template),
    )


def ppidm_loss(
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
    x0_hat: torch.Tensor,
    x0: torch.Tensor,
    template: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """PPIDM loss: L_score + α · L_phys."""
    return score_matching_loss(eps_pred, eps_true) + alpha * physics_loss(
        x0_hat, x0, template
    )


def ppidm_pi_loss(
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
    x0_hat: torch.Tensor,
    x0: torch.Tensor,
    template: torch.Tensor,
    pi: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 1.0,
) -> torch.Tensor:
    """Π-reweighted PPIDM loss: E[L_score + α · Π^β · L_phys]."""
    weights = pi ** beta                                         # [B]
    proj_hat = physics_projection(x0_hat, template)              # [B, L]
    proj_true = physics_projection(x0, template)                 # [B, L]
    per_sample_phys = (proj_hat - proj_true).pow(2).mean(dim=-1) # [B]
    weighted_phys = (weights * per_sample_phys).mean()           # scalar
    return score_matching_loss(eps_pred, eps_true) + alpha * weighted_phys


# ---------------------------------------------------------------------------
# DDPM reverse sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: CosineSchedule,
    shape: tuple[int, ...],
    y: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Full DDPM reverse diffusion loop (T steps).

    Args:
        model: Score network ``ε_θ(x_t, t, y)``.
        schedule: Cosine noise schedule.
        shape: ``(B, C, L)`` output shape.
        y: Class labels ``[B]``.
        device: Target device.

    Returns:
        Generated samples ``[B, C, L]``.
    """
    x = torch.randn(shape, device=device)
    schedule = schedule.to(device)

    for i in tqdm(reversed(range(schedule.T)), total=schedule.T, desc="DDPM sampling"):
        t_batch = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch, y)

        beta_t = schedule.beta[i]
        alpha_bar_t = schedule.alpha_bar[i]

        # Predicted mean
        x = (1 / torch.sqrt(1 - beta_t)) * (
            x - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred
        )

        # Add noise for all steps except the last
        if i > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)

    return x
