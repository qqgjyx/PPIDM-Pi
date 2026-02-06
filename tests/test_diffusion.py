"""Tests for ppidm_pi.diffusion."""

import torch
import pytest

from ppidm_pi.diffusion import (
    CosineSchedule,
    forward_process,
    tweedie_denoise,
    score_matching_loss,
    physics_loss,
    ppidm_loss,
    ppidm_pi_loss,
)
from ppidm_pi.physics import physics_projection


@pytest.fixture
def schedule():
    return CosineSchedule(T=1000)


class TestCosineSchedule:
    def test_alpha_decreasing(self, schedule):
        """α (signal coefficient) should be monotonically decreasing."""
        diffs = schedule.alpha[1:] - schedule.alpha[:-1]
        assert (diffs <= 1e-6).all()

    def test_sigma_increasing(self, schedule):
        """σ (noise coefficient) should be monotonically increasing."""
        diffs = schedule.sigma[1:] - schedule.sigma[:-1]
        assert (diffs >= -1e-6).all()

    def test_alpha_range(self, schedule):
        """α should be in (0, 1]."""
        assert (schedule.alpha > 0).all()
        assert (schedule.alpha <= 1).all()

    def test_sigma_range(self, schedule):
        """σ should be in [0, 1]."""
        assert (schedule.sigma >= 0).all()
        assert (schedule.sigma <= 1).all()

    def test_identity_at_zero(self, schedule):
        """At t=0, signal should dominate: α ≈ 1, σ ≈ 0."""
        assert schedule.alpha[0] > 0.99
        assert schedule.sigma[0] < 0.05

    def test_length(self, schedule):
        assert len(schedule.alpha) == 1000
        assert len(schedule.sigma) == 1000
        assert len(schedule.beta) == 1000


class TestForwardProcess:
    def test_noise_increases_with_t(self, schedule):
        """Later timesteps should produce noisier outputs."""
        x0 = torch.randn(2, 12, 256)
        t_early = torch.tensor([10, 10])
        t_late = torch.tensor([900, 900])
        xt_early, _ = forward_process(x0, t_early, schedule)
        xt_late, _ = forward_process(x0, t_late, schedule)
        # Deviation from x0 should be larger at later timesteps
        dev_early = (xt_early - x0).pow(2).mean()
        dev_late = (xt_late - x0).pow(2).mean()
        assert dev_late > dev_early

    def test_output_shape(self, schedule):
        x0 = torch.randn(4, 12, 256)
        t = torch.randint(0, 1000, (4,))
        xt, eps = forward_process(x0, t, schedule)
        assert xt.shape == x0.shape
        assert eps.shape == x0.shape


class TestTweedie:
    def test_perfect_prediction_recovers_x0(self, schedule):
        """If ε_pred = ε_true, Tweedie should recover x₀ exactly."""
        x0 = torch.randn(2, 12, 256)
        t = torch.tensor([100, 100])
        xt, eps = forward_process(x0, t, schedule)
        x0_hat = tweedie_denoise(xt, t, eps, schedule)
        assert torch.allclose(x0_hat, x0, atol=1e-4)


class TestLosses:
    def test_score_matching_finite(self):
        eps_pred = torch.randn(4, 12, 256)
        eps_true = torch.randn(4, 12, 256)
        loss = score_matching_loss(eps_pred, eps_true)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_score_matching_zero(self):
        eps = torch.randn(4, 12, 256)
        loss = score_matching_loss(eps, eps)
        assert loss.item() < 1e-6

    def test_physics_loss_finite(self, dummy_template):
        x0 = torch.randn(4, 256)
        x0_hat = torch.randn(4, 256)
        loss = physics_loss(x0_hat, x0, dummy_template)
        assert torch.isfinite(loss)

    def test_ppidm_loss_finite(self, dummy_template):
        eps = torch.randn(4, 12, 256)
        x0 = torch.randn(4, 256)
        loss = ppidm_loss(eps, eps + 0.1, x0 + 0.1, x0, dummy_template)
        assert torch.isfinite(loss)

    def test_ppidm_pi_loss_finite(self, dummy_template):
        eps = torch.randn(4, 12, 256)
        x0 = torch.randn(4, 256)
        pi = torch.tensor([0.5, 0.8, 0.3, 0.9])
        loss = ppidm_pi_loss(eps, eps + 0.1, x0 + 0.1, x0, dummy_template, pi)
        assert torch.isfinite(loss)

    def test_pi_weighting_effect(self, dummy_template):
        """ppidm_pi_loss = L_score + α · mean(Π^β · L_phys_per_sample)."""
        eps_pred = torch.randn(4, 12, 256)
        eps_true = torch.randn(4, 12, 256)
        x0 = torch.randn(4, 256)
        x0_hat = x0 + 0.5
        pi = torch.tensor([0.3, 0.7, 0.5, 0.9])
        alpha_val = 0.5
        beta_val = 2.0

        total = ppidm_pi_loss(
            eps_pred, eps_true, x0_hat, x0, dummy_template,
            pi, alpha=alpha_val, beta=beta_val,
        )
        l_score = score_matching_loss(eps_pred, eps_true)
        proj_hat = physics_projection(x0_hat, dummy_template)
        proj_true = physics_projection(x0, dummy_template)
        per_sample = (proj_hat - proj_true).pow(2).mean(dim=-1)
        expected = l_score + alpha_val * ((pi ** beta_val) * per_sample).mean()
        assert torch.allclose(total, expected, atol=1e-6)
