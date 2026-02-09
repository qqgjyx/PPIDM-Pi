"""Tests for ppidm_pi.unet."""

import torch
import pytest

from ppidm_pi.unet import UNet1d


@pytest.fixture
def small_unet():
    """Small U-Net for fast testing."""
    return UNet1d(
        in_channels=12,
        base_channels=16,
        channel_mults=(1, 2, 4, 8),
        num_classes=3,
        time_dim=32,
    )


class TestUNet1d:
    def test_output_shape(self, small_unet, dummy_ecg, dummy_labels, dummy_timesteps):
        """Output shape must match input shape."""
        out = small_unet(dummy_ecg, dummy_timesteps, dummy_labels)
        assert out.shape == dummy_ecg.shape

    def test_conditioning_changes_output(self, small_unet, dummy_ecg, dummy_timesteps):
        """Different class labels should produce different outputs."""
        y0 = torch.zeros(4, dtype=torch.long)
        y1 = torch.ones(4, dtype=torch.long)
        out0 = small_unet(dummy_ecg, dummy_timesteps, y0)
        out1 = small_unet(dummy_ecg, dummy_timesteps, y1)
        assert not torch.allclose(out0, out1)

    def test_timestep_changes_output(self, small_unet, dummy_ecg, dummy_labels):
        """Different timesteps should produce different outputs."""
        t0 = torch.zeros(4, dtype=torch.long)
        t1 = torch.full((4,), 500, dtype=torch.long)
        out0 = small_unet(dummy_ecg, t0, dummy_labels)
        out1 = small_unet(dummy_ecg, t1, dummy_labels)
        assert not torch.allclose(out0, out1)

    def test_gradient_flows(self, small_unet, dummy_ecg, dummy_labels, dummy_timesteps):
        """Gradients should flow to all parameters."""
        out = small_unet(dummy_ecg, dummy_timesteps, dummy_labels)
        loss = out.sum()
        loss.backward()
        for name, p in small_unet.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

    def test_different_sequence_lengths(self, small_unet, dummy_labels, dummy_timesteps):
        """U-Net should handle different input lengths (must be divisible by 16)."""
        x = torch.randn(4, 12, 128)
        out = small_unet(x, dummy_timesteps, dummy_labels)
        assert out.shape == x.shape
