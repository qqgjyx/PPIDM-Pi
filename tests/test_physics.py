"""Tests for ppidm_pi.physics."""

import torch
import pytest

from ppidm_pi.physics import (
    nernst_potential,
    electrolyte_to_concentration,
    mcsharry_template,
    physics_projection,
    physics_residual,
    plausibility_index,
)


class TestNernst:
    def test_potassium_equilibrium(self):
        """E_K should be about -90 mV at physiological conditions."""
        E = nernst_potential(conc_out=4.0, conc_in=140.0, z=1, T=310.0)
        assert -0.12 < E < -0.07  # roughly -96 to -80 mV

    def test_sign_reversal(self):
        """Reversing conc_out/conc_in should flip the sign."""
        E1 = nernst_potential(4.0, 140.0)
        E2 = nernst_potential(140.0, 4.0)
        assert abs(E1 + E2) < 1e-10

    def test_divalent_smaller(self):
        """z=2 should halve the potential magnitude."""
        E1 = nernst_potential(4.0, 140.0, z=1)
        E2 = nernst_potential(4.0, 140.0, z=2)
        assert abs(E2 - E1 / 2) < 1e-10


class TestElectrolyteConcentration:
    @pytest.mark.parametrize("electrolyte,bin_idx,expected", [
        ("K", 0, 3.0), ("K", 1, 4.25), ("K", 2, 5.5),
        ("Ca", 1, 9.5), ("Mg", 1, 1.95),
    ])
    def test_mapping(self, electrolyte, bin_idx, expected):
        assert electrolyte_to_concentration(bin_idx, electrolyte) == expected


class TestMcSharryTemplate:
    def test_shape(self):
        t = mcsharry_template("K", 1, length=256)
        assert t.shape == (256,)

    def test_different_lengths(self):
        t128 = mcsharry_template("K", 1, length=128)
        t512 = mcsharry_template("K", 1, length=512)
        assert t128.shape == (128,)
        assert t512.shape == (512,)

    def test_electrolyte_modulation(self):
        """Different bins should produce different templates."""
        t_low = mcsharry_template("K", 0)
        t_norm = mcsharry_template("K", 1)
        t_high = mcsharry_template("K", 2)
        assert not torch.allclose(t_low, t_norm)
        assert not torch.allclose(t_norm, t_high)

    def test_all_electrolytes(self):
        """Templates should generate for all electrolyte types."""
        for e in ["K", "Ca", "Mg"]:
            t = mcsharry_template(e, 1)
            assert t.shape == (256,)
            assert torch.isfinite(t).all()


class TestProjection:
    def test_decomposition(self, dummy_template):
        """x = K(x) + U(x) must hold."""
        x = torch.randn(256)
        K = physics_projection(x, dummy_template)
        U = physics_residual(x, dummy_template)
        assert torch.allclose(x, K + U, atol=1e-5)

    def test_orthogonality(self, dummy_template):
        """K and U should be orthogonal: <K, U> ≈ 0."""
        x = torch.randn(256)
        K = physics_projection(x, dummy_template)
        U = physics_residual(x, dummy_template)
        dot = torch.sum(K * U)
        assert abs(dot.item()) < 1e-4

    def test_idempotent(self, dummy_template):
        """Projecting twice should give the same result: K(K(x)) = K(x)."""
        x = torch.randn(256)
        K = physics_projection(x, dummy_template)
        KK = physics_projection(K, dummy_template)
        assert torch.allclose(K, KK, atol=1e-5)

    def test_batch_support(self, dummy_template):
        """Projection should work on batched inputs."""
        x = torch.randn(4, 256)
        K = physics_projection(x, dummy_template)
        assert K.shape == (4, 256)


class TestPlausibilityIndex:
    def test_range(self, dummy_template):
        """Π must be in [0, 1]."""
        x = torch.randn(10, 256)
        pi = plausibility_index(x, dummy_template)
        assert (pi >= 0).all() and (pi <= 1).all()

    def test_perfect_alignment(self, dummy_template):
        """If x is proportional to template, Π should be ~1."""
        x = dummy_template * 3.14
        pi = plausibility_index(x, dummy_template)
        assert pi.item() > 0.999

    def test_orthogonal_signal(self, dummy_template):
        """If x is orthogonal to template, Π should be ~0."""
        # Create a vector orthogonal to the template
        x = torch.randn(256)
        # Remove the template component
        x = physics_residual(x, dummy_template)
        pi = plausibility_index(x, dummy_template)
        assert pi.item() < 0.001
