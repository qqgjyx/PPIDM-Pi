"""Tests for ppidm_pi.metrics."""

import numpy as np
import torch

from ppidm_pi.metrics import (
    frechet_distance,
    mmd_rbf,
    psd_distance,
    risk_coverage_curve,
    expected_calibration_error,
    brier_score,
)


class TestFrechetDistance:
    def test_zero_for_identical(self):
        """FID should be 0 for identical distributions."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3)
        assert abs(frechet_distance(mu, sigma, mu, sigma)) < 1e-6

    def test_positive_for_different(self):
        mu1 = np.zeros(3)
        mu2 = np.ones(3)
        sigma = np.eye(3)
        assert frechet_distance(mu1, sigma, mu2, sigma) > 0

    def test_symmetric(self):
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([1.0, 1.0])
        s1 = np.eye(2)
        s2 = np.eye(2) * 2
        d12 = frechet_distance(mu1, s1, mu2, s2)
        d21 = frechet_distance(mu2, s2, mu1, s1)
        assert abs(d12 - d21) < 1e-6


class TestMMD:
    def test_zero_for_same_samples(self):
        """MMD should be ~0 for identical sample sets."""
        X = torch.randn(50, 10)
        mmd = mmd_rbf(X, X)
        assert mmd.item() < 1e-5

    def test_positive_for_different(self):
        X = torch.randn(50, 10)
        Y = torch.randn(50, 10) + 5
        mmd = mmd_rbf(X, Y)
        assert mmd.item() > 0

    def test_finite(self):
        X = torch.randn(20, 5)
        Y = torch.randn(20, 5)
        assert torch.isfinite(mmd_rbf(X, Y))


class TestPSDDistance:
    def test_zero_for_identical(self):
        """PSD distance should be ~0 for identical signals."""
        x = np.random.randn(50, 256)
        assert psd_distance(x, x) < 1e-6

    def test_positive_for_different(self):
        x = np.random.randn(50, 256)
        y = np.random.randn(50, 256) * 0.1
        assert psd_distance(x, y) > 0

    def test_finite(self):
        x = np.random.randn(20, 128)
        y = np.random.randn(20, 128)
        assert np.isfinite(psd_distance(x, y, fs=256))


class TestRiskCoverage:
    def test_aurc_range(self):
        risks = np.array([0, 0, 1, 1, 0])
        confs = np.array([0.9, 0.8, 0.3, 0.2, 0.7])
        _, _, aurc = risk_coverage_curve(risks, confs)
        assert 0 <= aurc <= 1

    def test_perfect_classifier(self):
        """Perfect confidence ordering → low AURC."""
        risks = np.array([0, 0, 0, 1, 1])
        confs = np.array([0.99, 0.95, 0.90, 0.10, 0.05])
        _, _, aurc = risk_coverage_curve(risks, confs)
        assert aurc < 0.5

    def test_coverage_shape(self):
        n = 100
        risks = np.random.randint(0, 2, n).astype(float)
        confs = np.random.rand(n)
        coverages, sel_risks, _ = risk_coverage_curve(risks, confs)
        assert len(coverages) == n
        assert len(sel_risks) == n


class TestECE:
    def test_range(self):
        probs = np.random.rand(100)
        labels = np.random.randint(0, 2, 100).astype(float)
        ece = expected_calibration_error(probs, labels)
        assert 0 <= ece <= 1

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions → low ECE."""
        # All predictions are 1.0 confidence and correct
        probs = np.ones(100)
        labels = np.ones(100)
        ece = expected_calibration_error(probs, labels)
        assert ece < 0.1


class TestBrierScore:
    def test_perfect(self):
        probs = np.array([1.0, 0.0, 1.0])
        labels = np.array([1.0, 0.0, 1.0])
        assert brier_score(probs, labels) < 1e-10

    def test_worst(self):
        probs = np.array([0.0, 1.0])
        labels = np.array([1.0, 0.0])
        assert abs(brier_score(probs, labels) - 1.0) < 1e-10

    def test_range(self):
        probs = np.random.rand(100)
        labels = np.random.randint(0, 2, 100).astype(float)
        bs = brier_score(probs, labels)
        assert 0 <= bs <= 1
