"""Evaluation metrics: FID, MMD, PSD distance, AURC, ECE, Brier score."""

import numpy as np
import torch
from scipy import linalg, signal


# ---------------------------------------------------------------------------
# Fréchet distance (FID-style)
# ---------------------------------------------------------------------------
def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Fréchet distance between two multivariate Gaussians.

    FID = ||μ₁ − μ₂||² + Tr(Σ₁ + Σ₂ − 2(Σ₁Σ₂)^{1/2})
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    # Numerical stability — discard imaginary part
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(
        diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    )


# ---------------------------------------------------------------------------
# MMD with RBF kernel
# ---------------------------------------------------------------------------
def mmd_rbf(
    X: torch.Tensor, Y: torch.Tensor, bandwidth: float | None = None
) -> torch.Tensor:
    """Maximum Mean Discrepancy with Gaussian RBF kernel.

    Uses the median heuristic for bandwidth if not specified.
    """
    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2

    if bandwidth is None:
        all_dists = torch.cat([XX.reshape(-1), YY.reshape(-1), XY.reshape(-1)])
        bandwidth = all_dists.median().item()
        if bandwidth < 1e-8:
            bandwidth = 1.0

    gamma = 1.0 / (2 * bandwidth)
    Kxx = torch.exp(-gamma * XX).mean()
    Kyy = torch.exp(-gamma * YY).mean()
    Kxy = torch.exp(-gamma * XY).mean()

    return Kxx + Kyy - 2 * Kxy


# ---------------------------------------------------------------------------
# Power Spectral Density distance
# ---------------------------------------------------------------------------
def psd_distance(
    real: np.ndarray, generated: np.ndarray, fs: int = 500
) -> float:
    """Mean absolute log-PSD distance (Welch method).

    Args:
        real: ``[N, L]`` array of real signals.
        generated: ``[N, L]`` array of generated signals.
        fs: Sampling frequency in Hz.
    """
    nperseg = min(256, real.shape[-1])
    _, psd_r = signal.welch(real, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_g = signal.welch(generated, fs=fs, nperseg=nperseg, axis=-1)

    log_r = np.log(psd_r.mean(axis=0) + 1e-12)
    log_g = np.log(psd_g.mean(axis=0) + 1e-12)

    return float(np.mean(np.abs(log_r - log_g)))


# ---------------------------------------------------------------------------
# Risk-coverage curve and AURC
# ---------------------------------------------------------------------------
def risk_coverage_curve(
    risks: np.ndarray, confidences: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute risk-coverage curve and AURC.

    Args:
        risks: Per-sample risk (e.g. 0/1 misclassification indicator).
        confidences: Per-sample confidence score (higher = more confident).

    Returns:
        (coverages, selective_risks, aurc)
    """
    order = np.argsort(-confidences)
    risks_sorted = risks[order]
    n = len(risks)
    coverages = np.arange(1, n + 1) / n
    selective_risks = np.cumsum(risks_sorted) / np.arange(1, n + 1)
    aurc = float(np.trapz(selective_risks, coverages))
    return coverages, selective_risks, aurc


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities for the true class ``[N]``.
        labels: Binary correctness indicators ``[N]`` (1 = correct).
        n_bins: Number of calibration bins.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------
def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean (p − y)²."""
    return float(np.mean((probs - labels) ** 2))
