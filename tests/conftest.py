"""Shared test fixtures."""

import torch
import pytest

from ppidm_pi.physics import mcsharry_template


@pytest.fixture(autouse=True)
def _set_seed():
    """Seed RNG for deterministic tests."""
    torch.manual_seed(42)


@pytest.fixture
def dummy_ecg():
    """Batch of random ECG signals [B=4, C=12, L=256]."""
    return torch.randn(4, 12, 256)


@pytest.fixture
def dummy_labels():
    """Batch of class labels [B=4], values in {0, 1, 2}."""
    return torch.randint(0, 3, (4,))


@pytest.fixture(scope="module")
def dummy_template():
    """McSharry ECG template for normal potassium, shape [256]."""
    return mcsharry_template(electrolyte="K", bin_idx=1, length=256)


@pytest.fixture
def dummy_timesteps():
    """Random diffusion timesteps [B=4] in [0, 999]."""
    return torch.randint(0, 1000, (4,))
