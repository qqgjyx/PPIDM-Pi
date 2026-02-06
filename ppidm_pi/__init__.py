"""PPIDM-Pi: Plausibility-Index Partial Physics-Informed Diffusion Model."""

__version__ = "0.1.0"

from .physics import (
    electrolyte_to_concentration,
    mcsharry_template,
    nernst_potential,
    physics_projection,
    physics_residual,
    plausibility_index,
)
from .unet import UNet1d
from .diffusion import CosineSchedule, ddpm_sample, forward_process, ppidm_pi_loss
from .classifier import ResNet1d
from .metrics import (
    brier_score,
    expected_calibration_error,
    frechet_distance,
    mmd_rbf,
    psd_distance,
    risk_coverage_curve,
)

__all__ = [
    "__version__",
    # physics
    "nernst_potential",
    "electrolyte_to_concentration",
    "mcsharry_template",
    "physics_projection",
    "physics_residual",
    "plausibility_index",
    # model
    "UNet1d",
    "ResNet1d",
    # diffusion
    "CosineSchedule",
    "forward_process",
    "ddpm_sample",
    "ppidm_pi_loss",
    # metrics
    "frechet_distance",
    "mmd_rbf",
    "psd_distance",
    "risk_coverage_curve",
    "expected_calibration_error",
    "brier_score",
]
