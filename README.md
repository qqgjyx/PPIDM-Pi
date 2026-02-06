# PPIDM-Pi

**Plausibility-Index Partial Physics-Informed Diffusion Model for ECG-Electrolyte Inference**

This repository contains the core model code for PPIDM-Pi, a diffusion model that integrates cardiac electrophysiology priors (Nernst equation, McSharry ECG model) into the score-matching objective via a learned plausibility index.

## Architecture Overview

```
  ECG x_0 ──► Forward Process (VP-SDE) ──► x_t
                                              │
                   ┌──────────────────────────┘
                   ▼
            ┌─────────────┐    t (timestep)
            │  1-D U-Net  │◄── y (electrolyte class)
            │   ε_θ(x,t,y)│
            └──────┬──────┘
                   │  ε_pred
                   ▼
            Tweedie ──► x̂_0
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
   K(x̂_0; y)              U(x̂_0; y)
   physics proj.           residual
       │                       │
       └───────┬───────────────┘
               ▼
        Π = ||K||²/(||K||²+||U||²)
               │
               ▼
  Loss = L_score + α · Π^β · L_phys
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/PPIDM-Pi.git
cd PPIDM-Pi

# Install in development mode (requires PyTorch, NumPy, SciPy, tqdm)
pip install -e .

# For running tests
pip install -e ".[test]"
```

## Module Overview

| Module | Description |
|--------|-------------|
| `ppidm_pi.physics` | Nernst equation, McSharry ODE ECG template, projection operators K/U, plausibility index Π |
| `ppidm_pi.unet` | 1-D U-Net score network with sinusoidal timestep embedding and class-conditioned AdaptiveGroupNorm |
| `ppidm_pi.diffusion` | Cosine noise schedule, forward process, Tweedie denoising, score-matching + physics losses, DDPM sampler |
| `ppidm_pi.classifier` | 1-D ResNet-18 for downstream ECG electrolyte classification |
| `ppidm_pi.data` | MIMIC-IV-ECG dataset stub with preprocessing documentation |
| `ppidm_pi.metrics` | FID, MMD, PSD distance, AURC, ECE, Brier score |

## Data

This project uses the [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) dataset from PhysioNet (credentialed access required).

**Preprocessing steps:**

1. Download 12-lead ECGs (500 Hz, 10 seconds) from MIMIC-IV-ECG
2. Segment into individual beats and resample to 256 samples
3. Match with lab results from MIMIC-IV Clinical (chartevents table)
4. Bin electrolyte values into low/normal/high:
   - K⁺: <3.5 / 3.5–5.0 / >5.0 mEq/L
   - Ca²⁺: <8.5 / 8.5–10.5 / >10.5 mg/dL
   - Mg²⁺: <1.7 / 1.7–2.2 / >2.2 mEq/L

## Quick Start

```python
import torch
from ppidm_pi import (
    mcsharry_template,
    physics_projection,
    physics_residual,
    plausibility_index,
)

# Generate a physics-informed ECG template for normal potassium
template = mcsharry_template(electrolyte="K", bin_idx=1, length=256)
print(template.shape)  # torch.Size([256])

# Decompose a synthetic ECG into physics/residual components
ecg = torch.randn(256)
K = physics_projection(ecg, template)   # physics-aligned component
U = physics_residual(ecg, template)     # residual component
pi = plausibility_index(ecg, template)  # Π ∈ [0, 1]

# Verify orthogonal decomposition
assert torch.allclose(ecg, K + U, atol=1e-5)
print(f"Plausibility index: {pi.item():.4f}")
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
