"""Physics-informed components for ECG-electrolyte modelling.

Implements the Nernst equation, McSharry synthetic ECG ODE with
electrolyte modulation, physics-based projection operators K and U,
and the plausibility index Π.
"""

import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R = 8.314          # J/(mol·K)
F = 96485.0        # C/mol

# Default extracellular / intracellular concentrations (mM)
_DEFAULT_CONC = {
    "K":  {"out": 4.0,  "in": 140.0, "z": 1},
    "Ca": {"out": 2.4,  "in": 0.0001, "z": 2},  # ~100 nM intracellular
    "Mg": {"out": 0.8,  "in": 0.5, "z": 2},
}

# Bin boundaries: low / normal / high  →  representative concentrations
ELECTROLYTE_BINS = {
    "K":  {"edges": [3.5, 5.0], "centers": [3.0, 4.25, 5.5]},
    "Ca": {"edges": [8.5, 10.5], "centers": [7.5, 9.5, 11.5]},
    "Mg": {"edges": [1.7, 2.2], "centers": [1.4, 1.95, 2.5]},
}


# ---------------------------------------------------------------------------
# Nernst equation
# ---------------------------------------------------------------------------
def nernst_potential(
    conc_out: float, conc_in: float, z: int = 1, T: float = 310.0
) -> float:
    """Equilibrium potential via the Nernst equation (volts).

    E = (RT / zF) ln(conc_out / conc_in)
    """
    return (R * T) / (z * F) * math.log(conc_out / conc_in)


# ---------------------------------------------------------------------------
# Bin → concentration mapping
# ---------------------------------------------------------------------------
def electrolyte_to_concentration(bin_idx: int, electrolyte: str) -> float:
    """Map a discrete bin index {0, 1, 2} to a representative concentration."""
    return ELECTROLYTE_BINS[electrolyte]["centers"][bin_idx]


# ---------------------------------------------------------------------------
# McSharry synthetic ECG ODE with electrolyte modulation
# ---------------------------------------------------------------------------

# Default McSharry parameters for one heartbeat (PQRST peaks)
#   (theta_i, a_i, b_i) for P, Q, R, S, T waves
_PEAKS = {
    "theta": np.array([-60, -15, 0, 15, 90]) * (np.pi / 180),
    "a":     np.array([1.2, -5.0, 30.0, -7.5, 0.75]),
    "b":     np.array([0.25, 0.1, 0.1, 0.1, 0.4]),
}


def _mcsharry_ode(t, state, omega, theta, a, b):
    """McSharry ODE: dx/dt, dy/dt, dz/dt for synthetic ECG."""
    x, y, z = state
    alpha = 1.0 - math.sqrt(x**2 + y**2)
    current_theta = math.atan2(y, x)

    dx = alpha * x - omega * y
    dy = alpha * y + omega * x

    dz = 0.0
    for i in range(len(a)):
        d_theta = current_theta - theta[i]
        # Wrap to [-pi, pi]
        d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
        dz += -a[i] * d_theta * math.exp(-0.5 * (d_theta / b[i]) ** 2)
    dz -= z

    return [dx, dy, dz]


def mcsharry_template(
    electrolyte: str,
    bin_idx: int,
    length: int = 256,
    heart_rate: float = 72.0,
) -> torch.Tensor:
    """Generate a single-lead McSharry ECG template modulated by electrolyte level.

    Electrolyte effects:
      - K⁺:  T-wave amplitude scaling  a_T' = a_T * (1 + 0.15 * ([K⁺] - 4.0))
      - Ca²⁺: QT interval shift  Δθ_T = -0.02 * ([Ca²⁺] - 9.0)  (rad)
      - Mg²⁺: Cofactor — attenuates K⁺/Ca²⁺ effects via T-wave width

    Returns:
        Tensor of shape ``[length]``
    """
    conc = electrolyte_to_concentration(bin_idx, electrolyte)

    theta = _PEAKS["theta"].copy()
    a = _PEAKS["a"].copy()
    b = _PEAKS["b"].copy()

    if electrolyte == "K":
        a[4] *= 1.0 + 0.15 * (conc - 4.0)
    elif electrolyte == "Ca":
        theta[4] += -0.02 * (conc - 9.0)
    elif electrolyte == "Mg":
        b[4] *= 1.0 + 0.05 * (conc - 2.0)

    omega = 2.0 * math.pi * heart_rate / 60.0
    t_span = (0.0, 60.0 / heart_rate)
    t_eval = np.linspace(t_span[0], t_span[1], length)

    sol = solve_ivp(
        _mcsharry_ode,
        t_span,
        y0=[1.0, 0.0, 0.0],
        t_eval=t_eval,
        args=(omega, theta, a, b),
        method="RK45",
        rtol=1e-6,
    )

    z = sol.y[2]  # ECG-like signal
    return torch.from_numpy(z.astype(np.float32))


# ---------------------------------------------------------------------------
# Projection operators
# ---------------------------------------------------------------------------
def physics_projection(x: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Project x onto the template subspace: K(x; y) = <x, t> / ||t||² · t.

    Args:
        x: ``[..., L]`` ECG signals.
        template: ``[L]`` physics template.

    Returns:
        Projection tensor, same shape as *x*.
    """
    t = template
    t_norm_sq = torch.sum(t * t) + 1e-8
    coeff = torch.sum(x * t, dim=-1, keepdim=True) / t_norm_sq
    return coeff * t


def physics_residual(x: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Residual: U(x; y) = x - K(x; y)."""
    return x - physics_projection(x, template)


def plausibility_index(x: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Plausibility index: Π = ||K||² / (||K||² + ||U||²).

    Returns:
        Per-sample tensor in [0, 1] (scalar for unbatched input).
    """
    k = physics_projection(x, template)
    u = x - k
    k_sq = torch.sum(k * k, dim=-1)
    u_sq = torch.sum(u * u, dim=-1)
    return k_sq / (k_sq + u_sq + 1e-8)
