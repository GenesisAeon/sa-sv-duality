"""S_A — Action Entropy: entropy produced along a UTAC trajectory.

S_A = ∫₀ᵀ σ_s(H(t), Γ(t)) dt

where the entropy production rate σ_s is derived from the UTAC ODE:
    σ_s(H, Γ) = r · H · (1 - H/K) · tanh(σ · Γ) · Φ'(H)

High S_A → irreversible, costly transition (ordered process).
Low  S_A → reversible, efficient transition (near equilibrium).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import SIGMA_UTAC


@dataclass
class UTACParams:
    """Parameters for the UTAC logistic ODE."""

    r: float = 1.0    # growth rate
    K: float = 1.0    # carrying capacity
    sigma: float = SIGMA_UTAC
    gamma: float = 0.251  # CREP coupling value


def utac_entropy_rate(
    H: float,
    gamma: float,
    *,
    r: float = 1.0,
    K: float = 1.0,
    sigma: float = SIGMA_UTAC,
) -> float:
    """Entropy production rate along UTAC trajectory at state H.

    σ_s = r · |H · (1 - H/K) · tanh(σΓ)|

    The absolute value ensures σ_s ≥ 0 (second law).
    """
    if K <= 0:
        raise ValueError("K must be positive")
    logistic = H * (1.0 - H / K)
    coupling = math.tanh(sigma * gamma)
    return r * abs(logistic * coupling)


class ActionEntropy:
    """Computes S_A = ∫ σ_s(H(t), Γ(t)) dt via trapezoidal integration.

    Args:
        params: UTAC parameters (r, K, sigma, gamma).
    """

    def __init__(self, params: UTACParams | None = None) -> None:
        self.params = params or UTACParams()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def entropy_rate(self, H: float, gamma: float | None = None) -> float:
        """Instantaneous entropy production rate σ_s at state H."""
        g = gamma if gamma is not None else self.params.gamma
        return utac_entropy_rate(
            H, g, r=self.params.r, K=self.params.K, sigma=self.params.sigma
        )

    def integrate(
        self,
        trajectory: Sequence[float],
        times: Sequence[float] | None = None,
        gamma_series: Sequence[float] | None = None,
    ) -> float:
        """Integrate entropy production along a trajectory H(t).

        Args:
            trajectory: H values along the path.
            times: Corresponding time points. If None, assumes unit spacing.
            gamma_series: Γ values per time step. If None, uses params.gamma.

        Returns:
            S_A ≥ 0: total action entropy.
        """
        H_arr: NDArray[np.float64] = np.asarray(trajectory, dtype=float)
        t_arr = (
            np.arange(len(H_arr), dtype=float)
            if times is None
            else np.asarray(times, dtype=float)
        )

        if len(H_arr) != len(t_arr):
            raise ValueError("trajectory and times must have the same length")

        if gamma_series is None:
            g_arr = np.full(len(H_arr), self.params.gamma)
        else:
            g_arr = np.asarray(gamma_series, dtype=float)
            if len(g_arr) != len(H_arr):
                raise ValueError("gamma_series must match trajectory length")

        rates = np.array(
            [self.entropy_rate(float(h), float(g)) for h, g in zip(H_arr, g_arr, strict=False)]
        )
        from scipy.integrate import trapezoid

        return float(trapezoid(rates, t_arr))

    # ------------------------------------------------------------------
    # Trajectory generation (UTAC ODE, simple Euler)
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        H0: float = 0.1,
        T: float = 10.0,
        n_steps: int = 1000,
        gamma: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate UTAC ODE forward and return (times, H_values).

        dH/dt = r · H · (1 - H/K) · tanh(σΓ)
        """
        g = gamma if gamma is not None else self.params.gamma
        dt = T / n_steps
        t = np.linspace(0.0, T, n_steps + 1)
        H = np.empty(n_steps + 1)
        H[0] = H0
        coupling = math.tanh(self.params.sigma * g)

        for i in range(n_steps):
            dH = self.params.r * H[i] * (1.0 - H[i] / self.params.K) * coupling
            H[i + 1] = max(0.0, H[i] + dt * dH)

        return t, H

    def from_utac_ode(
        self,
        H0: float = 0.1,
        T: float = 10.0,
        n_steps: int = 1000,
        gamma: float | None = None,
    ) -> float:
        """Convenience: generate UTAC trajectory and compute S_A."""
        t, H = self.generate_trajectory(H0=H0, T=T, n_steps=n_steps, gamma=gamma)
        return self.integrate(H.tolist(), times=t.tolist())
