"""S_V — Volume Entropy: Shannon entropy of the phase-space state distribution.

S_V = -∫ p(H) · ln p(H) dH

High S_V → many equally probable states (maximum uncertainty, equilibrium).
Low  S_V → concentrated around one attractor (ordered, certain).

For the UTAC ODE the stationary distribution p*(H) is derived from the
Fokker-Planck equation associated with the stochastic version of the logistic
UTAC dynamics.  In the deterministic limit we use a kernel-density estimate
over the trajectory.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import Q4_BITS, Q4_N_STATES


class VolumeEntropy:
    """Computes S_V from a trajectory or a probability density.

    Two modes:
      - *kde*: estimate p(H) via Gaussian KDE over trajectory samples.
      - *discrete*: given an explicit probability vector p_i sum(-p·log p).
    """

    def __init__(self, n_bins: int = 64, bandwidth: float | None = None) -> None:
        self.n_bins = n_bins
        self.bandwidth = bandwidth

    # ------------------------------------------------------------------
    # Continuous / KDE
    # ------------------------------------------------------------------

    def from_trajectory(self, trajectory: Sequence[float]) -> float:
        """Estimate S_V via histogram → Shannon entropy (nats).

        Args:
            trajectory: H(t) time series.

        Returns:
            S_V ≥ 0 in nats.
        """
        H = np.asarray(trajectory, dtype=float)
        if len(H) < 2:
            return 0.0

        counts, _ = np.histogram(H, bins=self.n_bins, density=False)
        probs = counts / counts.sum()
        # drop zero-probability bins
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def from_density(self, p: Sequence[float]) -> float:
        """Compute Shannon entropy from an explicit probability vector.

        Args:
            p: probability distribution (will be re-normalised).

        Returns:
            S_V in nats.
        """
        arr = np.asarray(p, dtype=float)
        arr = arr / arr.sum()
        arr = arr[arr > 0]
        return float(-np.sum(arr * np.log(arr)))

    # ------------------------------------------------------------------
    # Analytical / UTAC stationary distribution
    # ------------------------------------------------------------------

    @staticmethod
    def stationary_utac(
        r: float = 1.0,
        K: float = 1.0,
        sigma: float = 2.2,
        gamma: float = 0.251,
        n_points: int = 200,
    ) -> float:
        """Approximate S_V for the UTAC stationary distribution.

        The UTAC fixed point H* = K · tanh(σΓ) acts as a delta-like
        attractor in the deterministic limit.  We model the stationary
        distribution as a Gaussian centred at H* with width ∝ 1/r to
        capture diffusive fluctuations.
        """
        coupling = math.tanh(sigma * gamma)
        H_star = K * coupling
        width = max(K / (4.0 * r * math.sqrt(float(n_points))), 1e-6)

        H_grid = np.linspace(max(0.0, H_star - 4 * width), H_star + 4 * width, n_points)
        p = np.exp(-0.5 * ((H_grid - H_star) / width) ** 2)
        p /= p.sum()
        return float(-np.sum(p[p > 0] * np.log(p[p > 0])))

    # ------------------------------------------------------------------
    # Q4 discrete entropies
    # ------------------------------------------------------------------

    @staticmethod
    def q4_state_entropy(state_bits: int) -> float:
        """S_V for a Q4 state encoded as a 4-bit integer in [0, 15].

        Interpretation: each set bit contributes one active channel.
        S_V = ln(n_active + 1) — monotone proxy for state richness.
        """
        if not 0 <= state_bits <= Q4_N_STATES - 1:
            raise ValueError(f"state_bits must be in [0, {Q4_N_STATES - 1}]")
        n_active = bin(state_bits).count("1")
        return math.log(n_active + 1)  # ln(1)=0 for 0000, ln(5)≈1.61 for 1111

    @classmethod
    def q4_entropy_vector(cls) -> NDArray[np.float64]:
        """Returns S_V for all 16 Q4 states, ordered 0x0 … 0xF."""
        return np.array([cls.q4_state_entropy(i) for i in range(Q4_N_STATES)])
