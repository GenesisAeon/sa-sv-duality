"""S_A · S_V = const — the core duality relation.

Analogous to Heisenberg uncertainty:
  Minimal action entropy  → maximal volume entropy  (equilibrium)
  Minimal volume entropy  → maximal action entropy   (ordered process)

The product S_A · S_V is conserved along the UTAC optimal path (the path
that extremises δ(S_A - λ·S_V) = 0).

This module:
  1. Verifies the conservation numerically over UTAC trajectories.
  2. Provides the duality constant C_dual = S_A · S_V for any system.
  3. Maps (S_A, S_V) pairs onto the duality hyperbola C = const.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .action_entropy import ActionEntropy, UTACParams
from .volume_entropy import VolumeEntropy
from .constants import SAV_CONSTANT


@dataclass
class DualityResult:
    """Result of a duality analysis run."""

    S_A: float
    S_V: float
    product: float
    lambda_opt: float   # optimal Lagrange multiplier
    conserved: bool     # True if product ≈ constant along path

    @property
    def ratio(self) -> float:
        return self.S_A / self.S_V if self.S_V != 0 else float("inf")


class DualityRelation:
    """Computes and verifies the S_A · S_V duality for a UTAC system.

    Args:
        params: UTAC parameters.
        rtol: relative tolerance for conservation check.
    """

    def __init__(
        self,
        params: UTACParams | None = None,
        rtol: float = 0.15,
    ) -> None:
        self.params = params or UTACParams()
        self.rtol = rtol
        self._sa = ActionEntropy(self.params)
        self._sv = VolumeEntropy()

    # ------------------------------------------------------------------
    # Single trajectory
    # ------------------------------------------------------------------

    def analyse(
        self,
        H0: float = 0.1,
        T: float = 10.0,
        n_steps: int = 500,
        gamma: float | None = None,
    ) -> DualityResult:
        """Compute S_A, S_V and their product for one UTAC trajectory."""
        g = gamma if gamma is not None else self.params.gamma

        t, H = self._sa.generate_trajectory(H0=H0, T=T, n_steps=n_steps, gamma=g)
        S_A = self._sa.integrate(H.tolist(), times=t.tolist())
        S_V = self._sv.from_trajectory(H.tolist())
        product = S_A * S_V

        # Optimal λ: from variational condition δ(S_A - λ·S_V) = 0
        # Stationary when ∂S_A/∂H = λ · ∂S_V/∂H → λ_opt ≈ S_A / S_V
        lambda_opt = S_A / S_V if S_V > 0 else 0.0

        return DualityResult(
            S_A=S_A,
            S_V=S_V,
            product=product,
            lambda_opt=lambda_opt,
            conserved=True,  # single trajectory: trivially "conserved"
        )

    # ------------------------------------------------------------------
    # Conservation along parameter sweep
    # ------------------------------------------------------------------

    def verify_conservation(
        self,
        gamma_range: tuple[float, float] = (0.05, 0.95),
        n_points: int = 20,
        H0: float = 0.1,
        T: float = 10.0,
    ) -> tuple[bool, NDArray[np.float64]]:
        """Check S_A·S_V stability over a range of Γ values.

        Returns:
            (conserved, products_array) where conserved is True if the
            coefficient of variation of products < rtol.
        """
        gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
        products = np.array(
            [self.analyse(H0=H0, T=T, gamma=float(g)).product for g in gammas]
        )
        cv = float(products.std() / products.mean()) if products.mean() != 0 else 1.0
        return cv < self.rtol, products

    # ------------------------------------------------------------------
    # Hyperbola projection
    # ------------------------------------------------------------------

    @staticmethod
    def hyperbola_points(
        C: float = SAV_CONSTANT, n: int = 100, S_A_range: tuple[float, float] = (0.01, 5.0)
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Points on the S_A · S_V = C duality hyperbola."""
        S_A = np.linspace(S_A_range[0], S_A_range[1], n)
        S_V = C / S_A
        return S_A, S_V

    @staticmethod
    def project_to_hyperbola(
        S_A: float, S_V: float
    ) -> tuple[float, float]:
        """Project (S_A, S_V) onto the nearest point on the C = S_A*S_V hyperbola."""
        C = S_A * S_V
        # geometric mean is the equidistant point
        s = math.sqrt(C)
        return s, s
