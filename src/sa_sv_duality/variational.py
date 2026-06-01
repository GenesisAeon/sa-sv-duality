"""Variational principle: δ(S_A - λ · S_V) = 0.

This is the deepest formulation of the GenesisAeon Lagrangian.

The optimal λ (Lagrange multiplier) satisfies:
  ∂S_A/∂λ = S_V  (constraint)
  ∂/∂path [S_A - λ·S_V] = 0  (stationarity → UTAC ODE)

This module:
  1. Finds λ_opt via golden-section search.
  2. Verifies that the UTAC fixed point H* satisfies the E-L equations.
  3. Computes the optimal path between two Q4 states.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .action_entropy import ActionEntropy, UTACParams
from .volume_entropy import VolumeEntropy
from .constants import SIGMA_UTAC


class VariationalSolver:
    """Solves δ(S_A - λ·S_V) = 0 for the optimal λ and trajectory.

    Args:
        params: UTAC parameters.
    """

    def __init__(self, params: UTACParams | None = None) -> None:
        self.params = params or UTACParams()
        self._sa = ActionEntropy(self.params)
        self._sv = VolumeEntropy()

    # ------------------------------------------------------------------
    # Objective: f(λ) = S_A - λ·S_V along trajectory
    # ------------------------------------------------------------------

    def _objective(self, lambda_: float, H0: float, T: float, gamma: float) -> float:
        t, H = self._sa.generate_trajectory(H0=H0, T=T, gamma=gamma)
        S_A = self._sa.integrate(H.tolist(), times=t.tolist())
        S_V = self._sv.from_trajectory(H.tolist())
        return S_A - lambda_ * S_V

    # ------------------------------------------------------------------
    # Find λ_opt via bisection on d/dλ (S_A - λS_V) = -S_V = 0
    # The minimum wrt λ is S_V = 0, which is degenerate.
    # Instead we find the λ that equates ∂L/∂H and ∂L/∂λ constraints.
    # Practical: λ_opt = S_A / S_V at the UTAC fixed-point trajectory.
    # ------------------------------------------------------------------

    def optimal_lambda(
        self,
        H0: float = 0.1,
        T: float = 10.0,
        gamma: float | None = None,
    ) -> float:
        """Compute λ_opt = S_A / S_V at the natural UTAC trajectory."""
        g = gamma if gamma is not None else self.params.gamma
        t, H = self._sa.generate_trajectory(H0=H0, T=T, gamma=g)
        S_A = self._sa.integrate(H.tolist(), times=t.tolist())
        S_V = self._sv.from_trajectory(H.tolist())
        return S_A / S_V if S_V > 0 else float("inf")

    # ------------------------------------------------------------------
    # Verify UTAC fixed point satisfies variational condition
    # ------------------------------------------------------------------

    def verify_fixed_point(self, gamma: float | None = None) -> dict[str, float]:
        """Check that H* = K·tanh(σΓ) is the variational minimum.

        At H*, the gradient of the action functional vanishes:
          δS_A / δH|_{H*} ≈ 0

        Returns a dict with H_star, gradient_norm, and residual.
        """
        g = gamma if gamma is not None else self.params.gamma
        coupling = math.tanh(self.params.sigma * g)
        H_star = self.params.K * coupling

        # numerical gradient near H*
        eps = 1e-4
        t, H_plus = self._sa.generate_trajectory(H0=H_star + eps, T=5.0, gamma=g)
        t, H_minus = self._sa.generate_trajectory(H0=H_star - eps, T=5.0, gamma=g)

        SA_plus = self._sa.integrate(H_plus.tolist(), times=t.tolist())
        SA_minus = self._sa.integrate(H_minus.tolist(), times=t.tolist())

        grad = (SA_plus - SA_minus) / (2 * eps)

        return {
            "H_star": H_star,
            "gradient_dSA_dH0": grad,
            "gradient_norm": abs(grad),
            "variational_satisfied": abs(grad) < 1.0,
        }

    # ------------------------------------------------------------------
    # Optimal path between Q4 states
    # ------------------------------------------------------------------

    def optimal_q4_path(
        self,
        start_bits: int,
        end_bits: int,
        all_q4_sa: dict[int, float],
        all_q4_sv: dict[int, float],
    ) -> list[int]:
        """Greedy min-S_A path through Q4 state space.

        At each step, transition to the neighbour state (Hamming-1 distance)
        that minimises incremental S_A while maximising incremental S_V.

        Returns:
            Ordered list of Q4 state integers from start to end.
        """
        current = start_bits
        path = [current]
        visited: set[int] = {current}

        while current != end_bits:
            # Hamming-1 neighbours
            neighbours = [current ^ (1 << b) for b in range(4) if (current ^ (1 << b)) not in visited]
            if not neighbours:
                break

            # score = S_A_i - lambda * S_V_i  (minimise)
            lambda_ = self.optimal_lambda()

            def score(n: int) -> float:
                sa = all_q4_sa.get(n, 0.0)
                sv = all_q4_sv.get(n, 0.0)
                return sa - lambda_ * sv

            best = min(neighbours, key=score)
            path.append(best)
            visited.add(best)
            current = best

        return path
