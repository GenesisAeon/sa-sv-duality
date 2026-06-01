"""Connection of S_A / S_V to the Unified GenesisAeon Lagrangian.

L = T - V + Φ(H) + Γ(C,R,E,P)

where:
  T - V  = reversible dynamics (kinetic minus potential)
  Φ(H)   = S_A coupling: irreversible entropy production
  Γ(C,R,E,P) = S_V coupling: state richness / information content

The S_A / S_V duality IS the Lagrangian decomposition:
  S_A ↔ Φ(H)    (cost of transformation)
  S_V ↔ Γ       (richness of state)
  L   = S_V - λ · S_A  (with λ = Lagrange multiplier from variational)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import PHI, SIGMA_PHI, SIGMA_UTAC


@dataclass
class LagrangianState:
    """Full Lagrangian decomposition at a single state (H, Γ)."""

    H: float
    gamma: float
    T_minus_V: float   # reversible part
    phi_H: float       # AFET entropy potential = S_A coupling
    crep_gamma: float  # CREP value = S_V coupling
    L_total: float     # full Lagrangian value
    lambda_opt: float  # λ from S_A / S_V ratio


class LagrangianBridge:
    """Maps UTAC/CREP states to Lagrangian components.

    Implements:
      Φ(H) = α · H · ln(H/K) + β · H   (AFET potential)
      Γ(C,R,E,P) = (C · R · E · P)^(1/4)
      T - V = ½ · (dH/dt)²  (kinetic minus harmonic potential)
      L = (T-V) + Φ(H) + Γ
    """

    def __init__(
        self,
        K: float = 1.0,
        alpha: float = SIGMA_PHI,
        beta: float = 0.1,
        sigma: float = SIGMA_UTAC,
    ) -> None:
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    # ------------------------------------------------------------------
    # AFET potential Φ(H)
    # ------------------------------------------------------------------

    def phi(self, H: float) -> float:
        """AFET entropy potential.

        Φ(H) = α · H · ln(H/K) + β · H

        Represents irreversible entropy production cost (S_A coupling).
        Φ → 0 as H → K (near carrying capacity = minimal action cost).
        """
        if H <= 0:
            return 0.0
        return self.alpha * H * math.log(H / self.K) + self.beta * H

    # ------------------------------------------------------------------
    # CREP tensor Γ(C,R,E,P)
    # ------------------------------------------------------------------

    @staticmethod
    def crep(C: float, R: float, E: float, P: float) -> float:
        """CREP coupling tensor Γ = (C·R·E·P)^(1/4).

        Represents phase-space volume richness (S_V coupling).
        All components ∈ [0, 1].
        """
        if any(x < 0 for x in (C, R, E, P)):
            raise ValueError("CREP components must be non-negative")
        product = C * R * E * P
        return product ** 0.25

    # ------------------------------------------------------------------
    # Kinetic minus potential
    # ------------------------------------------------------------------

    @staticmethod
    def t_minus_v(dH_dt: float, H: float, K: float = 1.0) -> float:
        """T - V = ½ · (dH/dt)² - ½ · ω² · H²  (harmonic approximation).

        ω² = r²/K  characterises restoring force near fixed point.
        """
        omega_sq = 1.0 / max(K, 1e-9)
        return 0.5 * dH_dt ** 2 - 0.5 * omega_sq * H ** 2

    # ------------------------------------------------------------------
    # Full Lagrangian at a point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        H: float,
        dH_dt: float,
        C: float = 0.5,
        R: float = 0.5,
        E: float = 0.5,
        P: float = 0.5,
        lambda_: float = 1.0,
    ) -> LagrangianState:
        """Evaluate full Lagrangian at state (H, dH/dt, C, R, E, P)."""
        TV = self.t_minus_v(dH_dt, H, self.K)
        phi_val = self.phi(H)
        gamma = self.crep(C, R, E, P)

        # S_A proxy: |Φ(H)|, S_V proxy: Γ
        S_A_proxy = abs(phi_val)
        lambda_opt = S_A_proxy / gamma if gamma > 0 else 0.0

        L = TV + phi_val + gamma

        return LagrangianState(
            H=H,
            gamma=gamma,
            T_minus_V=TV,
            phi_H=phi_val,
            crep_gamma=gamma,
            L_total=L,
            lambda_opt=lambda_opt,
        )

    # ------------------------------------------------------------------
    # Euler-Lagrange check
    # ------------------------------------------------------------------

    def euler_lagrange_residual(
        self,
        trajectory: list[float],
        times: list[float],
        gamma: float = 0.251,
    ) -> NDArray[np.float64]:
        """Compute Euler-Lagrange residual d/dt(∂L/∂Ḣ) - ∂L/∂H ≈ 0.

        For the UTAC ODE the residual should be close to zero at the
        fixed point H* = K · tanh(σΓ), confirming that the UTAC dynamics
        extremise the action integral.
        """
        H = np.asarray(trajectory, dtype=float)
        t = np.asarray(times, dtype=float)
        dH = np.gradient(H, t)
        d2H = np.gradient(dH, t)

        # ∂L/∂Ḣ = dH/dt  →  d/dt(∂L/∂Ḣ) = d²H/dt²
        # ∂L/∂H = α·(ln(H/K) + 1) + β - ω²·H
        omega_sq = 1.0 / max(self.K, 1e-9)
        dL_dH = np.where(
            H > 0,
            self.alpha * (np.log(np.clip(H / self.K, 1e-12, None)) + 1.0) + self.beta - omega_sq * H,
            0.0,
        )

        return d2H - dL_dH
