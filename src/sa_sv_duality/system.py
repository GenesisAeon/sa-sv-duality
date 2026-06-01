"""SAVDuality — Diamond Interface for GenesisAeon Package 36.

Implements the standard GenesisAeon runtime contract:
  run_cycle(), get_crep_state(), get_utac_state(),
  get_phase_events(), to_zenodo_record()

Plus Package-36-specific methods:
  duality_constant(), q4_entropy_map(), optimal_path()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .action_entropy import ActionEntropy, UTACParams
from .volume_entropy import VolumeEntropy
from .duality_relation import DualityRelation, DualityResult
from .lagrangian_bridge import LagrangianBridge, LagrangianState
from .variational import VariationalSolver
from .q4_sav_map import Q4SAVMap, Q4EntropyState
from .network_routing import SAVRouter, RoutingResult
from .constants import (
    PHI, SIGMA_PHI, SIGMA_UTAC, GAMMA_UNIVERSAL,
    PACKAGE_REGISTRY_ID, PACKAGE_INFO, Q4_N_STATES,
)


@dataclass
class CycleResult:
    """Result of a full SAVDuality run_cycle()."""

    S_A: float
    S_V: float
    duality_constant: float
    lambda_opt: float
    gamma_effective: float
    conserved: bool
    q4_map: dict[int, Q4EntropyState]
    phase_events: list[dict[str, Any]]
    duration_s: float


class SAVDuality:
    """Diamond Interface — S_A / S_V Entropy Duality (Package 36).

    This is the entry point for the GenesisAeon runtime to interact with
    the S_A/S_V module.  It orchestrates all sub-components and exposes
    the standardised runtime contract.

    Args:
        r: UTAC growth rate.
        K: UTAC carrying capacity.
        sigma: UTAC coupling parameter (default σ = 2.2).
        gamma: CREP Γ value (default 0.251 — triple universality).
        lambda_: Lagrange multiplier for variational routing (default 1.0).
    """

    def __init__(
        self,
        r: float = 1.0,
        K: float = 1.0,
        sigma: float = SIGMA_UTAC,
        gamma: float = GAMMA_UNIVERSAL,
        lambda_: float = 1.0,
    ) -> None:
        self.params = UTACParams(r=r, K=K, sigma=sigma, gamma=gamma)
        self.lambda_ = lambda_

        self._sa = ActionEntropy(self.params)
        self._sv = VolumeEntropy()
        self._dr = DualityRelation(self.params)
        self._lb = LagrangianBridge(K=K)
        self._vs = VariationalSolver(self.params)
        self._q4 = Q4SAVMap(self.params)
        self._last_cycle: CycleResult | None = None

    # ------------------------------------------------------------------
    # Diamond contract — required methods
    # ------------------------------------------------------------------

    def run_cycle(self, duration_years: float = 10.0) -> dict[str, Any]:
        """Run one full S_A/S_V analysis cycle.

        Args:
            duration_years: integration time (mapped to T in ODE units).

        Returns:
            Dict with S_A, S_V, duality_constant, phase_events, metadata.
        """
        t0 = time.perf_counter()
        T = duration_years  # 1 year ↔ 1 ODE time unit

        # 1. Generate UTAC trajectory
        t_arr, H_arr = self._sa.generate_trajectory(T=T)

        # 2. Compute entropies
        S_A = self._sa.integrate(H_arr.tolist(), times=t_arr.tolist())
        S_V = self._sv.from_trajectory(H_arr.tolist())

        # 3. Duality product
        C_dual = S_A * S_V
        lambda_opt = S_A / S_V if S_V > 0 else 0.0

        # 4. Conservation check
        conserved, _ = self._dr.verify_conservation(n_points=10, T=T)

        # 5. Q4 map (cached)
        q4_map = self._q4.get()

        # 6. Phase events = points where d/dt(S_A rate) changes sign
        from .action_entropy import utac_entropy_rate
        rates = np.array(
            [utac_entropy_rate(float(h), self.params.gamma, r=self.params.r, K=self.params.K)
             for h in H_arr]
        )
        d_rates = np.gradient(rates, t_arr)
        sign_changes = np.where(np.diff(np.sign(d_rates)))[0]
        phase_events = [
            {"time": float(t_arr[i]), "H": float(H_arr[i]), "type": "entropy_rate_inflection"}
            for i in sign_changes
        ]

        elapsed = time.perf_counter() - t0

        self._last_cycle = CycleResult(
            S_A=S_A,
            S_V=S_V,
            duality_constant=C_dual,
            lambda_opt=lambda_opt,
            gamma_effective=self.params.gamma,
            conserved=conserved,
            q4_map=q4_map,
            phase_events=phase_events,
            duration_s=elapsed,
        )

        return {
            "S_A": S_A,
            "S_V": S_V,
            "duality_constant": C_dual,
            "lambda_opt": lambda_opt,
            "conserved": conserved,
            "n_phase_events": len(phase_events),
            "q4_states_computed": len(q4_map),
            "duration_s": elapsed,
        }

    def get_crep_state(self) -> dict[str, float]:
        """Return CREP tensor state {C, R, E, P, Gamma}."""
        # Map UTAC parameters to CREP components
        coupling = math.tanh(self.params.sigma * self.params.gamma)
        return {
            "C": coupling,
            "R": self.params.r / (self.params.r + 1.0),  # normalised
            "E": 1.0 - coupling,                          # entropy = 1 - order
            "P": self.params.K / (self.params.K + 1.0),  # normalised
            "Gamma": self.params.gamma,
        }

    def get_utac_state(self) -> dict[str, float]:
        """Return current UTAC parameter state."""
        H_star = self.params.K * math.tanh(self.params.sigma * self.params.gamma)
        return {
            "r": self.params.r,
            "K": self.params.K,
            "sigma": self.params.sigma,
            "gamma": self.params.gamma,
            "H_star": H_star,
        }

    def get_phase_events(self) -> list[dict[str, Any]]:
        """Return phase events from the last cycle (entropy rate inflections)."""
        if self._last_cycle is None:
            return []
        return self._last_cycle.phase_events

    def to_zenodo_record(self) -> dict[str, Any]:
        """Serialise current state to a Zenodo-compatible metadata dict."""
        state = self._last_cycle
        return {
            "package": PACKAGE_REGISTRY_ID,
            "name": PACKAGE_INFO["name"],
            "domain": PACKAGE_INFO["domain"],
            "zenodo_doi": PACKAGE_INFO["zenodo"],
            "utac_params": self.get_utac_state(),
            "crep_state": self.get_crep_state(),
            "last_cycle": {
                "S_A": state.S_A if state else None,
                "S_V": state.S_V if state else None,
                "duality_constant": state.duality_constant if state else None,
                "conserved": state.conserved if state else None,
            },
            "reference": "Johann Römer · MOR Research Collective · Mai 2026",
        }

    # ------------------------------------------------------------------
    # Package-36-specific methods
    # ------------------------------------------------------------------

    def duality_constant(self) -> float:
        """Returns S_A · S_V for the current trajectory."""
        result = self._dr.analyse()
        return result.product

    def q4_entropy_map(self) -> dict[int, tuple[float, float]]:
        """Returns {Q4_id: (S_A, S_V)} for all 16 Q4 states."""
        states = self._q4.get()
        return {sid: (s.S_A, s.S_V) for sid, s in states.items()}

    def optimal_path(self, start: int, end: int) -> list[int]:
        """Returns min-(S_A - λ·S_V) path through Q4 space.

        This is the S_A-minimising path — identical to the diffusive
        routing algorithm in P30 when projected onto the Q4 state graph.
        """
        router = SAVRouter.from_q4_adjacency(lambda_=self.lambda_)
        result = router.route(start, end)
        return result.path if result else []

    def v_rig_velocity(self) -> float:
        """Information-geometric velocity of current UTAC state.

        v = |dH*/dΓ| · Γ̇ — Fisher-Rao speed in the (K, Γ) parameter space.
        Normalised to [0, 1] relative to the v_RIG cosmological scale.
        """
        # Analytic: H* = K·tanh(σΓ) → dH*/dΓ = K·σ·sech²(σΓ)
        sigma_g = self.params.sigma * self.params.gamma
        sech2 = 1.0 / (math.cosh(sigma_g) ** 2)
        dH_dGamma = self.params.K * self.params.sigma * sech2
        return abs(dH_dGamma)
