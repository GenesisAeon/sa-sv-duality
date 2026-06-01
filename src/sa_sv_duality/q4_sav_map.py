"""Q4 State Space S_A / S_V Mapping.

Maps all 16 Q4 states (4-bit encoding 0x0…0xF) to their corresponding
(S_A, S_V) entropy pair.

Q4 state bit interpretation:
  bit 0 (LSB): C — Coherence active
  bit 1:       R — Resonance active
  bit 2:       E — Entropy coupling active
  bit 3 (MSB): P — Productivity active

Active bits raise S_V (more channels = richer state).
Active bits that are *coherence-breaking* (E, P) raise S_A (more costly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .action_entropy import ActionEntropy, UTACParams
from .volume_entropy import VolumeEntropy
from .constants import SIGMA_UTAC, Q4_N_STATES

# Per-bit gamma contributions (from CREP Atlas, P17-P38)
_BIT_GAMMA: dict[int, float] = {
    0: 0.45,   # C: Coherence  → high γ = organised
    1: 0.35,   # R: Resonance  → medium γ
    2: 0.15,   # E: Entropy    → low γ = diffuse
    3: 0.25,   # P: Productivity → medium γ
}


@dataclass
class Q4EntropyState:
    """S_A and S_V for a single Q4 state."""

    state_id: int
    label: str       # e.g. "0101" for bits 2 and 0 set
    S_A: float
    S_V: float
    product: float
    gamma: float     # effective CREP Γ for this state
    n_active: int    # number of active bits


class Q4SAVMap:
    """Compute and store the full S_A/S_V entropy map for all 16 Q4 states.

    The effective Γ for a state is the mean of the active-bit gammas.
    S_A is computed via ActionEntropy.from_utac_ode with the state's Γ.
    S_V uses VolumeEntropy.q4_state_entropy.
    """

    def __init__(
        self,
        utac_params: UTACParams | None = None,
        T: float = 10.0,
        n_steps: int = 200,
    ) -> None:
        self._params = utac_params or UTACParams()
        self._T = T
        self._n_steps = n_steps
        self._sa = ActionEntropy(self._params)
        self._sv = VolumeEntropy()
        self._cache: dict[int, Q4EntropyState] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label(state_id: int) -> str:
        return format(state_id, "04b")

    @staticmethod
    def _effective_gamma(state_id: int) -> float:
        bits = [i for i in range(4) if state_id & (1 << i)]
        if not bits:
            return 0.01  # near-zero activity
        return sum(_BIT_GAMMA[b] for b in bits) / len(bits)

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[int, Q4EntropyState]:
        """Compute (S_A, S_V) for all 16 Q4 states."""
        result: dict[int, Q4EntropyState] = {}
        for sid in range(Q4_N_STATES):
            gamma = self._effective_gamma(sid)
            S_A = self._sa.from_utac_ode(T=self._T, n_steps=self._n_steps, gamma=gamma)
            S_V = VolumeEntropy.q4_state_entropy(sid)
            n_active = bin(sid).count("1")
            result[sid] = Q4EntropyState(
                state_id=sid,
                label=self._label(sid),
                S_A=S_A,
                S_V=S_V,
                product=S_A * S_V,
                gamma=gamma,
                n_active=n_active,
            )
        self._cache = result
        return result

    def get(self) -> dict[int, Q4EntropyState]:
        """Return cached map, computing if necessary."""
        if self._cache is None:
            self.compute()
        assert self._cache is not None
        return self._cache

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def entropy_ordering_valid(self) -> bool:
        """Check that S_V is non-decreasing with number of active bits."""
        states = self.get()
        sv_by_active: dict[int, list[float]] = {}
        for s in states.values():
            sv_by_active.setdefault(s.n_active, []).append(s.S_V)
        groups = sorted(sv_by_active.items())
        for i in range(len(groups) - 1):
            if max(groups[i][1]) > min(groups[i + 1][1]) + 1e-9:
                return False
        return True

    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns (S_A_array, S_V_array) for states 0…15."""
        states = self.get()
        S_A = np.array([states[i].S_A for i in range(Q4_N_STATES)])
        S_V = np.array([states[i].S_V for i in range(Q4_N_STATES)])
        return S_A, S_V

    def summary_table(self) -> list[dict[str, float | int | str]]:
        """Return a list of dicts suitable for display / export."""
        return [
            {
                "id": s.state_id,
                "label": s.label,
                "S_A": round(s.S_A, 4),
                "S_V": round(s.S_V, 4),
                "S_A*S_V": round(s.product, 4),
                "gamma": round(s.gamma, 4),
                "n_active": s.n_active,
            }
            for s in sorted(self.get().values(), key=lambda x: x.state_id)
        ]
