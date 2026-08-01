"""Physical and mathematical constants for S_A/S_V Duality module."""

from __future__ import annotations

import math

# Golden ratio
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0  # 1.6180339887...

# UTAC canonical parameters
SIGMA_PHI: float = 1.0 / 16.0  # Frame Principle coupling ≈ 0.0625
SIGMA_UTAC: float = 2.2         # UTAC ODE steepness (= β-to-Γ scaling)

# CREP triple-universality value
GAMMA_UNIVERSAL: float = 0.251  # AMOC = Neural = Theta

# S_A / S_V duality constant (normalised)
SAV_CONSTANT: float = 1.0

# Q4 state count
Q4_N_STATES: int = 16
Q4_BITS: int = 4

# Package registry entry
PACKAGE_REGISTRY_ID: int = 36
PACKAGE_INFO: dict[str, object] = {
    "name": "sa-sv-duality",
    "domain": "mathematical-physics",
    "scale": "foundational",
    "zenodo": "10.5281/zenodo.20842509",
}
