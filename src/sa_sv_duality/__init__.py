"""S_A / S_V Entropie-Dualität — GenesisAeon Package 36.

Foundational mathematical physics module implementing the duality between
Action Entropy (S_A) and Volume Entropy (S_V), which underlies:
  - Diffusive network routing (P30)
  - The Unified GenesisAeon Lagrangian
  - AFET field equations
  - Q4-state space entropy ordering

Reference: GenesisAeon Feldtheorie Preprint, DOI 10.5281/zenodo.17472834
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Johann Römer · MOR Research Collective"
__zenodo__ = "10.5281/zenodo.17472834"

from .action_entropy import ActionEntropy
from .volume_entropy import VolumeEntropy
from .duality_relation import DualityRelation
from .lagrangian_bridge import LagrangianBridge
from .variational import VariationalSolver
from .q4_sav_map import Q4SAVMap
from .network_routing import SAVRouter
from .system import SAVDuality

__all__ = [
    "ActionEntropy",
    "VolumeEntropy",
    "DualityRelation",
    "LagrangianBridge",
    "VariationalSolver",
    "Q4SAVMap",
    "SAVRouter",
    "SAVDuality",
]
