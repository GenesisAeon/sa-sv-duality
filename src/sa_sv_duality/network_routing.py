"""Network routing as S_A minimisation — connection to diffusive-routing (P30).

Route = argmin S_A + argmax S_V
      = argmin (S_A - λ · S_V)

This implements a simple graph-based router where each edge carries
(S_A_cost, S_V_gain) and the optimal path minimises S_A - λ·S_V.

Connection to P30 (diffusive-routing): the diffusive routing algorithm
in P30 selects paths that minimise entropy production while maximising
state richness.  This module provides the foundational S_A/S_V scoring
that P30 builds upon.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(order=True)
class _PQItem:
    priority: float
    node: Any = field(compare=False)
    path: list[Any] = field(compare=False)


@dataclass
class RoutingResult:
    """Result of an S_A/S_V-optimal routing query."""

    path: list[Any]
    total_S_A: float
    total_S_V: float
    objective: float   # S_A - lambda * S_V
    hops: int


class SAVRouter:
    """Graph router minimising S_A - λ·S_V.

    Nodes represent Q4 states or arbitrary system states.
    Edges carry (S_A_cost, S_V_gain) tuples.

    Args:
        lambda_: Lagrange multiplier balancing cost vs. richness.
                 λ → 0: pure S_A minimisation (Dijkstra-like).
                 λ → ∞: pure S_V maximisation.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        self.lambda_ = lambda_
        self._graph: dict[Any, list[tuple[Any, float, float]]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_edge(
        self,
        src: Any,
        dst: Any,
        S_A: float,
        S_V: float,
        bidirectional: bool = False,
    ) -> None:
        """Add a directed edge with entropy costs."""
        if S_A < 0:
            raise ValueError("S_A cost must be non-negative")
        self._graph.setdefault(src, []).append((dst, S_A, S_V))
        if bidirectional:
            self._graph.setdefault(dst, []).append((src, S_A, S_V))

    @classmethod
    def from_q4_adjacency(cls, lambda_: float = 1.0) -> "SAVRouter":
        """Build a router on the 16 Q4 states connected by Hamming-1 edges.

        Edge weights come from the S_A / S_V Q4 map.
        """
        from .q4_sav_map import Q4SAVMap

        router = cls(lambda_=lambda_)
        q4map = Q4SAVMap()
        states = q4map.get()

        for sid in range(16):
            for bit in range(4):
                neighbour = sid ^ (1 << bit)
                s_edge = states[neighbour]
                router.add_edge(sid, neighbour, S_A=s_edge.S_A, S_V=s_edge.S_V)
        return router

    # ------------------------------------------------------------------
    # Routing algorithm (Dijkstra on objective = S_A - λ·S_V)
    # ------------------------------------------------------------------

    def route(self, source: Any, target: Any) -> RoutingResult | None:
        """Find the path minimising S_A - λ·S_V via modified Dijkstra.

        Returns None if no path exists.
        """
        dist: dict[Any, float] = {source: 0.0}
        sa_total: dict[Any, float] = {source: 0.0}
        sv_total: dict[Any, float] = {source: 0.0}

        pq: list[_PQItem] = [_PQItem(0.0, source, [source])]

        while pq:
            item = heapq.heappop(pq)
            curr_obj, node, path = item.priority, item.node, item.path

            if node == target:
                return RoutingResult(
                    path=path,
                    total_S_A=sa_total[node],
                    total_S_V=sv_total[node],
                    objective=curr_obj,
                    hops=len(path) - 1,
                )

            if curr_obj > dist.get(node, math.inf):
                continue

            for neighbour, edge_SA, edge_SV in self._graph.get(node, []):
                new_obj = curr_obj + edge_SA - self.lambda_ * edge_SV
                if new_obj < dist.get(neighbour, math.inf):
                    dist[neighbour] = new_obj
                    sa_total[neighbour] = sa_total[node] + edge_SA
                    sv_total[neighbour] = sv_total[node] + edge_SV
                    heapq.heappush(pq, _PQItem(new_obj, neighbour, path + [neighbour]))

        return None

    def improvement_vs_dijkstra(self, source: Any, target: Any) -> float:
        """Compare S_A cost: SAV-router vs. pure S_A Dijkstra.

        Returns fractional improvement:  (S_A_dijkstra - S_A_sav) / S_A_dijkstra
        """
        # SAV route
        sav_result = self.route(source, target)

        # Pure Dijkstra (λ = 0)
        dijkstra = SAVRouter(lambda_=0.0)
        dijkstra._graph = self._graph  # share graph
        dijk_result = dijkstra.route(source, target)

        if sav_result is None or dijk_result is None:
            return 0.0

        if dijk_result.total_S_A == 0:
            return 0.0

        return (dijk_result.total_S_A - sav_result.total_S_A) / dijk_result.total_S_A
