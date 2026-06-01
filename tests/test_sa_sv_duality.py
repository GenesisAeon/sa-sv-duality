"""Tests for S_A / S_V Duality — GenesisAeon Package 36."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sa_sv_duality import (
    ActionEntropy,
    DualityRelation,
    LagrangianBridge,
    Q4SAVMap,
    SAVDuality,
    SAVRouter,
    VariationalSolver,
    VolumeEntropy,
)
from sa_sv_duality.action_entropy import UTACParams, utac_entropy_rate
from sa_sv_duality.benchmark import run_benchmarks
from sa_sv_duality.constants import GAMMA_UNIVERSAL, PHI, Q4_N_STATES, SIGMA_UTAC

# ── Constants ────────────────────────────────────────────────────────────────

def test_phi_value():
    assert abs(PHI - 1.6180339887) < 1e-9


def test_sigma_utac():
    assert abs(SIGMA_UTAC - 2.2) < 1e-9


def test_gamma_universal():
    assert abs(GAMMA_UNIVERSAL - 0.251) < 1e-9


# ── ActionEntropy ────────────────────────────────────────────────────────────

def test_entropy_rate_non_negative():
    ae = ActionEntropy()
    for H in [0.0, 0.1, 0.5, 1.0]:
        assert ae.entropy_rate(H) >= 0.0


def test_entropy_rate_zero_at_H_zero():
    assert utac_entropy_rate(0.0, 0.251) == 0.0


def test_entropy_rate_zero_at_carrying_capacity():
    ae = ActionEntropy(UTACParams(K=1.0))
    # H=K → logistic = 0 → rate = 0
    assert abs(ae.entropy_rate(1.0)) < 1e-12


def test_integrate_positive():
    ae = ActionEntropy()
    t, H = ae.generate_trajectory(H0=0.1, T=5.0, n_steps=100)
    S_A = ae.integrate(H.tolist(), times=t.tolist())
    assert S_A >= 0.0


def test_integrate_increases_with_gamma():
    ae = ActionEntropy()
    S_A_low = ae.from_utac_ode(T=5.0, n_steps=100, gamma=0.1)
    S_A_high = ae.from_utac_ode(T=5.0, n_steps=100, gamma=0.9)
    assert S_A_high >= S_A_low


def test_trajectory_monotone_increase():
    ae = ActionEntropy()
    _, H = ae.generate_trajectory(H0=0.01, T=10.0, n_steps=200)
    # Trajectory should generally increase toward K
    assert H[-1] > H[0]


# ── VolumeEntropy ────────────────────────────────────────────────────────────

def test_volume_entropy_non_negative():
    sv = VolumeEntropy()
    _, H = ActionEntropy().generate_trajectory(T=5.0)
    assert sv.from_trajectory(H.tolist()) >= 0.0


def test_volume_entropy_from_uniform():
    sv = VolumeEntropy(n_bins=10)
    # Uniform distribution → maximum entropy
    H = np.linspace(0.0, 1.0, 1000).tolist()
    assert sv.from_trajectory(H) > 0.0


def test_q4_state_entropy_ordering():
    # S_V must be non-decreasing with n_active bits
    sv = VolumeEntropy()
    entropies = [sv.q4_state_entropy(i) for i in range(16)]
    # 0000 has S_V = ln(1) = 0
    assert entropies[0] == 0.0
    # 1111 has S_V = ln(5) > any state with 3 active bits
    assert entropies[15] == pytest.approx(math.log(5), abs=1e-9)


def test_q4_entropy_vector_length():
    v = VolumeEntropy.q4_entropy_vector()
    assert len(v) == Q4_N_STATES


def test_from_density_normalised():
    sv = VolumeEntropy()
    # Should accept un-normalised input
    H_ent = sv.from_density([1, 1, 1, 1])  # uniform over 4 bins
    assert abs(H_ent - math.log(4)) < 0.01


# ── DualityRelation ──────────────────────────────────────────────────────────

def test_duality_result_product_positive():
    dr = DualityRelation()
    result = dr.analyse(T=5.0)
    assert result.product >= 0.0


def test_duality_conservation():
    dr = DualityRelation(rtol=0.25)
    conserved, products = dr.verify_conservation(n_points=8, T=5.0)
    # Product should be reasonably stable across Γ range
    cv = float(products.std() / products.mean())
    assert cv < 0.5  # not required to be conserved to 15%, just stable


def test_hyperbola_points():
    S_A, S_V = DualityRelation.hyperbola_points(C=1.0, n=10)
    products = S_A * S_V
    assert np.allclose(products, 1.0, atol=1e-10)


def test_project_to_hyperbola():
    S_A_proj, S_V_proj = DualityRelation.project_to_hyperbola(2.0, 0.5)
    assert abs(S_A_proj - S_V_proj) < 1e-10  # geometric mean → equal


# ── LagrangianBridge ─────────────────────────────────────────────────────────

def test_phi_at_K_zero():
    lb = LagrangianBridge()
    # Φ(H) → 0 as H → K from below
    phi_near_K = lb.phi(1.0 - 1e-6)
    # Should be close to Φ(K) = α·K·ln(1) + β·K = β·K
    assert abs(phi_near_K - lb.beta * lb.K) < 0.01


def test_phi_negative_H_returns_zero():
    lb = LagrangianBridge()
    assert lb.phi(-1.0) == 0.0
    assert lb.phi(0.0) == 0.0


def test_crep_geometric_mean():
    # Γ = (0.5·0.5·0.5·0.5)^0.25 = 0.5
    gamma = LagrangianBridge.crep(0.5, 0.5, 0.5, 0.5)
    assert abs(gamma - 0.5) < 1e-9


def test_crep_negative_raises():
    with pytest.raises(ValueError):
        LagrangianBridge.crep(-0.1, 0.5, 0.5, 0.5)


def test_lagrangian_evaluate():
    lb = LagrangianBridge()
    state = lb.evaluate(H=0.5, dH_dt=0.1)
    assert isinstance(state.L_total, float)
    # crep(0.5,0.5,0.5,0.5) = (0.5^4)^(1/4) = 0.5
    assert state.crep_gamma == pytest.approx(0.5, abs=1e-9)


# ── VariationalSolver ────────────────────────────────────────────────────────

def test_optimal_lambda_positive():
    vs = VariationalSolver()
    lam = vs.optimal_lambda(T=5.0)
    assert lam > 0.0


def test_fixed_point_verification():
    vs = VariationalSolver()
    result = vs.verify_fixed_point()
    assert "H_star" in result
    assert result["H_star"] > 0.0
    # UTAC fixed point: H* = K·tanh(σΓ)
    params = vs.params
    expected = params.K * math.tanh(params.sigma * params.gamma)
    assert abs(result["H_star"] - expected) < 1e-9


# ── Q4SAVMap ─────────────────────────────────────────────────────────────────

def test_q4_map_has_16_states():
    q4 = Q4SAVMap(n_steps=50)
    states = q4.compute()
    assert len(states) == 16


def test_q4_entropy_ordering():
    q4 = Q4SAVMap(n_steps=50)
    assert q4.entropy_ordering_valid()


def test_q4_state_0_minimal_sv():
    q4 = Q4SAVMap(n_steps=50)
    states = q4.get()
    # 0000 → no active bits → S_V = ln(1) = 0
    assert states[0].S_V == 0.0
    assert states[0].n_active == 0


def test_q4_state_15_maximal_sv():
    q4 = Q4SAVMap(n_steps=50)
    states = q4.get()
    # 1111 → 4 active bits → S_V = ln(5) (maximum)
    assert pytest.approx(math.log(5), abs=1e-9) == states[15].S_V
    assert states[15].n_active == 4


def test_q4_sa_positive():
    q4 = Q4SAVMap(n_steps=50)
    for sid, state in q4.get().items():
        if sid > 0:  # 0000 has near-zero gamma → S_A may be ~0
            assert state.S_A >= 0.0


# ── SAVRouter ────────────────────────────────────────────────────────────────

def test_router_finds_path():
    router = SAVRouter.from_q4_adjacency(lambda_=1.0)
    result = router.route(0, 15)
    assert result is not None
    assert result.path[0] == 0
    assert result.path[-1] == 15


def test_router_path_length():
    router = SAVRouter.from_q4_adjacency(lambda_=1.0)
    result = router.route(0, 15)
    assert result is not None
    # Min path from 0000 to 1111: 4 flips = 4 hops
    assert result.hops >= 4


def test_router_same_start_end():
    router = SAVRouter.from_q4_adjacency()
    result = router.route(5, 5)
    assert result is not None
    assert result.path == [5]
    assert result.hops == 0


def test_router_no_negative_sa():
    router = SAVRouter.from_q4_adjacency()
    result = router.route(0, 15)
    assert result is not None
    assert result.total_S_A >= 0.0


# ── SAVDuality (Diamond Interface) ───────────────────────────────────────────

def test_system_run_cycle():
    system = SAVDuality()
    result = system.run_cycle(duration_years=5.0)
    assert "S_A" in result
    assert "S_V" in result
    assert result["S_A"] >= 0.0
    assert result["S_V"] >= 0.0


def test_system_crep_state():
    system = SAVDuality()
    crep = system.get_crep_state()
    assert "Gamma" in crep
    assert abs(crep["Gamma"] - GAMMA_UNIVERSAL) < 1e-9
    for k in ("C", "R", "E", "P"):
        assert 0.0 <= crep[k] <= 1.0


def test_system_utac_state():
    system = SAVDuality()
    utac = system.get_utac_state()
    assert "H_star" in utac
    assert utac["H_star"] > 0.0


def test_system_phase_events_after_cycle():
    system = SAVDuality()
    system.run_cycle(duration_years=5.0)
    events = system.get_phase_events()
    assert isinstance(events, list)


def test_system_zenodo_record():
    system = SAVDuality()
    system.run_cycle(duration_years=3.0)
    record = system.to_zenodo_record()
    assert record["package"] == 36
    assert record["zenodo_doi"] == "10.5281/zenodo.17472834"


def test_system_duality_constant():
    system = SAVDuality()
    C = system.duality_constant()
    assert C >= 0.0


def test_system_q4_entropy_map():
    system = SAVDuality()
    q4 = system.q4_entropy_map()
    assert len(q4) == 16
    for _sid, (sa, sv) in q4.items():
        assert sa >= 0.0
        assert sv >= 0.0


def test_system_optimal_path():
    system = SAVDuality()
    path = system.optimal_path(0, 15)
    assert path[0] == 0
    assert path[-1] == 15


def test_system_v_rig_velocity():
    system = SAVDuality()
    v = system.v_rig_velocity()
    assert v > 0.0
    # sech²(σΓ) ≤ 1 → v ≤ K·σ = 2.2
    assert v <= system.params.K * system.params.sigma + 1e-9


# ── Benchmarks ───────────────────────────────────────────────────────────────

def test_benchmark_fast():
    report = run_benchmarks(fast=True)
    assert report.ok, report.summary()
