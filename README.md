# sa-sv-duality

**S_A / S_V Entropy Duality** -- GenesisAeon Package 36

[![CI](https://github.com/GenesisAeon/sa-sv-duality/actions/workflows/ci.yml/badge.svg)](https://github.com/GenesisAeon/sa-sv-duality/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17472834-blue)](https://doi.org/10.5281/zenodo.17472834)

Standalone foundational module implementing the duality between **Action Entropy** (S_A)
and **Volume Entropy** (S_V) -- the mathematical core underlying diffusive routing (P30),
the unified GenesisAeon Lagrangian, and the AFET field equations.

---

## The Duality

```
S_A  =  integral of sigma_s(H,Gamma) dt    (entropy produced along a path)
S_V  =  -integral p(H) ln p(H) dH          (Shannon entropy of state distribution)

S_A * S_V = const  (along the optimal UTAC trajectory)
```

**Lagrangian interpretation:**

```
L = T - V + Phi(H) + Gamma(C,R,E,P)
         ^^^^^^^^^^   ^^^^^^^^^^^^^
         S_A coupling  S_V coupling
```

The variational principle `delta(S_A - lambda * S_V) = 0` reproduces the UTAC ODE --
this is the deepest formulation of the GenesisAeon dynamics.

---

## Installation

```bash
pip install sa-sv-duality
# or
uv tool install sa-sv-duality
```

## Usage

### CLI

```bash
# Run a full S_A/S_V cycle (Gamma = 0.251, sigma = 2.2)
sav run

# Custom parameters
sav run --gamma 0.46 --sigma 2.2 --duration 20

# Show S_A, S_V for all 16 Q4 states
sav q4-map

# Find the optimal entropy path through Q4 state space
sav route 0 15 --lambda 1.0

# Run benchmarks
sav benchmark
sav benchmark --fast
```

### Python API

```python
from sa_sv_duality import SAVDuality, ActionEntropy, VolumeEntropy, Q4SAVMap

# Full Diamond interface
system = SAVDuality(gamma=0.251, sigma=2.2)
result = system.run_cycle(duration_years=10.0)
print(result["S_A"], result["S_V"], result["duality_constant"])

# CREP state
print(system.get_crep_state())  # {"C": ..., "R": ..., "E": ..., "P": ..., "Gamma": 0.251}

# Q4 entropy map
q4 = system.q4_entropy_map()   # {0: (S_A, S_V), ..., 15: (S_A, S_V)}

# Optimal path through Q4 space (= diffusive routing from P30)
path = system.optimal_path(start=0, end=15)

# Zenodo-compatible record
record = system.to_zenodo_record()

# Lower-level components
ae = ActionEntropy()
t, H = ae.generate_trajectory(H0=0.1, T=10.0)
S_A = ae.integrate(H.tolist(), times=t.tolist())

sv = VolumeEntropy()
S_V = sv.from_trajectory(H.tolist())
print(S_A * S_V)   # duality constant
```

## Module Structure

```
src/sa_sv_duality/
  __init__.py           -- public API
  constants.py          -- Phi, sigma_UTAC, Gamma_universal, Q4 constants
  action_entropy.py     -- S_A = integral sigma_s(H,Gamma) dt
  volume_entropy.py     -- S_V = -integral p*ln(p) + Q4 discrete entropies
  duality_relation.py   -- S_A * S_V conservation + hyperbola geometry
  lagrangian_bridge.py  -- Phi(H) = S_A coupling, Gamma = S_V coupling
  variational.py        -- delta(S_A - lambda*S_V) = 0 solver
  q4_sav_map.py         -- (S_A, S_V) for all 16 Q4 states
  network_routing.py    -- SAV-optimal routing via modified Dijkstra
  system.py             -- Diamond interface (run_cycle, CREP/UTAC state)
  cli.py                -- `sav` CLI
  benchmark.py          -- Package 36 benchmark targets
```

## Q4 State Entropy Map

Each of the 16 Q4 states (4-bit encoding of C/R/E/P activation) carries
a characteristic (S_A, S_V) pair:

| State | Bits | S_V    | Meaning                    |
|-------|------|--------|----------------------------|
| 0     | 0000 | 0.000  | No active channels          |
| 1     | 0001 | 0.693  | C only (coherence)          |
| 15    | 1111 | 1.609  | All channels active         |

S_V ordering: monotone non-decreasing with number of active bits.
S_A ordering: depends on effective Gamma of each state.

## Benchmark Targets (from Feldtheorie Preprint)

| Target                          | Expected | Tolerance |
|---------------------------------|----------|-----------|
| duality_constant_stability      | True     | --        |
| routing_improvement_vs_dijkstra | 10%      | +/-8%     |
| q4_entropy_ordering             | True     | --        |
| lagrangian_consistency          | True     | --        |
| variational_minimum             | True     | --        |

## Connection to Other Packages

| Package | Connection |
|---------|-----------|
| P30 diffusive-routing | route = argmin S_A + argmax S_V |
| P31 vrig-cosmological | v_RIG = information-geometric speed in S_A/S_V space |
| P32 beta-clustering   | beta_domain proportional to S_A/S_V ratio |
| P34 afet-tensions     | H0_eff(beta) driven by S_A / S_V hierarchy |
| P37 eml-utac-bridge   | EML operator encodes L = T-V + S_A + S_V |
| P38 phi-validator     | Phi^(1/3) scaling in S_A/S_V Q4 entropy ratios |

## Development

```bash
git clone https://github.com/GenesisAeon/sa-sv-duality.git
cd sa-sv-duality
uv sync --dev
uv run pytest
```

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20842509.svg)](https://doi.org/10.5281/zenodo.20842509)

---

**Reference:** Johann Roemer * MOR Research Collective * Mai 2026
DOI: [10.5281/zenodo.17472834](https://doi.org/10.5281/zenodo.17472834)
UTAC v1.0: [10.5281/zenodo.17472834](https://doi.org/10.5281/zenodo.17472834)
