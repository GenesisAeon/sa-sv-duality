# sa-sv-duality

**S_A / S_V Entropy Duality** -- GenesisAeon Package 36

Standalone foundational module for the duality between Action Entropy (S_A)
and Volume Entropy (S_V).

## Core Concept

```
S_A = action entropy   = integral sigma_s(H,Gamma) dt
S_V = volume entropy   = -integral p(H) ln p(H) dH
S_A * S_V = const      (duality product, conserved along optimal path)
```

This duality IS the GenesisAeon Lagrangian decomposition:

```
L = T - V + Phi(H)       + Gamma(C,R,E,P)
           ^^^^^           ^^^^^^^^^^^^
           S_A coupling    S_V coupling
```

## Quickstart

```bash
pip install sa-sv-duality

sav run
sav q4-map
sav route 0 15
sav benchmark
```

## Commands

| Command | Description |
|---------|-------------|
| `sav run` | Full S_A/S_V analysis cycle |
| `sav q4-map` | Entropy map for all 16 Q4 states |
| `sav route <src> <dst>` | Optimal path through Q4 space |
| `sav benchmark` | Run all benchmark tests |
| `sav version` | Show version |

## Reference

- UTAC v1.0: DOI 10.5281/zenodo.17472834
- Johann Roemer * MOR Research Collective * Mai 2026
