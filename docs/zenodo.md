# Zenodo Record -- sa-sv-duality (GenesisAeon Package 36)

## Metadata

```yaml
title: "S_A / S_V Entropy Duality -- GenesisAeon Package 36"
doi: "10.5281/zenodo.17472834"
version: "0.1.0"
upload_type: software
creators:
  - name: "Roemer, Johann"
    affiliation: "MOR Research Collective"
description: >
  Standalone Python implementation of the S_A/S_V entropy duality --
  the foundational principle underlying diffusive routing (P30), the
  unified GenesisAeon Lagrangian (L = T-V + Phi(H) + Gamma), and the
  AFET field equations. Implements S_A (action entropy = integral of
  entropy production along UTAC trajectories) and S_V (volume entropy =
  Shannon entropy of phase-space distribution), together with the
  duality relation S_A * S_V = const, variational solver
  delta(S_A - lambda*S_V) = 0, Q4 state entropy map, and SAV-optimal
  network routing via modified Dijkstra.
keywords:
  - entropy
  - duality
  - UTAC
  - CREP
  - GenesisAeon
  - mathematical-physics
  - complex-systems
  - action-entropy
  - volume-entropy
  - Lagrangian
license: MIT
related_identifiers:
  - identifier: "10.5281/zenodo.17472834"
    relation: isDerivedFrom
    scheme: doi
  - identifier: "https://github.com/GenesisAeon/sa-sv-duality"
    relation: isSupplementTo
    scheme: url
```

## Package Registry Entry (CREP Spectrum)

```python
PACKAGE_REGISTRY[36] = {
    "name": "sa-sv-duality",
    "domain": "mathematical-physics",
    "scale": "foundational",
    "zenodo": "10.5281/zenodo.17472834",
}
```

## CREP Spectrum Position

```
P17  Cygnus X-1 Jet         Gamma ~ 0.046   astrophysics
P18  AMOC Ocean Current      Gamma ~ 0.251   oceanography
P19  Amazon Rainforest        Gamma ~ 0.116   ecology
P22  BTW Sandpile SOC         Gamma ~ 0.376   statistical mechanics
P30  Diffusive Routing        Gamma ~ 0.443   network science
P36  S_A/S_V Duality          foundational   mathematical physics  <-- HERE
P37  EML-UTAC Bridge          meta           mathematical foundations
P38  Phi^(1/3) Validator      meta-analysis  cross-domain
```
