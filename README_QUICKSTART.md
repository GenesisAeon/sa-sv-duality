# Quick Start -- sa-sv-duality (GenesisAeon Package 36)

## Installation

```bash
pip install sa-sv-duality
# or with uv:
uv tool install sa-sv-duality
```

## Development Setup

```bash
git clone https://github.com/GenesisAeon/sa-sv-duality.git
cd sa-sv-duality
uv sync --dev
pre-commit install
```

## Running Tests

```bash
uv run pytest
# or skip coverage:
uv run pytest --override-ini="addopts="
```

## CLI Quick Reference

```bash
sav run                          # S_A/S_V cycle with default parameters
sav run --gamma 0.46             # custom CREP Gamma
sav q4-map                       # entropy map for all 16 Q4 states
sav route 0 15                   # optimal path through Q4 state space
sav benchmark                    # all benchmark targets
sav benchmark --fast             # skip slow EL-residual test
sav version
```

## Linting

```bash
ruff check src tests
mypy src
```

## Building Docs

```bash
mkdocs serve
```

## Release

Tag a commit to trigger the automated release workflow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow will:
1. Build the wheel + sdist
2. Publish to PyPI via Trusted Publishing
3. Create a GitHub Release with auto-generated notes
4. Trigger Zenodo archival (configure `ZENODO_TOKEN` secret)

---

GenesisAeon Package 36 -- S_A / S_V Entropy Duality
DOI: 10.5281/zenodo.17472834
