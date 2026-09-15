# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.1] - 2026-09-15

### Fixed (test suite only, no behavior change)
- Removed `tests/test_preset.py`, `tests/test_validator.py`: unmodified
  copies of `diamond-setup`'s own test suite, exercising only
  `diamond_setup` internals, never `sa-sv-duality` code.
- `tests/test_cli.py` replaced with real tests against this package's
  own `cli.py` (`run`/`q4-map`/`route`/`version`), previously a copy of
  diamond-setup's CLI tests with zero actual coverage of this package's
  own CLI.
- `tests/test_sa_sv_duality.py::test_system_zenodo_record` asserted the
  wrong DOI (`10.5281/zenodo.17472834`, actually `afet-tensions`'
  DOI — evidently a copy-paste error), while `system.py`'s
  `to_zenodo_record()` correctly reports this package's own
  `10.5281/zenodo.20842509` (matching `__zenodo__` in `__init__.py`).
  Corrected the test.

### Fixed (metadata, no behavior change)
- `src/sa_sv_duality/__init__.py`'s `__version__` was never updated for
  the 1.0.0 release — still read `"0.1.0"`. Corrected to track the
  actual released version going forward.

## [1.0.0] - 2026
### Added
- Initial v1.0.0 release as part of the GenesisAeon ecosystem-wide 1.0.0
  milestone.
- Standardized release tooling: `.zenodo.json`, GitHub Actions release
  workflow (`.github/workflows/release.yml`), `RELEASE_GUIDE.md`,
  `CONTRIBUTING.md`, issue/PR templates.

### Changed
- Project metadata (`pyproject.toml`) normalized: version bumped from
  0.1.0 to 1.0.0.
