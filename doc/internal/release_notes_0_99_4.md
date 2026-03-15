# Release Notes & Checklist: PyEQSP 0.99.4

This document summarizes the changes and verification steps completed for the **0.99.4** release.

## Changes Overview

### Documentation Restructuring
- **Two-Volume Series**: Formally bifurcated documentation into Volume 1 (User Guide) and Volume 2 (Maintenance Guide).
- **Interactive Citations**: Consolidated all bibliographies and updated citations to clickable MyST links.
- **New Hubs**: Added `user_guide.md` and `maintenance_guide.md`.
- **Troubleshooting**: Documented procedures for TestPyPI upload "gotchas" and pip caching issues.
- **Diagrams**: Integrated Mermaid architectural diagrams.

### Build Improvements
- **Makefile**: Fixed `doc/Makefile` to explicitly use `python3`.
- **Ruff Configuration**: Transitioned `ruff.toml` to a flat format for cross-environment compatibility.

## Verification Results
- [x] **Pylint**: 10.00/10 (Verified by `verify_all.py`)
- [x] **Ruff**: Passed (Verified by `verify_all.py`)
- [x] **Coverage**: 100% (Verified by `verify_all.py`)
- [x] **Sphinx**: Build successful with 0 warnings.

## Release Metadata
- **Version**: 0.99.4
- **Target Platform**: TestPyPI / PyPI
- **Tag Convention**: `release_0_99_4`

## Remaining Steps
- [ ] Finalize documentation content for all hub sections.
- [ ] Perform TestPyPI verification.
- [ ] Merge to `main`.
