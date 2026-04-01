# Release Notes & Checklist: PyEQSP 0.99.4

This document summarizes the changes and verification steps completed for the **0.99.4** release.

## Changes Overview

### Documentation Restructuring
- **Two-Volume Series**: Formally bifurcated documentation into Volume 1 (User Guide) and Volume 2 (Maintenance Guide).
- **Interactive Citations**: Consolidated all bibliographies and updated citations to clickable MyST links.
- **New Hubs**: Added `user_guide.md` and `maintenance_guide.md`.
- **Linguistic Standardization**: Adopted **Australian -ize English** (Oxford spelling) – using *organized* and *analyze* alongside *centre* and *colour*.
- **Shell Compatibility**: Quoted all `pip install` extras (e.g., `".[docs]"`) to ensure safe shell expansion across all platforms.
- **Troubleshooting**: Documented procedures for TestPyPI upload "gotchas" and pip caching issues.
- **Diagrams**: Integrated Mermaid architectural diagrams.

### Build Improvements
- **Makefile**: Fixed `doc/Makefile` to explicitly use `python3`.
- **Ruff Configuration**: Transitioned `ruff.toml` to a flat format for cross-environment compatibility (maintains IDE support in restricted legacy environments).
- **Diagnostic Tool Infrastructure**: Enhanced `doc/ci_scripts/` with `sys.path` isolation and headless Matplotlib defaults to ensure verification stability.
- **Warning Mitigation**: Resolved Matplotlib `FigureCanvasAgg` warnings by guarding `plt.show()` calls in example and test scripts.

### Code Cleanup
- **Visualization Stubs**: Removed redirection stubs (`show_s2_sphere`, etc.) from `eqsp.illustrations`. Users are now directed to `eqsp.visualizations`.
- **Legal Consolidation**: Unified project identity on the single `COPYING` file and standardized the project name as the **Python Equal Area Sphere Partitioning Library**.

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
- [x] Update `CHANGELOG.md` with the finalized 0.99.4 entry.
- [x] Perform TestPyPI verification.
- [x] Merge to `main`.
