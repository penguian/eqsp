# Release Notes & Checklist: PyEQSP 0.99.6

This document summarizes the changes and verification steps completed for the **0.99.6** maintenance release (2026-03-21).

## Summary: Distribution & Governance Finalization

Release **0.99.6** is a high-fidelity documentation and distribution audit. It establishes the final infrastructure for automated PyPI deployment and formalizes the maintenance standards required for a stable 1.0.0 release.

## Changes Overview

### 1. Release Automation & Distribution
- **New `scripts/` Suite**: Introduced `build_dist.py`, `pypi_readme_fix.py`, and `upload_release.py` to automate the clean-build-check-upload cycle.
- **PyPI-Ready Documentation**: Automated the conversion of relative GitHub links in `README.md` to absolute URLs to ensure correct rendering on PyPI/TestPyPI.
- **Enhanced Verification**: Integrated `cd doc && make doctest` into `verify_all.py` to ensure all documentation code samples remain valid and synchronized with the API.

### 2. Tonal & Prose Audit (Vale)
- **Active Voice Standard**: Refactored `README.md`, `INSTALL.md`, and the `User Guide` to use a consistent active and professional voice.
- **Step-Based Scannability**: Restructured the **Quick Start** and **Practical Usage** guides around a "Step X" procedural format for improved scannability.
- **Readability Baseline**: Established Flesch-Kincaid and Gunning-Fog readability scores across all documentation tiers.

### 3. Bibliography & Research Integrity
- **Reference Alignment**: Conducted a manual audit to align all citation keys across `AUTHORS.md` and both documentation volumes with the PhD thesis canonical keys.
- **Automated Parity**: Extended `quality_check.py` to strictly enforce metadata and citation key consistency across the entire documentation suite.

### 4. Project Governance
- **Role Formalization**: Added a **Roles and Responsibilities** matrix to the `Maintenance Guide`.
- **Credential Management**: Documented security policies for PyPI tokens and SourceForge SSH keys.

## Verification Results

| Step | Status | Notes |
|---|---|---|
| **Ruff Linter** | [PASSED] | 0 errors |
| **Pylint** | [PASSED] | 10.00/10 rating |
| **Unit Tests** | [PASSED] | 100% coverage on core logic |
| **Doctests** | [PASSED] | All documentation examples verified |
| **Sphinx Build** | [PASSED] | 0 warnings, version 0.99.6 confirmed |

## Release Metadata
- **Version**: 0.99.6
- **Build Date**: 2026-03-21
- **Tag Convention**: `release_0_99_6`
- **Distribution Files**: `dist/pyeqsp-0.99.6.tar.gz`, `dist/pyeqsp-0.99.6-py3-none-any.whl`
