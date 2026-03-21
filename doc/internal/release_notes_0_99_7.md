# Release Notes & Checklist: PyEQSP 0.99.7

This document summarizes the changes and verification steps completed for the **0.99.7** maintenance release (2026-03-21).

## Summary: CI & Quality Hardening

Release **0.99.7** is a critical synchronization release. It incorporates automated CI fixes, headless environment support, and legacy-compatible configuration to ensure that the 100% quality baseline is maintained across all development and production environments.

## Changes Overview

### 1. CI Pipeline Hardening
- **Dependency Resolution**: Fixed "missing module" errors in GitHub Actions by ensuring all documentation dependencies (Sphinx, MyST, etc.) are installed in the test runner.
- **Headless Environment Support**: Implemented extensive mocking for `mayavi` and `PyQt5` in the Sphinx configuration. This allows the full suite of doctests (including 3D visualizations) to pass on CI runners without requiring a physical GPU or display.

### 2. Environment Compatibility
- **Ruff Configuration**: Reverted `ruff.toml` to the flat-format configuration to ensure full compatibility with legacy Ruff versions (e.g., 0.0.291) used in system-integrated setups like `.venv_sys`.

### 3. Credential Logic Refinement
- **Secure Uploads**: Updated `upload_release.py` to correctly validate standard `TWINE_PASSWORD` environment variables and tokens, removing non-standard variable checks that could lead to authentication false positives.

### 4. Quality Reporting Transparency
- **Expanded Coverage**: Included the `scripts/` directory in formal coverage reports to ensure that all new automation tools meet the project's strict reliability standards.

## Verification Results

| Step | Status | Notes |
|---|---|---|
| **Ruff Linter** | [PASSED] | 0 errors (legacy-compatible mode) |
| **Pylint** | [PASSED] | 10.00/10 rating |
| **Unit Tests** | [PASSED] | 100% coverage on core logic |
| **Doctests** | [PASSED] | All documentation examples verified (mocked in CI) |
| **Sphinx Build** | [PASSED] | 0 warnings, version 0.99.7 confirmed |

## Release Metadata
- **Version**: 0.99.7
- **Build Date**: 2026-03-21
- **Tag Convention**: `release_0_99_7`
- **Distribution Files**: `dist/pyeqsp-0.99.7.tar.gz`, `dist/pyeqsp-0.99.7-py3-none-any.whl`
