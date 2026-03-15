---
orphan: true
---
# Release Notes & Checklist: PyEQSP 0.99.3

This document summarizes the changes and verification steps completed for the **0.99.3** release.

## Changes Overview

### Branding & Consistency
- **Branding Standardized**: Replaced instances of "EQSP" with **PyEQSP** in `README.md`, `CHANGELOG.md`, and benchmark outputs.
- **Removed "legacy"**: Standardized terminology to "original Matlab toolbox" or similar across docstrings and documentation.
- **Math Notation**: Converted inline LaTeX to Unicode (e.g., **S²**, **S³**, **O(N)**) in all top-level Markdown files for better rendering on PyPI/SourceForge.

### Code Cleanup
- **Visualization Stubs**: Removed redirection stubs (`show_s2_sphere`, etc.) from `eqsp.illustrations`. Users are now directed to `eqsp.visualizations`.
- **Lint Configuration**: Updated `ruff.toml` to the modern `[lint]` section format to resolve deprecation warnings.

### Verification Results
- **Pylint**: 10.00/10
- **Ruff**: 0 errors, 0 warnings.
- **Test Coverage**: **100.0%** across all modules.
- **Benchmarks**: Verified $O(N^{0.6})$ scaling parity with the original thesis.

## Release Metadata
- **Version**: 0.99.3
- **Target Platform**: TestPyPI (Initial), PyPI (Final)
- **Tag Convention**: `release_0_99_3`

## Remaining Manual Steps before 1.0.0
- [ ] Merge `release_0_99_3` to `main`.
- [ ] Tag the merge commit as `release_0_99_3`.
- [ ] Perform final SourceForge project web upload (see [doc/internal/upload_guide.md](upload_guide.md)).
