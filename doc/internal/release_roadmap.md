# Release Roadmap: PyEQSP

This document outlines the technical path from the current **0.99 beta** state to a production **1.00** release on PyPI. It is intended for beta testers and contributors to understand the remaining milestones and stability goals.

## Current Status: 0.99.x Beta
The codebase has been successfully migrated from Matlab to a vectorized Python implementation. Core features, high-fidelity thesis reproduction scripts, and $O(\mathcal{N}^{0.6})$ scaling have been verified.

---

## Phase 1: 0.99 Beta (Functional & Quality Parity)
**Goal**: Final feature inclusion (`even_collars` symmetry), resolution of phase-one code review defects, subsequent feature freeze, CI stabilization, and release candidate verification.

### API Stabilization & Cleanup
- **Final Feature**: Implement the symmetric `even_collars` partition feature to support SO(3) sampling and 3rd-party use cases.
- **API Freeze**: Execute a strict feature freeze. Perform a final audit of the Public API (`point_set_props.py`, `region_props.py`, `visualizations.py`) to ensure naming consistency and stability.
- **Enhanced Documentation**: Ensure all internal math optimizations and complex logic are professionally documented in docstrings.

### Automated Testing Infrastructure
- **Comprehensive CI**: Expand GitHub Actions to automate testing across Python 3.11, 3.12, and 3.13.
- **Strict Linting**: Continuous enforcement of `ruff` and `pylint` (10.00/10 target) via CI gates.

### Build & Distribution
- **Distribution Verification**: Verify `.tar.gz` and `.whl` builds using the `build` module.
- **TestPyPI Milestone**: Perform internal distribution testing on TestPyPI to ensure seamless installation and dependency resolution.

---

## Phase 2: 1.00 Release (PyPI & Public Handover)
**Goal**: Production deployment and public availability of the complete project.

### Integration of Planned Documentation Guides
The following guides are planned for finalization and integration as official documentation books:
- **User Guide**: Comprehensive documentation for end-users, covering installation, basic usage, and 3D visualizations.
- **Maintenance Guide**: Technical details on the library's internal math, performance optimizations, and contribution workflows.

### Production Deployment
- **Official Release**: Bump version to `1.0.0` in `pyproject.toml`.
- **Git Milestone**: Create a signed Git tag `v1.0.0`.
- **PyPI Upload**: Final production upload for public availability (`pip install pyeqsp`).

### Final Verification
- **Clean-room Validation**: Comprehensive final verification of the library installed directly from PyPI against the thesis baseline.
- **Release Ready**: Final audit to ensure all developer-only artifacts and internal tools are purged from the public release branch.
