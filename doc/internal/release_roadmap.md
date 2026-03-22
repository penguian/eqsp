# PyEQSP Release Roadmap

This roadmap outlines the development phases from the initial beta through the 1.0 general release.

## Phase 1: Symmetric Partitioning & Sphinx — [COMPLETED] (2026-03-08)
**Goal**: Implement `even_collars` and establish professional documentation.

- [x] **Symmetric Logic**: Enable equatorial hyperplane alignment for S² and S³.
- [x] **Sphinx Framework**: Initialize multi-format documentation with MyST-Parser.

## Phase 2: Documentation Audit & Quality Hardening — [COMPLETED] (2026-03-17)
**Goal**: Finalize NumPy docstrings and establish bibliographic consistency.

- [x] **Docstring Audit**: Standardize NumPy Google-style docstrings across all modules.
- [x] **Bibliographic Parity**: Align bibliography with finalized PhD thesis [Leo07].

### 0.99.4 Beta: Maintenance Consolidation
**Released: 2026-03-17** | **Git Tag: release_0_99_4** | **Distribution: PyPI**

- [x] **Porting Guide**: Document changes from MATLAB to Python.
- [x] **Quality Checks**: Automated Aussie -ize orthography enforcement.

## Phase 2b: 0.99.7 Infrastructure Hardening — [COMPLETED] (2026-03-22)
**Goal**: Harden CI infrastructure, implement automated quality guardrails, and address legacy environment compatibility.

### 0.99.7 Beta: Quality Hardening
**Released: 2026-03-22** | **Git Tag: release_0_99_7** | **Distribution: PyPI**

- [x] **Pre-commit Layer**: Formalized the first tier of "Defense in Depth" (Item 1-2).
- [x] **Zero-Warning Policy**: Integrated `make html SPHINXOPTS="-W"` into `verify_all.py` to prevent documentation drift.
- [x] **Environment Isolation**: Improved `verify_all.py` to manage PATH for subprocesses across diverse virtual environments.

### 0.99.8 Beta: Infrastructure Hardening
Scheduled for **2026-03-31** | **Git Tag: Planned** | **Distribution: Planned**

- [ ] **Automation Coverage Plan**: Achieve 100% coverage on the script automation hub (`scripts/`, `doc/maint/`) using modular refactoring, interactive doctests, and mocks.
- [ ] **Verified Deployment**: Confirm stability of the `build_dist.py` atomic backup and robust `doc/conf.py` mocking across all CI runners.
- [ ] **Technical Symmetry Audit**: Review historical and modern Sphinx configurations for project-wide consistency.

### 1.0 General Release: Research Reliability Foundation
**Scheduled for 2026-04-15**

- [ ] **Release Verification**: Verified 1.0 release artifacts and distribution.
- [ ] **User Feedback Audit**: Address final community feedback from the beta cycle.
- [ ] **Final 1.0 Tag**: Canonical production release tag.
