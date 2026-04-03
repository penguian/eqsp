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

### 0.99.8 Beta: Infrastructure Hardening — [COMPLETED] (2026-04-01)
**Goal**: Semantically reorganize the maintenance repository, achieve 100% automation coverage, and finalize the JOSS submission.

- [x] **JOSS Submission**: Initiate the peer-review process by submitting the voice-modeled `paper.md` and `paper.bib` artifacts to the Journal of Open Source Software.
- [x] **Automation Coverage Plan**: Achieved 100% coverage on the script automation hub (`release/`, `validation/`) using modular refactoring and the `test_ci_scripts.py` suite.
- [x] **Verified Deployment**: Confirmed stability of the `build_dist.py` atomic backup and robust `doc/conf.py` mocking across all CI runners.
- [x] **Technical Symmetry Audit**: Reviewed historical and modern Sphinx configurations for project-wide consistency.
- [x] **Rename COPYING to LICENSE and adopt PEP 639**: Renamed `COPYING` → `LICENSE` via `git mv`; updated prose references in `README.md` and `AUTHORS.md`; replaced the deprecated `license = { file = "COPYING" }` table form in `pyproject.toml` with `license = "MIT"` (SPDX expression), added `license-files = ["LICENSE"]`, and removed the deprecated `License :: OSI Approved :: MIT License` Trove classifier.
- [x] **Rename `doc/internal/` to `doc/maintainer/`**: Replaced the misleading directory name with one that accurately reflects these files as public maintainer-facing documentation. Updated all references in `doc/index.rst`, `doc/maintenance_guide.md`, and seven other files that linked into the directory.

### 0.99.9 Beta: Polishing & Final Verification — [PLANNED]
**Goal**: Address final quality issues, optimize benchmark reporting, and finalize user documentation.

- [ ] **Benchmark Cleanup**: Update benchmark scripts (`benchmarks/src/*.py`) to remove trailing whitespace from outputs and align with pre-commit standards.
- [ ] **Maintenance Consolidation**: Merge legacy profiling scripts into a unified `profile_thesis_figures.py` with full CLI support and Pylint compliance.
- [ ] **Validation Hardening**: Update `check_links.py` to verify inline code snippet paths and evaluate `nox`/`Makefile` task orchestration.
- [ ] **Coverage Deep-Dive**: Exercise `dim=1` scalar paths in `polar_colat` and recursive logic in high-dimensional solvers.
- [ ] **Gradual Typing**: Implement PEP 484 type hints for core package parameters (`dim`, `N`, `s`).
- [ ] **Doc Re-organization**: Refactor guides to use alphabetical appendices (A, B, C) and rename the Migration Guide to "Migration from MATLAB" (removing redundant install blocks).
- [ ] **Visual Audit**: Final side-by-side verification of example figures vs. PhD Thesis PDF.

### 1.0 General Release: Research Reliability Foundation

- [ ] **Release Verification**: Verified 1.0 release artifacts and distribution.
- [ ] **User Feedback Audit**: Address final community feedback from the beta cycle.
- [ ] **Final 1.0 Tag**: Canonical production release tag.

### Future Roadmap: Post-1.0 (Phase 2) — [UNDER CONSIDERATION]
**Goal**: Potential strategic enhancements for extreme performance and automation (not yet formally scheduled).

#### Optimization & Vectorization
- [ ] **Dot Product Energy Optimization**: Evaluate replacing Euclidean `cdist` in `point_set_energy_dist` with a faster $2(1 - X^T X)$ dot-product approach for points on the unit sphere.
- [ ] **Vectorized Area Calculations**: Update `eq_area_error` to calculate area once per collar and broadcast, eliminating redundant $O(N)$ calculations.
- [ ] **Histogram Scaling**: Full vectorization of S2 longitude lookups in `_private/_histograms.py` to support multi-million point samples.
- [ ] **Acceleration Research**: Evaluate `Numba` or `Cython` for core recursive partitioning loops.

#### Infrastructure & Automation
- [ ] **Centralized Task Runner**: Implement a `Makefile` or `nox` configuration as the primary interface for developers and CI.
- [ ] **Vale Full Automation**: Integrate `vale` project-wide to enforce Australian English (Oxford spelling) project-wide.
- [ ] **Doctest Hardening**: Fully automate the verification of all documentation code snippets.
- [ ] **Lychee Integration**: Transition to the `Lychee` link checker for deep markdown semantic analysis and external URL verification.

#### Research & Dimensionality
- [ ] **Dimensionality Expansion**: Establish reference baselines and mathematical invariant tests for $d \ge 4$.
