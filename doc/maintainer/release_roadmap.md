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

### 0.99.8 Beta: Infrastructure Hardening — [COMPLETED] (2026-04-03)
**Released: 2026-04-03** | **Git Tag: release_0_99_8** | **Distribution: PyPI**
**Goal**: Semantically reorganize the maintenance repository, achieve 100% automation coverage, and finalize the JOSS submission.

- [x] **JOSS Submission**: Initiate the peer-review process by submitting the voice-modeled `paper.md` and `paper.bib` artifacts to the Journal of Open Source Software.
- [x] **Automation Coverage Plan**: Achieved 100% coverage on the script automation hub (`release/`, `validation/`) using modular refactoring and the `test_ci_scripts.py` suite.
- [x] **Verified Deployment**: Confirmed stability of the `build_dist.py` atomic backup and robust `doc/conf.py` mocking across all CI runners.
- [x] **Technical Symmetry Audit**: Reviewed historical and modern Sphinx configurations for project-wide consistency.
- [x] **Rename COPYING to LICENSE and adopt PEP 639**: Renamed `COPYING` → `LICENSE`; updated `pyproject.toml` with `license = "MIT"` and `license-files = ["LICENSE"]`.
- [x] **Rename `doc/internal/` to `doc/maintainer/`**: Reflected that these are public maintainer-facing docs. Updated all internal references.

### 0.99.9 Beta: Full Project Coverage & Dimensional Robustness — [COMPLETED] (2026-04-07)
**Released: 2026-04-07** | **Git Tag: release_0_99_9** | **Distribution: PyPI**
**Goal**: Reconcile performance benchmarks, achieve 100% project-wide coverage, and verify higher-dimension robustness.

- [x] **Histogram Logic Alignment**: Back-ported "Index Rotation" (longitude lookup fix) into `eqsp/histograms.py` and implemented high-$N$ wrap-around tests. Removed legacy `lookup_table()` in favor of domain-translated `np.searchsorted()` for 100% coverage.
- [x] **Coverage Deep-Dive**: Reached 100% functional coverage in all core maintenance and release scripts (`release/`, `validation/`) and core algorithm edge cases (e.g., $dim=1$ scalar paths).
- [x] **Higher-Dimension Robustness**: Verified recursive partitioning for $S^4$ and $S^5$, ensuring coordinate bounds and unit-norm properties hold for $dim \ge 4$.
- [x] **Benchmark Alignment**: Synchronized Python benchmark logic, warm-up phases, and $N_{max}$ parameters with the original MATLAB EQSP Toolbox (Conversation 86f21121).
- [x] **Release Tooling Fix**: Corrected the SourceForge project name in `release/upload_sourceforge.py` and verified `scp` deployment steps.
- [x] **Doc Re-organization**: Refactored guides to use alphabetical appendices and renamed the Migration Guide to "Migration from MATLAB" with integrated performance baselines.

### 1.0 General Release [PLANNED]

- [ ] **User Feedback Audit**: Address final community feedback from the beta cycle.
- [ ] **4-Tier Verification Structure**: Implement the structural separation of logic (Tier 1) and reproduction (Tier 3/4) to ensure CI agility.
- [ ] **Visual Audit**: Final side-by-side verification of example figures vs. PhD Thesis PDF.
- [ ] **Release Verification**: Verified 1.0 release artifacts and distribution.
- [ ] **Final 1.0 Tag**: Canonical production release tag.

### Post -1.0 [UNDER CONSIDERATION]
**Goal**: Strategic enhancements for extreme performance, automation, and developer experience.

#### Quality & Type Safety
- [ ] **Gradual Typing**: Implement PEP 484 type hints for core package parameters (`dim`, `N`, `s`).
- [ ] **Validation Hardening**: Update `check_links.py` to verify inline code snippet paths.
- [ ] **Centralized Task Runner**: Evaluate `nox` or a unified `Makefile` for task orchestration.
- [ ] **Doctest Hardening**: Fully automate the verification of all documentation code snippets.

#### Performance & Optimization
- [ ] **Cut-down Performance Regression (Tier 2)**: Formalize automated performance gating against a baseline JSON.
- [ ] **Dot Product Energy Optimization**: Evaluate faster $2(1 - X^T X)$ dot-product approach for points on the unit sphere.
- [ ] **Vectorized Area Calculations**: Update `eq_area_error` to calculate area once per collar and broadcast.
- [ ] **Acceleration Research**: Evaluate `Numba` or `Cython` for core recursive partitioning loops.

#### Maintenance & Automation
- [ ] **Maintenance Consolidation**: Full Pylint-audit and refactor of all legacy benchmark and profiling scripts.
- [ ] **Vale Full Automation**: Integrate `vale` project-wide to enforce Australian English.
- [ ] **Lychee Integration**: Transition to the `Lychee` link checker for deep markdown analysis.

#### Research & Dimensionality
- [ ] **Dimensionality Expansion**: Establish reference baselines and mathematical invariant tests for $d \ge 4$.
