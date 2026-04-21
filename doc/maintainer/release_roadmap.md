# Appendix H: Release Roadmap

This roadmap outlines the development phases from the initial beta through the 1.0 general release.

## Phase 1: Symmetric Partitioning & Sphinx
**Goal**: Implement `even_collars` and establish professional documentation.

### 0.99.0 Beta: Symmetric Logic and Documentation
**Released: 2026-03-08** | **Git Tag: release_0_99_0** | **Distribution: PyPI**

- [x] **Symmetric Logic**: Enable equatorial hyperplane alignment for S² and S³.
- [x] **Sphinx Framework**: Initialize multi-format documentation with MyST-Parser.

### 0.99.3 Beta: Branding & Notation
**Released: 2026-03-14** | **Git Tag: release_0_99_3** | **Distribution: PyPI**

- [x] **PyPI Branding**: Standardized project identity and mathematical notation for web rendering.
- [x] **Scaling Parity**: Verified $O(N^{0.6})$ scaling parity with original MATLAB research.


## Phase 2: Documentation Audit & Quality Hardening
**Goal**: Finalize NumPy docstrings and establish bibliographic consistency.

- [x] **Docstring Audit**: Standardize NumPy Google-style docstrings across all modules.
- [x] **Bibliographic Parity**: Align bibliography with canonical PhD thesis keys [Leo07].


### 0.99.4 Beta: Maintenance Consolidation
**Released: 2026-03-17** | **Git Tag: release_0_99_4** | **Distribution: PyPI**

- [x] **Porting Guide**: Document changes from MATLAB to Python.
- [x] **Quality Checks**: Automated Aussie -ize orthography enforcement.

### 0.99.6 Beta: Release Automation & Governance
**Released: 2026-03-21** | **Git Tag: release_0_99_6** | **Distribution: PyPI**

- [x] **Release Automation**: Introduced `release/` suite to automate build-upload cycles.
- [x] **Governance**: Added Roles & Responsibilities matrix to the Maintenance Guide.
- [x] **Bibliography Consistency**: Manual audit to align all internal citation keys.


## Phase 3: Infrastructure Hardening & Full Coverage
**Goal**: Harden CI infrastructure, implement automated quality guardrails, achieve 100% project-wide coverage, and verify dimensional robustness.

### 0.99.7 Beta: Quality Hardening
**Released: 2026-03-22** | **Git Tag: release_0_99_7** | **Distribution: PyPI**

- [x] **Pre-commit Layer**: Formalized the first tier of "Defense in Depth" (Item 1-2).
- [x] **Zero-Warning Policy**: Integrated `make html SPHINXOPTS="-W"` into `validation/verify_all.py` to prevent documentation drift.
- [x] **Environment Isolation**: Improved `validation/verify_all.py` to manage PATH for subprocesses across diverse virtual environments.

### 0.99.8 Beta: Infrastructure Hardening
**Released: 2026-04-03** | **Git Tag: release_0_99_8** | **Distribution: PyPI**

- [x] **JOSS Submission**: Initiate the peer-review process by submitting the voice-modeled `paper.md` and `paper.bib` artifacts to the Journal of Open Source Software. This establishes the project's scholarly identity alongside the 2007 PhD thesis [Leo07] and 2024 JAS paper [Leo24-JAS].
- [x] **Automation Coverage Plan**: Achieved 100% coverage on the script automation hub (`release/`, `validation/`) using modular refactoring and the `test_ci_scripts.py` suite.
- [x] **Verified Deployment**: Confirmed stability of the `build_dist.py` atomic backup and robust `doc/conf.py` mocking across all CI runners.
- [x] **Technical Symmetry Audit**: Reviewed historical and modern Sphinx configurations for project-wide consistency.
- [x] **Rename COPYING to LICENSE and adopt PEP 639**: Renamed `COPYING` → `LICENSE`; updated `pyproject.toml` with `license = "MIT"` and `license-files = ["LICENSE"]`.
- [x] **Rename `doc/internal/` to `doc/maintainer/`**: Reflected that these are public maintainer-facing docs. Updated all internal references.

### 0.99.9 Beta: Full Project Coverage & Dimensional Robustness
**Released: 2026-04-09** | **Git Tag: release_0_99_9** | **Distribution: PyPI**

- [x] **Histogram Logic Alignment**: Back-ported "Index Rotation" (longitude lookup fix) into `eqsp/histograms.py` and implemented high-$N$ wrap-around tests. Removed legacy `lookup_table()` in favor of domain-translated `np.searchsorted()` for 100% coverage.
- [x] **Coverage Deep-Dive**: Reached 100% functional coverage in all core maintenance and release scripts (`release/`, `validation/`) and core algorithm edge cases (e.g., $dim=1$ scalar paths).
- [x] **Higher-Dimension Robustness**: Verified recursive partitioning for $S^4$ and $S^5$, ensuring coordinate bounds and unit-norm properties hold for $dim \ge 4$.
- [x] **Benchmark Alignment**: Synchronized Python benchmark logic, warm-up phases, and $N_{max}$ parameters with the original MATLAB EQSP Toolbox. Implemented binary scaling (1-2-5) for symmetric benchmarks.
- [x] **Equatorial Symmetry Expansion**: Formalized `even_collars` support across the entire public API (including `eq_find_s2_region`).
- [x] **Verification Hardening**: Implemented robust venv isolation in `validation/verify_all.py` and corrected `sradius` dimension logic.
- [x] **Histogram Boundary Safety**: Added index clamping to `lookup_s2_region` to prevent `IndexError` at poles.
- [x] **CLI Standardization**: Refined `--even-collars` flag logic for intuitive explicit opt-in and added `os.pathsep` portability.
- [x] **Local Config & Environment Standardization**: Isolated AI assistant skills and standardized documentation with generic placeholders (`VENV`) for improved portability and privacy.
- [x] **Documentation Parity**: Conducted a thorough audit of the benchmark suite documentation to ensure $S^2$ and $S^3$ default dimensions are explicitly noted across the User and Maintenance guides.
- [x] **Doc Re-organization**: Refactored guides to use alphabetical appendices and renamed the Migration Guide to "Migration from MATLAB" with integrated performance baselines.

## Phase 4: Open Beta & Community Engagement
**Goal**: Finalize production-ready artifacts, incorporate community feedback, and establish a long-term strategic evolution.

### 1.0b1 Open Beta: Feedback Hub & Community Infrastructure
**Released: 2026-04-19** | **Git Tag: release_1_0b1** | **Distribution: PyPI / GitHub**

- [x] **Feedback Hub**: Integrated a sidebar CTA and global version warning banner into the documentation.
- [x] **Beta Feedback Channel**: Added a specialized GitHub Issue template for verification reports and success stories.
- [x] **Code of Conduct**: Adopted the Contributor Covenant to foster a welcoming community.
- [x] **Enhanced Issue Routing**: Clearly partitioned detailed bug reports from general beta testing observations.
- [x] **Portability Audit**: Added explicit POSIX/Linux caveats for the maintenance environment.
- [x] **Metadata Sync**: Formally bumped project status to `Development Status :: 4 - Beta` and synchronized version counters (1.0b1).

### 1.0 General Release [PLANNED]

- [ ] **User Feedback Audit**: Address final community feedback from the beta cycle.
- [x] **Master ToC Refinement (index.rst)**: Redesigned the master Table of Contents to resolve appendix numbering conflicts and correctly sequence volume references.
- [ ] **Visual Audit**: Final side-by-side verification of example figures vs. PhD Thesis PDF.
- [ ] **Release Verification**: Verified 1.0 release artifacts and distribution.
- [ ] **Final 1.0 Tag**: Canonical production release tag.

## Phase 5: Post-1.0 [UNDER CONSIDERATION]

### Quality & Type Safety
- [ ] **Gradual Typing**: Implement PEP 484 type hints for core package parameters (`dim`, `N`, `s`).
- [ ] **Validation Hardening**: Update `check_links.py` to verify inline code snippet paths.
- [ ] **Centralized Task Runner**: Evaluate `nox` or a unified `Makefile` for task orchestration.
- [ ] **Doctest Hardening**: Fully automate the verification of all documentation code snippets.

### Performance & Optimization
- [ ] **Cut-down Performance Regression (Tier 2)**: Formalize automated performance gating against a baseline JSON.
- [ ] **Dot Product Energy Optimization**: Evaluate faster $2(1 - X^T X)$ dot-product approach for points on the unit sphere.
- [ ] **Vectorized Area Calculations**: Update `eq_area_error` to calculate area once per collar and broadcast.
- [ ] **Acceleration Research**: Evaluate `Numba` or `Cython` for core recursive partitioning loops.

### Maintenance & Automation
- [ ] **Maintenance Consolidation**: Full Pylint-audit and refactor of all legacy benchmark and profiling scripts.
- [ ] **Vale Full Automation**: Integrate `vale` project-wide to enforce Australian English.
- [ ] **Lychee Integration**: Transition to the `Lychee` link checker for deep markdown analysis.

### Research & Dimensionality
- [ ] **Dimensionality Expansion**: Establish reference baselines and mathematical invariant tests for $d \ge 4$.
