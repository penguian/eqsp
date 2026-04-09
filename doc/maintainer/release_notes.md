# Historical Release Notes

This document contains the historical configuration, changes, and checklists for PyEQSP beta releases prior to 1.0.

## 0.99.9
**2026-04-09**

Release **0.99.9** achieves **100% project-wide coverage** across both the core library and the entire maintenance ecosystem. It finalizes the dimensional robustness and equatorial symmetry features for the stable 1.0 release.

### Key Features & Improvements
- **Equatorial Symmetry**: Expanded the `even_collars` parameter across the core API (`eq_regions`, `eq_point_set`, `eq_find_s2_region`, etc.) to ensure partitions align perfectly with the equatorial hyperplane.
- **Symmetric Benchmark Suite**: Added a full suite of even-collar benchmark runners (`run_benchmarks_even.py`) with 1-2-5 logarithmic scaling to verify performance parity.
- **Robustness Testing**: Extended formalized testing to higher dimensions ($S^4$ and $S^5$), verifying the recursive algorithm's mathematical stability for $dim \ge 4$.
- **Histogram Logic Alignment**: Back-ported "Index Rotation" (longitude lookup fix) into `eqsp/histograms.py` and implemented high-$N$ wrap-around tests. Removed legacy `lookup_table()` in favor of domain-translated `np.searchsorted()` for 100% coverage.
- **Verification Hardening**: Implemented robust venv isolation in `verify_all.py` and corrected `sradius` dimension logic.
- **Histogram Boundary Safety**: Added index clamping to `lookup_s2_region` to prevent `IndexError` at poles.
- **CLI Standardization**: Refined `--even-collars` flag logic for intuitive explicit opt-in.
- **Verification Suite**: Integrated the unified `verify_all.py` script and benchmark suite.
- **100% Coverage**: Reached 100% functional coverage across all maintenance and release tools.
- **Release Metadata Sync**: Updated versioning and copyright dates for the 1.0 release.
- **Documentation Parity**: Conducted a thorough audit of the benchmark suite documentation to ensure $S^2$ and $S^3$ default dimensions are explicitly noted across the User and Maintenance guides.
- **Local Config Isolation**: Isolated AI assistant skills and standardized virtual environment documentation with generic placeholders (`VENV`) for improved portability and privacy.
- **Environment Support**: Formalized the requirement for Mayavi-enabled environments for 3D visualization documentation builds.

### Release Metadata
- **Version**: 0.99.9
- **Tag**: `release_0_99_9`
- **Distribution**: PyPI, SourceForge
- **Verification**: [PASSED] 100% Core Coverage, 0 Ruff errors, 0 Sphinx warnings.

---

## 0.99.8
**2026-04-03**

Release **0.99.8** establishes the definitive architectural layout for the PyEQSP maintenance ecosystem and finalizes the JOSS submission artifacts.

### Key Features & Improvements
- **Infrastructure Reorganization**: Redesigned the maintenance workspace into intent-based `release/` (distribution) and `validation/` (quality) directories.
- **100% Automation Coverage**: Achieved full unit test coverage for the CI/CD suite via the `test_ci_scripts.py` framework.
- **JOSS Submission**: Finalized the `paper.md` and `paper.bib` artifacts, harmonizing the project's scholarly identity with the 2007 PhD thesis (@Leo07) and 2024 JAS paper (@Leo24).
- **Quality Gate Stability**: Enhanced validation scripts with AST parsing to monitor paths inside Python `Usage:` blocks and enforced structural integrity in `index.rst`.
- **Hemisphere Refinement**: Explicitly documented the `even_collars` implementation for $S^3 \to \text{SO}(3)$ sampling applications.

### Release Metadata
- **Version**: 0.99.8
- **Tag**: `release_0_99_8`
- **Distribution**: PyPI
- **Verification**: [PASSED] 100% Core Coverage, 10.00/10 Pylint, 0 Sphinx warnings.

---

## 0.99.7
**2026-03-22**

Release **0.99.7** is a critical infrastructure synchronization release focusing on headless CI support and across-environment compatibility.

### Key Features & Improvements
- **CI Pipeline Hardening**: Implemented mocking for `mayavi` and `PyQt5` in Sphinx to allow 3D visualization doctests to pass in headless CI runners.
- **Environment Isolation**: Improved `verify_all.py` to manage the execution `PATH` dynamically, resolving path-shadowing issues in complex virtual environments.
- **Legacy Compatibility**: Reverted `ruff.toml` to the flat-format configuration to support institutional environments using older toolchain versions (e.g., Ruff 0.0.291).
- **Credential Logic**: Refined `upload_release.py` to correctly handle standard `TWINE_PASSWORD` and token authentication.

### Release Metadata
- **Version**: 0.99.7
- **Tag**: `release_0_99_7`
- **Distribution**: PyPI
- **Verification**: [PASSED] 100% Core Coverage, 0 Ruff errors (legacy-mode), 0 Sphinx warnings.

---

## 0.99.6
**2026-03-21**

Release **0.99.6** establishes the final infrastructure for automated PyPI deployment and formalizes project governance standards.

### Key Features & Improvements
- **Release Automation**: Introduced the `release/` suite (`build_dist.py`, `pypi_readme_fix.py`) to automate the clean-build-upload cycle.
- **Prose & Tonal Audit**: Standardized on an active, professional voice across the User Guide and README, verified against Flesch-Kincaid readability baselines.
- **Bibliography Parity**: Conducted a manual audit to align all citation keys across `AUTHORS.md` and documentation with canonical PhD thesis keys.
- **Governance**: Added a Roles and Responsibilities matrix and credential management policies to the Maintenance Guide.

### Release Metadata
- **Version**: 0.99.6
- **Tag**: `release_0_99_6`
- **Distribution**: PyPI
- **Verification**: [PASSED] 100% Core Coverage, 10.00/10 Pylint, 0 Sphinx warnings.

---

## 0.99.4
**2026-03-17**

Release **0.99.4** bifurcates the documentation suite and introduces automated orthography enforcement.

### Key Features & Improvements
- **Two-Volume Documentation**: Split the documentation into a dedicated User Guide (Vol 1) and Maintenance Guide (Vol 2).
- **Orthography Enforcement**: Introduced automated project-wide enforcement of **Australian -ize English** (Oxford spelling).
- **Shell Compatibility**: Standardized on quoted `pip install` extras for improved cross-platform reliability.
- **Visualization Cleanup**: Removed legacy illustration stubs in favor of the specialized `eqsp.visualizations` module.
- **Infrastructure**: Transitioned to flat-format linter configurations and enforced `sys.path` isolation for all documentation diagnostic tools.

### Release Metadata
- **Version**: 0.99.4
- **Tag**: `release_0_99_4`
- **Distribution**: PyPI, TestPyPI
- **Verification**: [PASSED] 100% coverage, 10.00/10 Pylint, 0 Sphinx warnings.

---

## 0.99.3
**2026-03-14**

Release **0.99.3** standardizes project branding and enhances mathematical notation for distribution platforms.

### Key Features & Improvements
- **PyEQSP Branding**: Standardized the project identity as **PyEQSP** across all documentation, metadata, and benchmark artifacts.
- **Unicode Mathematics**: Converted inline LaTeX to Unicode (S², S³, O(N)) in top-level Markdown for better rendering on PyPI and SourceForge.
- **Terminology Alignment**: Replaced "legacy" terminology with "original Matlab toolbox" across the internal documentation.
- **Scaling Parity**: Verified $O(N^{0.6})$ scaling parity with the original thesis benchmarks.

### Release Metadata
- **Version**: 0.99.3
- **Tag**: `release_0_99_3`
- **Distribution**: PyPI, TestPyPI
- **Verification**: [PASSED] 100% coverage, 0 Ruff errors.

---

## 0.99.0
**2026-03-08**

Initial Beta release for public testing, introducing symmetric partitioning and the Sphinx documentation framework.

### Key Features & Improvements
- **Symmetric Partitions**: Introduced the `even_collars` parameter to enable precise S² hemisphere splitting and S³ → SO(3) sampling.
- **Sphinx Framework**: Initialized the multi-format documentation system with MyST-Parser and localized cross-references.
- **CI Robustness**: Established the `verify_all.py` unified verification layer and GitHub Actions integration.
- **Docstring Audit**: Completed the first project-wide audit and standardization of NumPy-format docstrings.

### Release Metadata
- **Version**: 0.99.0
- **Tag**: `release_0_99_0`
- **Distribution**: PyPI
- **Verification**: [PASSED] Initial project-wide quality baseline met.
