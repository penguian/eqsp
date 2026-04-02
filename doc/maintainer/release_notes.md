# Historical Release Notes

This document contains the historical configuration, changes, and checklists for PyEQSP beta releases prior to 1.0.

## 0.99.8

This section summarizes the completion of the infrastructure and automation hardening phase (2026-04-01).

### Summary: Infrastructure Semantic Hardening

Release **0.99.8** provides the definitive architectural layout for the PyEQSP maintenance ecosystem. By partitioning tools into `release/` and `validation/` and achieving 100% automation coverage, we have reached the highest state of quality and documentation visibility to date. This release also marks the finalization of the JOSS submission draft (`paper.md` and `paper.bib`), harmonized with the project's long-term academic heritage.

### Changes Overview

#### 1. Infrastructure Reorganization
- **Semantic Partitioning**: Redesigned the maintenance directory from a generic `scripts/` layout to specialized **intent-based** directories:
    - **`release/`**: Publishing and distribution tools (Build, PyPI, SourceForge).
    - **`validation/`**: Internal quality gates and metrics (Link check, Quality audit, Readability).
- **Doc-to-Code Alignment**: Updated the entire internal manual set and all `validation/verify_all.py` logic to follow the semantic shift.

#### 2. 100% Automation Coverage
- **Refactored CI/CD Scripts**: Reworked all quality and deployment tools to share a standard, testable `main()` entry point, enabling robust subprocess mocking and dynamic root pathing.
- **Coverage Master**: Achieved a 100% core coverage baseline and a **97% overall project coverage** by implementing the comprehensive `test_ci_scripts.py` suite.

#### 3. Quality Gate Stability
- **Implicit Path Resilience**: Enhanced internal validation scripts with explicit `sys.path` root injection, ensuring they operate flawlessly across local, CI, and pre-commit environments.
- **Hook Integration**: Updated `.pre-commit-config.yaml` to leverage the new semantic paths and project-wide linting.

#### 4. JOSS Submission & Scholar Identity
- **Voice-Modeled Paper**: Rewrote `paper.md` in a professional academic voice, modeling your established prose style from the 2024 JAS paper (@Leo24).
- **Academic Citation Alignment**: Implemented a dual-citation strategy for the MATLAB EQ toolbox, correctly crediting the 2007 PhD thesis (@Leo07) for its origins and the 2024 JAS paper (@Leo24) for the definitive toolbox reference.
- **Bibliography Synchronization**: Hardened `paper.bib` by synchronizing its keys with the project's **User Guide** (Volume 1) and **Maintenance Guide** (Volume 2), ensuring a unified scholarly identity across the documentation and the submission.
- **Hemisphere Refinement**: Explicitly documented the `even_collars` implementation decision to ensure partitions align with the equatorial hyperplane, essential for $S^3$ to $\text{SO}(3)$ sampling applications.

### Verification Results

| Step | Status | Notes |
|---|---|---|
| **Ruff Linter** | [PASSED] | 0 errors |
| **Pylint** | [PASSED] | 10.00/10 rating (Overall) |
| **Unit Tests** | [PASSED] | 100% core coverage / 97% total |
| **Sphinx Build** | [PASSED] | 0 warnings |
| **Audit Checks** | [PASSED] | Links, Quality, and Orthography verified |

### Release Metadata
- **Version**: 0.99.8
- **Build Date**: 2026-04-01
- **Tag Convention**: `release_0_99_8`
- **Distribution Files**: `Planned`

---

## 0.99.7

This section summarizes the changes and verification steps completed for the **0.99.7** maintenance release (2026-03-22).

### Summary: CI & Quality Hardening

Release **0.99.7** is a critical synchronization release. It incorporates automated CI fixes, headless environment support, and legacy-compatible configuration to ensure that the 100% quality baseline is maintained across all development and production environments.

### Changes Overview

#### 1. CI Pipeline Hardening
- **Dependency Resolution**: Fixed "missing module" errors in GitHub Actions by ensuring all documentation dependencies (Sphinx, MyST, etc.) are installed in the test runner.
- **Headless Environment Support**: Mocks `mayavi` and `PyQt5` in the Sphinx configuration. This enables the full suite of doctests (including 3D visualizations) to pass on CI runners without a physical display.
- **Environment Synchronization**: Updated `verify_all.py` to manage the execution `PATH`, resolving path-shadowing issues that caused documentation builds to fail in certain virtual environment configurations.
- **Pre-commit Integration**: Integrated pre-commit hooks as the first tier of "Defense in Depth" to catch documentation typos and broken links locally before code reaches Git.

#### 2. Environment Compatibility
- **Ruff Configuration**: Reverted `ruff.toml` to the flat-format configuration to ensure full compatibility with legacy Ruff versions (e.g., 0.0.291) used in system-integrated setups like `.venv_sys`.

#### 3. Credential Logic Refinement
- **Secure Uploads**: Updated `upload_release.py` to correctly validate standard `TWINE_PASSWORD` environment variables and tokens, removing non-standard variable checks that could lead to authentication false positives.

#### 4. Quality Reporting Transparency
- **Expanded Coverage**: Included the `release/` directory in formal coverage reports to ensure that all new automation tools meet the project's strict reliability standards.

### Verification Results

| Step | Status | Notes |
|---|---|---|
| **Ruff Linter** | [PASSED] | 0 errors (legacy-compatible mode) |
| **Pylint** | [PASSED] | 10.00/10 rating |
| **Unit Tests** | [PASSED] | 100% coverage on core logic |
| **Doctests** | [PASSED] | All documentation examples verified (mocked in CI) |
| **Sphinx Build** | [PASSED] | 0 warnings, version 0.99.7 confirmed |

### Release Metadata
- **Version**: 0.99.7
- **Build Date**: 2026-03-22
- **Tag Convention**: `release_0_99_7`
- **Distribution Files**: `dist/pyeqsp-0.99.7.tar.gz`, `dist/pyeqsp-0.99.7-py3-none-any.whl`

---

## 0.99.6

This section summarizes the changes and verification steps completed for the **0.99.6** maintenance release (2026-03-21).

### Summary: Distribution & Governance Finalization

Release **0.99.6** is a high-fidelity documentation and distribution audit. It establishes the final infrastructure for automated PyPI deployment and formalizes the maintenance standards required for a stable 1.0.0 release.

### Changes Overview

#### 1. Release Automation & Distribution
- **New `release/` Suite**: Introduced `build_dist.py`, `pypi_readme_fix.py`, and `upload_release.py` to automate the clean-build-check-upload cycle.
- **PyPI-Ready Documentation**: Automated the conversion of relative GitHub links in `README.md` to absolute URLs to ensure correct rendering on PyPI/TestPyPI.
- **Enhanced Verification**: Integrated `cd doc && make doctest` into `verify_all.py` to ensure all documentation code samples remain valid and synchronized with the API.

#### 2. Tonal & Prose Audit (Vale)
- **Active Voice Standard**: Refactored `README.md`, `INSTALL.md`, and the `User Guide` to use a consistent active and professional voice.
- **Step-Based Scannability**: Restructured the **Quick Start** and **Practical Usage** guides around a "Step X" procedural format for improved scannability.
- **Readability Baseline**: Established Flesch-Kincaid and Gunning-Fog readability scores across all documentation tiers.

#### 3. Bibliography & Research Integrity
- **Reference Alignment**: Conducted a manual audit to align all citation keys across `AUTHORS.md` and both documentation volumes with the PhD thesis canonical keys.
- **Automated Parity**: Extended `quality_check.py` to strictly enforce metadata and citation key consistency across the entire documentation suite.

#### 4. Project Governance
- **Role Formalization**: Added a **Roles and Responsibilities** matrix to the `Maintenance Guide`.
- **Credential Management**: Documented security policies for PyPI tokens and SourceForge SSH keys.

### Verification Results

| Step | Status | Notes |
|---|---|---|
| **Ruff Linter** | [PASSED] | 0 errors |
| **Pylint** | [PASSED] | 10.00/10 rating |
| **Unit Tests** | [PASSED] | 100% coverage on core logic |
| **Doctests** | [PASSED] | All documentation examples verified |
| **Sphinx Build** | [PASSED] | 0 warnings, version 0.99.6 confirmed |

### Release Metadata
- **Version**: 0.99.6
- **Build Date**: 2026-03-21
- **Tag Convention**: `release_0_99_6`
- **Distribution Files**: `dist/pyeqsp-0.99.6.tar.gz`, `dist/pyeqsp-0.99.6-py3-none-any.whl`

---

## 0.99.4

This section summarizes the changes and verification steps completed for the **0.99.4** release.

### Changes Overview

#### Documentation Restructuring
- **Two-Volume Series**: Formally bifurcated documentation into Volume 1 (User Guide) and Volume 2 (Maintenance Guide).
- **Interactive Citations**: Consolidated all bibliographies and updated citations to clickable MyST links.
- **New Hubs**: Added `user_guide.md` and `maintenance_guide.md`.
- **Linguistic Standardization**: Adopted **Australian -ize English** (Oxford spelling) – using *organized* and *analyze* alongside *centre* and *colour*.
- **Shell Compatibility**: Quoted all `pip install` extras (e.g., `".[docs]"`) to ensure safe shell expansion across all platforms.
- **Troubleshooting**: Documented procedures for TestPyPI upload "gotchas" and pip caching issues.
- **Diagrams**: Integrated Mermaid architectural diagrams.

#### Build Improvements
- **Makefile**: Fixed `doc/Makefile` to explicitly use `python3`.
- **Ruff Configuration**: Transitioned `ruff.toml` to a flat format for cross-environment compatibility (maintains IDE support in restricted legacy environments).
- **Diagnostic Tool Infrastructure**: Enhanced `validation/` with `sys.path` isolation and headless Matplotlib defaults to ensure verification stability.
- **Warning Mitigation**: Resolved Matplotlib `FigureCanvasAgg` warnings by guarding `plt.show()` calls in example and test scripts.

#### Code Cleanup
- **Visualization Stubs**: Removed redirection stubs (`show_s2_sphere`, etc.) from `eqsp.illustrations`. Users are now directed to `eqsp.visualizations`.
- **Legal Consolidation**: Unified project identity on the single `COPYING` file and standardized the project name as the **Python Equal Area Sphere Partitioning Library**.

### Verification Results
- [x] **Pylint**: 10.00/10 (Verified by `verify_all.py`)
- [x] **Ruff**: Passed (Verified by `verify_all.py`)
- [x] **Coverage**: 100% (Verified by `verify_all.py`)
- [x] **Sphinx**: Build successful with 0 warnings.

### Release Metadata
- **Version**: 0.99.4
- **Target Platform**: TestPyPI / PyPI
- **Tag Convention**: `release_0_99_4`

### Remaining Steps
- [x] Update `CHANGELOG.md` with the finalized 0.99.4 entry.
- [x] Perform TestPyPI verification.
- [x] Merge to `main`.

---

## 0.99.3

This section summarizes the changes and verification steps completed for the **0.99.3** release.

### Changes Overview

#### Branding & Consistency
- **Branding Standardized**: Replaced instances of "EQSP" with **PyEQSP** in `README.md`, `CHANGELOG.md`, and benchmark outputs.
- **Removed "legacy"**: Standardized terminology to "original Matlab toolbox" or similar across docstrings and documentation.
- **Mathematics Notation**: Converted inline LaTeX to Unicode (e.g., **S²**, **S³**, **O(N)**) in all top-level Markdown files for better rendering on PyPI/SourceForge.

- **Lint Configuration**: Updated `ruff.toml` to the modern `[lint]` section format to resolve deprecation warnings.

### Verification Results
- **Pylint**: 10.00/10
- **Ruff**: 0 errors, 0 warnings.
- **Test Coverage**: **100.0%** across all modules.
- **Benchmarks**: Verified $O(N^{0.6})$ scaling parity with the original thesis.

### Release Metadata
- **Version**: 0.99.3
- **Target Platform**: TestPyPI (Initial), PyPI (Final)
- **Tag Convention**: `release_0_99_3`

### Remaining Manual Steps before 1.0.0
- [x] Merge `release_0_99_3` to `main`.
- [x] Tag the merge commit as `release_0_99_3`.
- [x] Perform final SourceForge project web upload (see [upload_guide.md](upload_guide.md)).
