# Changelog - PyEQSP

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0b1] - 2026-04-19
**Git Tag**: `release_1_0b1` | **Distribution**: `PyPI / GitHub`

### Added
- **Open Beta Engagement**: Formalized the "Open Beta" cycle with dedicated infrastructure for tester feedback.
- **Feedback Hub**: Integrated a sidebar CTA and global version warning into the documentation.
- **Beta Feedback Template**: Added a specialized GitHub Issue template for verification and success reports.
- **Maintenance Guide Hardening**: Restructured the Maintenance Guide (Volume 2) to prioritize project governance and establish a professional sequence for quality gates and release lifecycle.
- **Portability Hardening**: Added explicit cross-platform caveats and maintenance environment documentation.
- **Community Governance**: Adopted the Contributor Covenant Code of Conduct.
- **Enhanced Issue Routing**: Refined the README and issue templates to clearly distinguish between general beta feedback and detailed bug reports.

### Changed
- **Metadata Synchronization**: Formally bumped project status to `Development Status :: 4 - Beta` and synchronized version counters across all documentation and README headers.
- **Sphinx Optimization**: Added `sphinx-version-warning` to documentation dependencies for better version management on Read the Docs.
- **Orthography Enforcement**: Hardened the automated quality checks to include `synchronization` and `sanitization` invariants.

## [0.99.9] - 2026-04-09
**Git Tag**: `release_0_99_9` | **Distribution**: `PyPI`

### Added
- **Verification Hardening**: Implemented robust venv isolation in `verify_all.py` and corrected `sradius` dimension logic.
- **Histogram Boundary Safety**: Added index clamping to `lookup_s2_region` to prevent `IndexError` at poles.
- **CLI Standardization**: Refined `--even-collars` flag logic for intuitive explicit opt-in.
- **Full Project Coverage**: Reached 100% functional line coverage for the entire PyEQSP repository, including core maintenance, CI, and release infrastructure.
- **Equatorial Symmetry**: Expanded the `even_collars` parameter across the core API and implemented the full `run_benchmarks_even.py` suite with 1-2-5 logarithmic scaling.
- **Higher-Dimension Robustness**: Formalized unit tests for recursive partitioning on $S^4$ and $S^5$, verifying coordinate bounds and strict unit norm properties.
- **Deep-Dive Verification**: Implemented resilient mock tests for elusive edge cases, including network timeouts and filesystem permission errors in CI.
- **Benchmark Documentation Parity**: Audited all project documentation to ensure default benchmark dimensions ($S^2$ vs $S^3$) are explicitly described, matching both Python and MATLAB source baselines.

- **Histogram Logic Alignment**: Back-ported "Index Rotation" (longitude lookup fix) into `eqsp/histograms.py` and implemented polar index clamping in `eqsp/_private/_histograms.py` to prevent `IndexError` at extreme boundaries.
- **Pylint Import Resolution**: Hardened `verify_all.py` with an automated `init-hook` to correctly resolve specialized benchmark source roots during project-wide audits.
- **Benchmark Core Correctness**: Fixed dimension parameters in `sradius` benchmarks to ensure area calculations consistently use the correct embedding space dimensionality.
- **Local Configuration isolation**: Untracked assistant skills and standardized documentation with generic placeholders (`VENV`) for improved privacy and portability.
- **Infrastructure Isolation**: Redesigned virtual environment deactivation in `verify_all.py` to ensure robust subprocess isolation and portable `PYTHONPATH` handling via `os.pathsep`.

### Changed
- **Metadata Synchronization**: Unified versioning across `pyproject.toml` and documentation for the final pre-1.0 release candidate.
- **Maintenance Hardening**: Finalized the "Defense in Depth" strategy with 100% automated verification of all maintenance scripts.

## [0.99.8] - 2026-04-03
**Git Tag**: `release_0_99_8` | **Distribution**: `PyPI`

### Added
- **100% Automation Coverage**: Achieved full unit test coverage for the entire project maintenance ecosystem (Release & Validation tools) via `tests/src/test_ci_scripts.py`.
- **Infrastructure Semantic Reorganization**: Refactored the internal workspace to partition scripts into `release/` (packaging and hosting) and `validation/` (quality gates and metrics).
- **JOSS Submission Finalization**: Prepared the `paper.md` and `paper.bib` artifacts for the Journal of Open Source Software, incorporating a professional academic voice modeled after the 2024 JAS paper (@Leo24).

### Changed
- **Hardened Validation**: Updated `quality_check.py` and `check_links.py` with robust repository root pathing and standard `main()` entry points to ensure stability in pre-commit and CI environments.
- **Unified Documentation Alignment**: Synchronized all internal guides, checklists, and the `Maintenance Guide` with the new semantic directory layout.
- **Academic Lineage Alignment**: Established a dual-citation strategy, referencing the 2007 PhD thesis (@Leo07) for the algorithm's foundation and the 2024 JAS paper (@Leo24) for the definitive toolbox reference.
- **Hemisphere Partitioning Refinement**: Updated the `paper.md` narrative to highlight the library's support for equatorial alignment via the `even_collars` implementation decision, essential for $S^3$ to $\text{SO}(3)$ quaternion sampling.
- **Pre-commit Layer**: Hardened the pre-commit environment with updated hook definitions and expanded project-wide linting coverage.

### Fixed
- **Sphinx Infrastructure**: Created programmatic checks to enforce uniform indentation on `toctree` blocks in `index.rst`, preventing silent file dropping.
- **PR #21 Resolution**: Corrected `ruff.toml` include paths to scan `release/` and `validation/`, introduced Python AST parsing to validate `Usage:` docstrings, and corrected the *Journal of Approximation Software* reference.

## [0.99.7] - 2026-03-22
**Git Tag**: `release_0_99_7` | **Distribution**: `PyPI`

### Added
- **Pre-commit Layer**: Formalized the first tier of "Defense in Depth" by integrating pre-commit hooks for project-wide documentation quality, link validation, and Python linting.

### Fixed
- **CI & Local Hardening**: Resolved "module not found" errors in both GHA and local virtual environments by ensuring all documentation dependencies (Sphinx, MyST) are correctly synchronized.
- **Environment Isolation**: Enhanced `verify_all.py` to manage the system `PATH` dynamically, ensuring that subprocesses always use the tools from the active virtual environment and resolving path-shadowing failures.
- **Headless Doctests**: Mocks `mayavi` and `PyQt5` in `doc/conf.py` to allow 3D visualization doctests to pass in headless CI environments.
- **Environment Compatibility**: Reverted `ruff.toml` to the legacy-compatible flat configuration format to support restricted environments (e.g., `.venv_sys`).
- **Credential Validation**: Refined `upload_release.py` to correctly validate `TWINE_PASSWORD` (and handle tokens) without misidentifying `TWINE_TOKEN` as a standard variable.
- **Coverage Transparency**: Expanded `tests/run_coverage.py` to include the `scripts/` directory in formal quality and coverage reports.
- **Documentation Refinement**: Resolved `toc.not_included` warnings by properly indexing all historical release notes and formalizing the "Defense in Depth" strategy in the Maintenance Guide.

## [0.99.6] - 2026-03-21
**Git Tag**: `release_0_99_6` (retroactive) | **Distribution**: `PyPI`

### Added
- **Release Automation**: Introduced `scripts/` to automate build orchestration and PyPI deployment.
  - `pypi_readme_fix.py`: Sanitizes documentation links by converting relative paths to absolute GitHub URLs (targeting the `main` branch).
  - `build_dist.py`: Orchestrates the build cycle using a "swap-and-restore" mechanism for PyPI-ready READMEs.
  - `upload_release.py`: Manages authenticated uploads to PyPI and TestPyPI.
- **Bibliography Consistency**: Enhanced `quality_check.py` to strictly enforce metadata parity across all reference documents (`AUTHORS.md`, `doc/references_vol*.md`).
- **Reference Alignment**: Conducted a manual audit to align citation keys across multiple volumes, ensuring all documents use the final, canonical keys from the PhD thesis.
- **Readability Integration**: Integrated Vale `Readability` style to establish and monitor readability baselines across three documentation tiers.

### Changed
- **Tonal Alignment**: Shifted the project's primary voice from passive/academic to active/guidance-oriented across `README.md`, `INSTALL.md`, and the `User Guide`.
- **Structural Improvements**: Consolidated redundant definitions and simplified technical terminology in `README.md` for better accessibility.
- **Refined Standards**: Updated the `Maintenance Guide` with formalized project roles, security credential management, and automated Sphinx doctest verification in `verify_all.py`.

### Fixed
- **Migration Guide**: Corrected several porting attributions and coordinate convention notes to match the original Matlab toolbox logic.

## [0.99.5] - Skipped
**Git Tag**: None | **Distribution**: Skipped (TestPyPI iteration only)

- Version 0.99.5 was bypassed in favour of 0.99.6 to resolve TestPyPI immutability conflicts during distribution testing.

## [0.99.4] - 2026-03-17
**Git Tag**: `release_0_99_4` | **Distribution**: `PyPI`

### Added
- **Multi-Volume Documentation**: Established a comprehensive User Guide (Volume 1) and Maintenance Guide (Volume 2) with consolidated bibliographies, Mermaid diagrams, and interactive citations.
- **Quality Safeguards**: Introduced automated drift-prevention scripts (`check_links.py`, `quality_check.py`) to verify documentation links, function references, array conventions, and **Australian -ize English** orthography.
- **Simplified Examples**: Promoted core documentation snippets to standalone, localized examples with integrated reference artifacts.
- **API Visibility**: Formally exported point-set property functions at the `eqsp` package level.

### Changed
- **Research Integrity**: Conducted a comprehensive bibliographic audit, upgrading internal citations to finalized peer-reviewed publications (e.g., `[Kui07]`, `[Leo24-JAS]`) and ensuring all paper titles are quoted **verbatim**, regardless of regional spelling standards.
- **Repository Structure**: Restructured examples into `src/` and `results/` hubs and promoted core property functions to the package level.
- **Standards & Compatibility**: Standardized on canonical Sphinx `{ref}` labels and "flattened" linter configurations for better platform resilience.
- **Verification**: Achieved 100% project-wide coverage and aligned test baselines with the removal of legacy illustration stubs.

### Removed
- **Illustration Stubs**: Removed four `eqsp.illustrations` migration stubs (`show_s2_sphere`, `show_r3_point_set`, `show_s2_region`, `show_s2_partition`) that raised `NotImplementedError`.

### Fixed
- **Robust Visualization**: Applied robust case-insensitive guards (`.lower() != 'agg'`) to all Matplotlib backend checks project-wide, ensuring warning-free operation in both interactive and headless environments.
- **PR #17 Resolution**: Addressed 11 technical and documentation issues, including toctree indentation, array shape descriptions, quoted `pip install` extras, and MyST configuration types.
- **Diagnostic Portability**: Implemented `sys.path` isolation in standalone inspection and quality scripts to allow direct execution from any environment.
- **Build Integrity**: Resolved all "ghost" function references and build warnings to achieve a 100% warning-free Sphinx build.

## [0.99.3] - 2026-03-14
**Git Tag**: `release_0_99_3` (retroactive) | **Distribution**: `PyPI`

### Added
- **Internal Maintenance**: Started tracking internal documentation for release procedures (`upload_guide.md`) and version stability.

### Changed
- **Linter Configuration**: Updated `ruff.toml` to the modern `[lint]` section format to resolve deprecation warnings.

### Fixed
- **PyPI Rendering**: Switched `README.md` to use Unicode symbols for inline mathematical notation to ensure proper rendering on PyPI/TestPyPI.
- **Branding Standardized**: Replaced instances of "EQSP" with **PyEQSP** and standardized terminology to "original Matlab toolbox" across docstrings and documentation.
- **Mathematics Notation**: Converted inline LaTeX to Unicode (e.g., S², S³, O(N)) in all top-level Markdown files.

## [0.99.2] - 2026-03-10
**Git Tag**: `release_0_99_2` | **Distribution**: `PyPI`

### Changed
- **Branding**: Rebranded the project as **PyEQSP** in documentation and narrative contexts to clarify the distinction between the project name and the `eqsp` package name.
- **Metadata**: Updated the distribution name to `pyeqsp` in `pyproject.toml` to align with the repository name.

## [0.99.1] - 2026-03-09
**Git Tag**: `release_0_99_1` | **Distribution**: `PyPI`

### Fixed
- **Beta Documentation**: Resolved an issue where LaTeX equations failed to render on Read the Docs by enabling MyST `dollarmath` and `amsmath` extensions.
- **TOC Formatting**: Standardized the Table of Contents to use automatic numbering and suppressed redundant bullet points on the index page.
- **CSS Specificity**: Strengthened custom CSS to correctly override Read the Docs theme defaults for a cleaner layout.

### Changed
- **Documentation Standardization**: Removed all hardcoded numbering from Markdown headers in favor of Sphinx's automatic `:numbered:` directive.

## [0.99.0] - 2026-03-08
**Git Tag**: `release_0_99_0` | **Distribution**: `PyPI`

### Added
- **Beta Release**: Initial beta release for public testing.
- **Symmetric Partitions**: Added the `even_collars` parameter to `eq_caps`, `eq_regions`, and `eq_point_set`. This ensures the equatorial hyperplane aligns with a cap boundary, enabling precise S² hemisphere splitting and S³ → SO(3) quaternion sampling.
- **Improved Docstrings**: Completed a comprehensive audit and standardization of NumPy-format docstrings across the entire public API.
- **CI Robustness**: Enhanced GitHub Actions and verification scripts (`verify_all.py`) to be more resilient across different Python environments.

## [0.98.1] - 2026-03-03
**Git Tag**: `release_0_98_1` | **Distribution**: `PyPI`

### Fixed
- **Pull Request Reviews**: Addressed code review comments from PR #8 and #9, focusing on robustness, error handling (e.g., `PackageNotFoundError`), and consistency.
- **Linter Robustness**: Improved `ruff` and `pylint` configurations for more consistent development environment checks.

## [0.98.0] - 2026-03-01
**Git Tag**: `release_0_98_0` | **Distribution**: `PyPI`

### Added
- **Alpha Release**: First functional alpha release of the PyEQSP Python port.
- **Core Algorithms**: Implementation of `eq_regions`, `eq_point_set`, and `eq_caps` for Sᵈ (d ≥ 1).
- **Performance Optimizations**: Implemented O(N log N) min-distance calculations and O(N) memory Riesz energy summation.
- **Thesis Reproductions**: Included scripts and results to reproduce figures and benchmarks from the original PhD thesis [Leo07].
- **Optional Visualizations**: Logic for 2D Matplotlib projections and 3D Mayavi/PyQt interactive renderings.
- **Documentation**: Initialized Sphinx documentation with Markdown support, including guides for installation and testing.
- **Test suite**: Comprehensive testing framework with a strict 100% code coverage policy.
