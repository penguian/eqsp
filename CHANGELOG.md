# Changelog - PyEQSP

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.99.6] - 2026-03-21

### Added
- **Release Automation**: Introduced `scripts/` to automate build orchestration and PyPI deployment.
  - `pypi_readme_fix.py`: Sanitises documentation links by converting relative paths to absolute GitHub URLs (targeting the `main` branch).
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
- Version 0.99.5 was bypassed in favour of 0.99.6 to resolve TestPyPI immutability conflicts during distribution testing.

## [0.99.4] - 2026-03-17

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

### Added
- **Internal Maintenance**: Started tracking internal documentation for release procedures (`upload_guide.md`) and version stability.

### Changed
- **Linter Configuration**: Updated `ruff.toml` to the modern `[lint]` section format to resolve deprecation warnings.

### Fixed
- **PyPI Rendering**: Switched `README.md` to use Unicode symbols for inline mathematical notation to ensure proper rendering on PyPI/TestPyPI.
- **Branding Standardized**: Replaced instances of "EQSP" with **PyEQSP** and standardized terminology to "original Matlab toolbox" across docstrings and documentation.
- **Mathematics Notation**: Converted inline LaTeX to Unicode (e.g., S², S³, O(N)) in all top-level Markdown files.

## [0.99.2] - 2026-03-10

### Changed
- **Branding**: Rebranded the project as **PyEQSP** in documentation and narrative contexts to clarify the distinction between the project name and the `eqsp` package name.
- **Metadata**: Updated the distribution name to `pyeqsp` in `pyproject.toml` to align with the repository name.

## [0.99.1] - 2026-03-09

### Fixed
- **Beta Documentation**: Resolved an issue where LaTeX equations failed to render on Read the Docs by enabling MyST `dollarmath` and `amsmath` extensions.
- **TOC Formatting**: Standardized the Table of Contents to use automatic numbering and suppressed redundant bullet points on the index page.
- **CSS Specificity**: Strengthened custom CSS to correctly override Read the Docs theme defaults for a cleaner layout.

### Changed
- **Documentation Standardization**: Removed all hardcoded numbering from Markdown headers in favor of Sphinx's automatic `:numbered:` directive.

## [0.99.0] - 2026-03-08

### Added
- **Beta Release**: Initial beta release for public testing.
- **Symmetric Partitions**: Added the `even_collars` parameter to `eq_caps`, `eq_regions`, and `eq_point_set`. This ensures the equatorial hyperplane aligns with a cap boundary, enabling precise S² hemisphere splitting and S³ → SO(3) quaternion sampling.
- **Improved Docstrings**: Completed a comprehensive audit and standardization of NumPy-format docstrings across the entire public API.
- **CI Robustness**: Enhanced GitHub Actions and verification scripts (`verify_all.py`) to be more resilient across different Python environments.

## [0.98.1] - 2026-03-03

### Fixed
- **Pull Request Reviews**: Addressed code review comments from PR #8 and #9, focusing on robustness, error handling (e.g., `PackageNotFoundError`), and consistency.
- **Linter Robustness**: Improved `ruff` and `pylint` configurations for more consistent development environment checks.

## [0.98.0] - 2026-03-01

### Added
- **Alpha Release**: First functional alpha release of the PyEQSP Python port.
- **Core Algorithms**: Implementation of `eq_regions`, `eq_point_set`, and `eq_caps` for Sᵈ (d ≥ 1).
- **Performance Optimizations**: Implemented O(N log N) min-distance calculations and O(N) memory Riesz energy summation.
- **Thesis Reproductions**: Included scripts and results to reproduce figures and benchmarks from the original PhD thesis [Leo07].
- **Optional Visualizations**: Logic for 2D Matplotlib projections and 3D Mayavi/PyQt interactive renderings.
- **Documentation**: Initialized Sphinx documentation with Markdown support, including guides for installation and testing.
- **Test Suite**: Comprehensive testing framework with a strict 100% code coverage policy.
