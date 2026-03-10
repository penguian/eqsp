# Changelog - PyEQSP

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Symmetric Partitions**: Added the `even_collars` parameter to `eq_caps`, `eq_regions`, and `eq_point_set`. This ensures the equatorial hyperplane aligns with a cap boundary, enabling precise S² hemisphere splitting and $S^3 \rightarrow SO(3)$ quaternion sampling.
- **Improved Docstrings**: Completed a comprehensive audit and standardization of NumPy-format docstrings across the entire public API.
- **CI Robustness**: Enhanced GitHub Actions and verification scripts (`verify_all.py`) to be more resilient across different Python environments.

## [0.98.1] - 2026-03-03

### Fixed
- **Pull Request Reviews**: Addressed code review comments from PR #8 and #9, focusing on robustness, error handling (e.g., `PackageNotFoundError`), and consistency.
- **Linter Robustness**: Improved `ruff` and `pylint` configurations for more consistent development environment checks.

## [0.98.0] - 2026-03-01

### Added
- **Alpha Release**: First functional alpha release of the EQSP Python port.
- **Core Algorithms**: Implementation of `eq_regions`, `eq_point_set`, and `eq_caps` for $S^d$ ($d \ge 1$).
- **Performance Optimizations**: Implemented $O(N \log N)$ minimum distance calculations and $O(N)$ memory Riesz energy summation.
- **Thesis Reproductions**: Included scripts and results to reproduce figures and benchmarks from the original PhD thesis [Leo07].
- **Optional Visualizations**: Logic for 2D Matplotlib projections and 3D Mayavi/PyQt interactive renderings.
- **Documentation**: Initialized Sphinx documentation with Markdown support, including guides for installation and testing.
- **Test Suite**: Comprehensive testing framework with a strict 100% code coverage policy.
