# Volume 2: The Maintenance Guide

This guide is intended for developers, contributors, and project maintainers of the **PyEQSP** library. It covers the technical architecture, optimization strategies, and the release lifecycle.

## System Requirements for Maintenance

> [!IMPORTANT]
> To date, all development and maintenance of PyEQSP has been performed exclusively on **Linux**.

The maintenance infrastructure—including `verify_all.py`, the `release/` suite, and the documentation `Makefile`—currently assumes a **POSIX-compatible environment**. Developers using native Windows are encouraged to use **WSL (Windows Subsystem for Linux)** or manually adapt the scripts for their local environment. We provide no guarantees for non-POSIX platforms until they are formally verified by the community.

## Architecture & Design

The `eqsp` package is designed as a vectorized Python port of the original MATLAB toolbox. The core logic resides in:
- `eqsp/partitions.py`: Recursive zonal partitioning algorithms.
- `eqsp/point_set_props.py`: Metric calculation and energy summation.
- `eqsp/region_props.py`: Geometric property analysis.
- `eqsp/utilities.py`: Coordinate transforms and manifold mathematics.

For more details on the internal layout, see the [Design & Architecture](maintainer/design_and_architecture.md) guide.
For the transition from MATLAB, see the [Migration from MATLAB Toolbox](user/migration_matlab.md).

## Algorithmic Optimizations

PyEQSP leverages NumPy for vectorization and spatial indexing to achieve $O(N \log N)$ or $O(N)$ performance for most operations. Technical details can be found in:
- [Algorithmic Optimizations](maintainer/algorithmic_optimizations.md)
- [Performance Benchmarks](maintainer/benchmarks.md)

## Quality Assurance & Verification

We maintain a strict quality policy to ensure the reliability of the research outputs.

### Automated Verification

To prevent regressions, **pre-commit hooks** are used to validate every commit locally. Run once to set up:
```bash
pre-commit install
```
The hooks encompass formatting, linting, documentation quality checks, and link validation.

The primary project-wide entry point for global quality control is `verify_all.py` (located in `validation/`).
- **Pull Requests**: Every PR must pass all pre-commit hooks and `python3 validation/verify_all.py` (Ruff, Pylint, Pytest, Doctest). See the internal [Pull Request Checklist](maintainer/pr_checklist.md) for a manual pre-submission guide.
- **Environment Isolation**: Use `python3 validation/verify_all.py --venv DIR` to run the suite in a specific virtual environment. This automatically "deactivates" any currently active environment to ensure the audit runs against the intended site-packages.
- **Clean Verification**: Use `python3 validation/verify_all.py --uninstall` to remove any existing `pyeqsp` installation from the target environment. This is recommended to ensure that stale package data in `site-packages` does not shadow the local code being verified.
- **Maintenance & Infrastructure**: When modifying repository tools, scripts, or core documentation, follow the internal [Maintenance Implementation Checklist](maintainer/maintenance_implementation_checklist.md) to ensure tonal and technical consistency.
- **Pre-release**: Use `python3 validation/verify_all.py --pre-release` to build the distribution and verify metadata before any upload.

### Verification Strategy: Defense in Depth

PyEQSP employs a three-tier **"Defense in Depth"** strategy to ensure project-wide reliability:

-   **Layer 1: Pre-commit Hooks (Local)**: Provides rapid feedback for formatting, linting, and documentation errors before you commit code.
-   **Layer 2: Unified Verification Script (Local/Orchestration)**: A high-fidelity "dry run" that synchronizes the execution environment to verify 100% test coverage and build stability before you open a Pull Request.
-   **Layer 3: Continuous Integration (CI/Autoritative)**: Verifies the codebase in a clean-room environment across multiple Python versions (3.11–3.13) to catch platform regressions.

This layered approach is complemented by **Project-Specific Guardrails** that enforce research integrity (e.g., bibliographic consistency, manifold naming, and positional-only argument audits).

### Maintenance Scripts Inventory

| Script | Location | Purpose |
|---|---|---|
| **Verification** | `validation/verify_all.py` | Orchestrates Ruff, Pylint, and Pytest with coverage. |
| **Readability** | `validation/compute_readability.py` | Monitors Flesch-Kincaid and Gunning-Fog scores. |
| **Link Check** | `validation/check_links.py` | Validates internal and external documentation URLs. |
| **Quality Audit** | `validation/quality_check.py` | Enforces bibliography/citation consistency. |
| **Packaging** | `release/build_dist.py` | Orchestrates link sanitisation and distribution build. |
| **Link Fix** | `release/pypi_readme_fix.py` | Converts relative GitHub links to absolute URLs for PyPI. |
| **Upload** | `release/upload_release.py` | Manages authenticated uploads to PyPI/TestPyPI. |
| **SourceForge** | `release/upload_sourceforge.py` | Generates the SCP command for website hosting. |
| **PR Checklist** | `doc/maintainer/pr_checklist.md` | General technical verification for code contributions. |
| **Maint Checklist** | `doc/maintainer/maintenance_implementation_checklist.md` | Audit for infrastructure and documentation hardening. |

For technical details on the testing infrastructure, see **Appendix A**: [Technical Testing & Verification](maintainer/testing_details.md).

## Documentation Management

Documentation is managed using Sphinx and MyST-Parser.
- **Build Command**: `cd doc && make html`
- **Configuration**: Managed via `doc/conf.py`.
- **Branding**: Ensure the distinction between the project name (**PyEQSP**) and the import name (`eqsp`) is maintained in all documents.
- **Linguistic Standard**: Adopt **Australian -ize English** (Oxford spelling).
  - Use `-re` and `-our` (e.g., *centre*, *colour*).
  - Prefer `-ize` and `-yze` suffixes (e.g., *organized*, *analyze*).

For standard operating procedures about building and hosting, see the [Documentation Maintenance Guide](maintainer/documentation_maintenance.md).

## Project Governance

### Roles and Responsibilities

| Role | Scope | Production Credential Access |
|---|---|---|
| **Owner** | Full admin of GitHub, SourceForge, PyPI | Yes (all) |
| **Administrator** | CI secrets, API tokens, ReadTheDocs | Yes (scoped) |
| **Maintainer** | PR review, merges into `main`, release tags | No |
| **Contributor** | Forked PRs, bug reports, documentation | No |

### Security & Credential Management

Release operations to PyPI and SourceForge require owner or administrator credentials.
- **PyPI**: Use API tokens rather than account passwords. Store tokens in `~/.pypirc` or provide them via the `TWINE_PASSWORD` environment variable (the standard for both tokens and passwords).
- **SourceForge**: Managed via SSH keys. The `upload_sourceforge.py` script generates an `scp` command but does not execute it, allowing the maintainer to review and authenticate manually.

Non-owners should never have access to production secrets. All automation is designed to be run from developers' local machines using their own credentials.

## Release & Lifecycle

### Release Procedures

Release 0.99.7 introduced a suite of automated scripts and strict quality guardrails to ensure consistency:

1. **Build and Check**: Use `release/build_dist.py` to generate the distribution and run `twine check`.
2. **TestPyPI Upload**: Use `release/upload_release.py --testpypi` to verify documentation link rendering on the TestPyPI project page.
3. **Internal Review**: Review the PyPI overview and confirm all relative links now point correctly to GitHub.
4. **Production PyPI Upload**: Once the PR is approved and CI passes, use `release/upload_release.py --pypi` for the final deployment.
5. **SourceForge Upload**: Use `release/upload_sourceforge.py` to host the Sphinx HTML documentation.

For detailed instructions on these scripts, see **Appendix D**: [Upload Guide](maintainer/upload_guide.md).

### Historical Release Notes
Historical and current release details are tracked in the `doc/maintainer/` directory:
- [Historical Release Notes](maintainer/release_notes.md)
- [Release Roadmap](maintainer/release_roadmap.md)
- [Maintenance Checklist](maintainer/maintenance_implementation_checklist.md)

## Appendices

The following appendices provide detailed technical checklists, mathematical derivations, and historical data for project maintenance:

- **Appendix A**: [Technical Testing & Verification](maintainer/testing_details.md)
- **Appendix B**: [Maintenance Checklist](maintainer/maintenance_implementation_checklist.md)
- **Appendix C**: [Pull Request Checklist](maintainer/pr_checklist.md)
- **Appendix D**: [Upload Guide](maintainer/upload_guide.md)
- **Appendix E**: [Mathematical Symmetry Derivations](maintainer/technical_symmetry.md)
- **Appendix F**: [Algorithmic & Performance Details](maintainer/algorithmic_optimizations.md)
- **Appendix G**: [Historical Release Notes](maintainer/release_notes.md)
- **Appendix H**: [Release Roadmap](maintainer/release_roadmap.md)

### Troubleshooting Release Issues

#### Version Mismatch on TestPyPI
If you install from TestPyPI and see an older version (e.g., seeing 0.99.3 when 0.99.4 was expected):

1. **Clear Pip Cache**: Pip may be using a cached version of a previous installation.
   ```bash
   pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyeqsp
   ```
2. **Uninstall First**: Sometimes a clean uninstall is necessary before reinstalling.
   ```bash
   pip uninstall pyeqsp
   ```
3. **Check Propagation**: TestPyPI may take a minute to propagate new uploads. If the mismatch persists, verify the version on the [TestPyPI project page](https://test.pypi.org/project/pyeqsp/).

For the full list of mathematical foundations and technical resources cited in this volume, see the [References](maintainer/references_vol2.md) chapter.
