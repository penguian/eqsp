# Volume 2: The Maintenance Guide

This guide is intended for developers, contributors, and project maintainers of the **PyEQSP** library. It covers the technical architecture, optimization strategies, and the release lifecycle.

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

## System Requirements for Maintenance

:::{important}
To date, all development and maintenance of PyEQSP has been performed exclusively on **Linux**.
:::

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
| **Packaging** | `release/build_dist.py` | Orchestrates link sanitization and distribution build. |
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

## Release & Lifecycle

### Release Procedures

PyEQSP uses a staging-based release workflow to protect the `main` branch from unverified distribution metadata:

1. **Staging & TestPyPI**: From a new **release branch**, use `release/upload_release.py --testpypi`. This automatically builds the distribution and uploads to TestPyPI for verification.
2. **Internal Review**: Review the TestPyPI project page to confirm that documentation links correctly point to absolute GitHub URLs.
3. **Integration & PR**: Push the release branch and open a PR to `main`. Once CI passes and the PR is merged, the authoritative `main` branch is ready for deployment.
4. **Production PyPI Upload & Tagging**: From the updated `main` branch, use `release/upload_release.py --pypi` for the deployment, then create and push the version tag (e.g., `git tag release_1_0b1 && git push origin release_1_0b1`).
5. **SourceForge Upload**: Use `release/upload_sourceforge.py` to host the mirror of the Sphinx HTML documentation. Note: the primary Sphinx HTML documentation is uploaded to [`readthedocs.io`](https://readthedocs.io) automatically.

For detailed instructions on these scripts, see **Appendix D**: [Upload Guide](maintainer/upload_guide.md).

### Branching and Tagging Strategy

PyEQSP follows a strategy where the `main` branch serves as the authoritative record for all production releases.

| Phase | Branch | Action | Upload Target |
| :--- | :--- | :--- | :--- |
| **1. Staging** | `release_branch_1_0b1` | Create branch; Version bump; `upload_release.py --testpypi` | **TestPyPI** |
| **2. Verification** | `release_branch_1_0b1` | Confirm rendering; Push PR to GitHub | *CI Only* |
| **3. Integration** | `main` | **Merge** PR into `main` | — |
| **4. Deployment** | `main` | `upload_release.py --pypi` | **Production PyPI** |
| **5. Snapshot** | `main` | `git tag release_1_0b1` | **GitHub Tags** |

#### Commits and Tags
Every release tag (e.g., `release_1_0b1`) is applied exclusively to the **merge commit** on the `main` branch. This ensures that the tag points to a state of the code that has been fully integrated, verified by CI, and matches the distribution uploaded to PyPI. Consequently, **tagged commits are always on the `main` branch**.

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

#### Production PyPI Upload Failure
If an upload to the production PyPI repository fails, it is often due to PyPI's strict immutability rules.

1.  **Check Immutability**: If any artifact (sdist or wheel) was successfully accepted before the failure, you **cannot** re-upload the same version string.
2.  **Consult Failure Recovery**: For a detailed step-by-step recovery process, see the **PyPI Immutability & Failure Recovery** caution in the [Upload Guide](maintainer/upload_guide.md).
3.  **Increment Version**: You must increment the version in `pyproject.toml` (e.g., to `1.0b2`) before restarting the release process.

### Historical Release Notes
Unlike the user-facing [CHANGELOG.md](https://github.com/penguian/pyeqsp/blob/main/CHANGELOG.md), which focuses on library features and API changes, the **Historical Release Notes** track the internal evolution of the project's infrastructure, CI/CD pipelines, and maintenance automation.

Detailed logs and historical metadata are tracked in:
- [Historical Release Notes](maintainer/release_notes.md): History of infrastructure changes and release-time quality metrics.
- [Release Roadmap](maintainer/release_roadmap.md): Strategic development phases and planned milestones.

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

For the full list of mathematical foundations and technical resources cited in this volume, see the [References](maintainer/references_vol2.md) chapter.
