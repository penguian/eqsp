# Volume 2: The Maintenance Guide

This guide is intended for developers, contributors, and project maintainers of the **PyEQSP** library. It covers the technical architecture, optimization strategies, and the release lifecycle.

## Architecture & Design

The `eqsp` package is designed as a vectorized Python port of the original MATLAB toolbox. The core logic resides in:
- `eqsp/partitions.py`: Recursive zonal partitioning algorithms.
- `eqsp/point_set_props.py`: Metric calculation and energy summation.
- `eqsp/region_props.py`: Geometric property analysis.
- `eqsp/utilities.py`: Coordinate transforms and manifold mathematics.

For more details on the internal layout, see the [Design & Architecture](design_and_architecture.md) guide.
For the transition from MATLAB, see the [User Migration Guide](user_migration_guide.md).

## Algorithmic Optimizations

PyEQSP leverages NumPy for vectorization and spatial indexing to achieve $O(N \log N)$ or $O(N)$ performance for most operations. Technical details can be found in:
- [Algorithmic Optimizations](internal/algorithmic_optimizations.md)
- [Performance Benchmarks](benchmarks.md)

## Quality Assurance & Verification

We maintain a strict quality policy to ensure the reliability of the research outputs:
- **Linting**: 10.00/10 Pylint and zero-Ruff compliance is required. The `ruff.toml` uses a [flat compatibility format](internal/testing_details.md#configuration-compatibility) to support both legacy and modern environments.
- **Testing**: See [Technical Testing & Verification](internal/testing_details.md) for details on the `pytest` suite and coverage analysis.

## Documentation Management

Documentation is managed using Sphinx and MyST-Parser.
- **Build Command**: `cd doc && make html`
- **Configuration**: Managed via `doc/conf.py`.
- **Branding**: Ensure the distinction between the project name (**PyEQSP**) and the import name (`eqsp`) is maintained in all documents.
- **Linguistic Standard**: Adopt **Australian -ize English** (Oxford spelling).
  - Use `-re` and `-our` (e.g., *centre*, *colour*).
  - Prefer `-ize` and `-yze` suffixes (e.g., *organized*, *analyze*).

For standard operating procedures regarding building and hosting, see the [Documentation Maintenance Guide](documentation_maintenance.md).

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
- **PyPI**: Use API tokens rather than account passwords. Store tokens in `~/.pypirc` or provide them via the `TWINE_TOKEN` environment variable.
- **SourceForge**: Managed via SSH keys. The `upload_sourceforge.py` script generates an `scp` command but does not execute it, allowing the maintainer to review and authenticate manually.

Non-owners should never have access to production secrets. All automation is designed to be run from developers' local machines using their own credentials.

## Release & Lifecycle

### Release Procedures

Release 0.99.5 introduced a suite of automated scripts to ensure consistency and prevent common errors:

1. **Build and Validate**: Use `scripts/build_dist.py` to generate the distribution and run `twine check`.
2. **PyPI Upload**: Use `scripts/upload_release.py --testpypi` or `--pypi`. This script ensures a fresh build and converts relative documentation links to absolute URLs for correct rendering on PyPI.
3. **SourceForge Upload**: Use `doc/maint/upload_sourceforge.py` to generate the `scp` command for hosting the HTML documentation.

For detailed instructions on these scripts, see the internal [Upload Guide](internal/upload_guide.md).

### Latest Release Notes
Historical and current release details are tracked in the `doc/internal/` directory:
- [Release Notes 0.99.4](internal/release_notes_0_99_4.md)
- [Release Roadmap](internal/release_roadmap.md)

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

For the full list of mathematical foundations and technical resources cited in this volume, see the [References](references_vol2.md) chapter.
