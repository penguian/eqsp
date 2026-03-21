# Release Roadmap: PyEQSP

This document outlines the technical path from the current **0.99.4** state to a production **1.00** release on PyPI. Beta testers and contributors can use this to understand the remaining milestones and stability goals.

## Current Status: 0.99.4 (Feature Complete)
The codebase has been successfully migrated from Matlab to a vectorized Python implementation. Core features, high-fidelity thesis reproduction scripts, and $O(\mathcal{N}^{0.6})$ scaling have been verified. The `even_collars` feature was shipped in 0.99.0.

---

## Phase 1: 0.99 Beta (Functional & Quality Parity) - [COMPLETED]
**Goal**: Final feature inclusion (`even_collars` symmetry), resolution of phase-one code review defects, then feature freeze, CI stabilization, and release candidate verification.

### API Stabilisation & Cleanup — ✅ Done
- **Symmetric Partitions** (`even_collars`): Shipped in **0.99.0**. The `even_collars` parameter in `eq_caps`, `eq_regions`, and `eq_point_set` ensures the equatorial hyperplane aligns with a cap boundary, enabling S² hemisphere splitting and S³ → SO(3) quaternion sampling.
- **API Visibility**: Point-set property functions formally exported at the `eqsp` package level in **0.99.4**.
- **Docstrings**: Comprehensive NumPy-format docstring audit completed in **0.99.0**.

### Documentation — ✅ Done
- **Multi-Volume Documentation**: User Guide (Volume 1) and Maintenance Guide (Volume 2) established in **0.99.4**, with consolidated bibliographies, Mermaid diagrams, and interactive citations.
- **Bibliographic Audit**: All citations upgraded to finalized peer-reviewed publications in **0.99.4**.

### Quality Safeguards & Linting — ✅ Done
- **Automated Drift Prevention**: `doc/maint/check_links.py` and `doc/maint/quality_check.py` introduced in **0.99.4**.
- **Linting**: 10.00/10 Pylint and zero-Ruff compliance enforced project-wide.
- **Coverage**: 100% test coverage achieved in **0.99.4**.

### Build & TestPyPI — ✅ Done
- **Distribution Verification**: `.tar.gz` and `.whl` builds verified using the `build` module.
- **TestPyPI**: Successful internal distribution testing on TestPyPI across 0.99.x releases.
- **Known Issue**: Relative links in `README.md` render incorrectly on PyPI (addressed in 0.99.5).

---

## Phase 2: 0.99.6 Maintenance — [COMPLETED] (2026-03-21)
**Goal**: Automate the distribution pipeline, resolve PyPI link rendering, and formalise project governance.

### Automated Distribution Pipeline
- [x] **`scripts/pypi_readme_fix.py`**: Convert relative Markdown links in `README.md` to absolute GitHub URLs for the PyPI distribution.
- [x] **`scripts/build_dist.py`**: Automate the clean-build-check cycle with a "swap-and-restore" mechanism to ensure absolute links are used in the distribution. [VERIFIED in METADATA]
- [x] **`scripts/upload_release.py`**: Automate TestPyPI and PyPI uploads.
- [x] **`doc/maint/upload_sourceforge.py`**: Automate the SourceForge documentation upload workflow.

### Verification Runner Enhancements
- [x] Extend Ruff/Pylint scans to include the new `scripts/` directory.
- [x] Integrate `make doctest` into `verify_all.py`.
- [x] Add `--pre-release` flag to `verify_all.py` (runs `build_dist.py` without uploading).

### Project Governance
- [x] **Role Matrix**: Define Owner, Administrator, Maintainer, and Contributor roles in the Maintenance Guide.
- [x] **Security Audit**: Document credential management for PyPI tokens, CI secrets, and ReadTheDocs webhooks.

### Testing
- [x] Mock-based tests for all new distribution scripts in `scripts/`.

---

## Phase 2b: 0.99.6 — Optional Quality Polish (Deferred)
**Goal**: Apply prose-level linting and deepen link validation. Lower priority; should not block the path to 1.00.

- [ ] **Vale Prose Linting**: Configure `vale` with the Google style base and custom rules for Australian English. Integrate as a warning-only step in `verify_all.py`.
- [ ] **Sphinx Inventory Validation**: Upgrade `check_links.py` to check intersphinx targets via `objects.inv`. Prerequisite: adopt `intersphinx_mapping` in `doc/conf.py` for NumPy/SciPy.

---

## Phase 3: 1.00 Release (API Freeze & Public Deployment)
**Goal**: Lock the API surface, expand CI coverage, and execute the production deployment.

### API Freeze & CI
- [ ] **API Freeze**: Document the completed API freeze (signatures frozen at 0.99.0, exports at 0.99.4) in the Maintenance Guide; no new code change required.
- [ ] **CI Matrix Expansion**: Expand GitHub Actions to test Python 3.11, 3.12, and 3.13 with Ruff and Pylint as a fast pre-merge gate.

### Production Deployment
- [ ] Bump version to `1.0.0` in `pyproject.toml`.
- [ ] Update `CHANGELOG.md` with the 1.00 entry.
- [ ] Run `scripts/upload_release.py --testpypi` and perform a visual audit.
- [ ] Create signed tag: `git tag -s v1.0.0 -m "PyEQSP 1.0.0"` and push.
- [ ] Run `scripts/upload_release.py --pypi` for production upload.
- [ ] Update ReadTheDocs, SourceForge website, and SourceForge code repository.

### Final Verification
- [ ] **Clean-room Validation**: Install from PyPI in a fresh venv; confirm all PhD thesis reproduction scripts pass.
- [ ] **Documentation Release**: Review Volume 1 and Volume 2 as official public-facing books. Confirm all developer-only artifacts are excluded from the public release branch.

---

## Future Growth (Post-1.0)
These items represent the long-term vision for the library:

| Item | Description |
|---|---|
| **Compiled Kernels** | Port recursive collar loops to Cython or Numba for near-native C/LLVM speed. |
| **Persistent Caching** | Disk-based or memory-resident caches for regional properties across sequential research runs. |
| **Advanced Vectorization** | Research `awkward-array` ragged structures to remove loops over varying partition sizes $N$. |
| **Intersphinx Integration** | Link PyEQSP docs to NumPy/SciPy API references; prerequisite for Phase 2b Sphinx Inventory Validation. |
