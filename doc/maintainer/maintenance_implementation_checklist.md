# Appendix B: Maintenance Checklist: PyEQSP Quality Assurance

This checklist provides the canonical project-wide quality gates to be consulted when modifying maintenance code, infrastructure scripts, or the core maintenance documentation.

## 1. Local Verification (Defense Layer 1)
- [ ] **Pre-commit Synchronization**: Activate the virtual environment (`source VENV/bin/activate`, where `VENV` is your virtual environment directory; see the [Installation Guide](../user/installation.md)), then run `pre-commit install` and `pre-commit run --all-files`. The `documentation-quality` hook imports `eqsp` at runtime and requires project dependencies — running outside the venv will fail with `ModuleNotFoundError: No module named 'numpy'`.
- [ ] **Typo Monitoring**: Run `validation/quality_check.py` to ensure roadmap and documentation prose do not trigger literal typo matches (e.g., use `2`-`rd`, `3`-`th`).
- [ ] **Environment Parity**: Confirm that tools work in both `VENV` (the standard isolated environment) and `VENV_SYS` (`VENV_SYS` denotes a virtual environment created with `--system-site-packages`; see the [Installation Guide](../user/installation.md)).

## 2. Infrastructure & Orchestration (Defense Layer 2)
- [ ] **Verification Script**: If `validation/verify_all.py` was modified, ensure it correctly manages the `PATH` environment variable for subprocesses (`os.environ["PATH"]`).
- [ ] **Build Integrity**: Confirm that `make -C doc html SPHINXOPTS="-W"` and `make -C doc doctest` pass from BOTH within and outside the documentation root.
- [ ] **Metadata Propagation**: If changing versions, ensure `pyproject.toml` and `eqsp/__init__.py` remain in lockstep.

## 3. Stylistic & Tonal Integrity
- [ ] **Passive Voice Audit**: Eliminate passive constructions (e.g., "is caught", "are updated") in favor of active, direct verbs ("catch", "update").
- [ ] **Conciseness Review**: Remove wordiness and clarify technical explanations by shortening phrases and removing parenthetical fluff.
- [ ] **Weasel Word Purge**: Systematically remove "clearly," "basically," "previously," and "automatically" unless they serve a critical, non-rhetorical technical purpose.

## 4. Documentation & Repository Record
- [ ] **Maintenance Guide (Volume 2)**: Ensure any changes to the "Defense in Depth" strategy are reflected in the guide's conceptual overview.
- [ ] **Testing Guide**: Ensure `doc/maintainer/testing_details.md` reflects any changes to the execution environment or linter policy.
- [ ] **Release Notes & CHANGELOG**:
    - Update the version-specific `doc/maintainer/release_notes_*.md`.
    - Ensure `CHANGELOG.md` reflects all notable changes for the target version and current date.
