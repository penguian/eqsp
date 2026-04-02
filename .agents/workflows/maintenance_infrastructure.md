---
description: Verification steps to run after modifying maintenance code or infrastructure.
---

When modifying scripts in `scripts/`, `doc/ci_scripts/`, or project-wide documentation (Maintenance/User Guides), you MUST perform the following checks before concluding the session.

// turbo-all
1. **Initialize and Verify Local Hooks**: Run `pre-commit install` followed by `pre-commit run --all-files` to catch linting, formatting, and typo regressions.
2. **Execute Unified Verification**: Run `python3 validation/verify_all.py` (ideally across target virtual environments like `.venv` and `.venv_sys`) to ensure the full test suite and documentation builds pass with a "Zero-Warning Policy."
3. **Tone and Directness Audit**: Review all NEW or MODIFIED documentation for:
    - **Passive Voice**: (e.g., "is caught" \u2192 "hooks catch").
    - **Wordiness**: (e.g., "resolving ... errors that previously caused" \u2192 "resolving ... build errors").
    - **Weasels**: Remove "clearly," "basically," "previously," and "automatically."
4. **Synchronize Records**:
    - Update the **Maintenance Guide** (`doc/ci_scriptsenance_guide.md`) if the "Defense in Depth" strategy or tool inventory changed.
    - Update the version-specific **Release Notes** (e.g., `doc/internal/release_notes_0_99_7.md`).
    - Synchronize the **Changelog** (`CHANGELOG.md`) with the new version and date.
5. **Release Parity**: Confirm version parity between `pyproject.toml` and `eqsp/__init__.py`.
