# Pull Request Checklist: PyEQSP Quality Assurance

Use this checklist to verify the technical and procedural integrity of a Pull Request before submission. This captures common infrastructure and documentation "gotchas" identified during the 0.99.7 release cycle.

## 1. Automated Verification
- [ ] Run `python3 verify_all.py`.
- [ ] Ensure **0 errors** in Ruff and **10.00/10** in Pylint.
- [ ] Confirm Sphinx documentation builds with **0 warnings** (`make -C doc html SPHINXOPTS="-W"`).
- [ ] Confirm all **Doctests** pass (`make -C doc doctest`).
- [ ] Run `validation/quality_check.py` to catch terminology errors (e.g. "2rd", "3rd-sphere") and positional-only argument violations in docs.

## 2. Infrastructure & Configuration
- [ ] **Ruff Format**: Verify `ruff.toml` uses the **flat configuration** (no `[lint]` section) for legacy environment compatibility.
- [ ] **Coverage Scope**: If new directories were added (e.g., `release/`), ensure they are included in `tests/run_coverage.py` `--source`.
- [ ] **Credential Logic**: If modifying cloud/PyPI automation, ensure `TWINE_PASSWORD` is used for authentication checks (not `TWINE_TOKEN`).

## 3. Code Quality & Formatting
- [ ] **Effective Pragmas**: Ensure `# pragma: no cover` is attached to a statement (e.g., `if __name__ == "__main__": # pragma: no cover`), not isolated on its own line.
- [ ] **Australian -ize**: Run `validation/quality_check.py` to ensure Oxford spelling (Standardization, Analyze) is used project-wide.

## 4. Documentation Consistency
- [ ] **Reference Parity**: Ensure any new references in `doc/references_vol*.md` are synchronized with `AUTHORS.md`.
- [ ] **Version Alignment**: If the change affects library core, ensure `pyproject.toml`, `README.md`, and `INSTALL.md` reflect the target version.
- [ ] **Table of Contents**: If new doc files were added, ensure they are indexed in `doc/index.rst`.
