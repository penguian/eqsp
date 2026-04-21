# Documentation Maintenance Guide

PyEQSP uses **Sphinx** and **ReadTheDocs (RTD)** to deliver both the User and Maintenance guides in a unified book series format.

## Technology Stack

- **Sphinx**: The core documentation generator.
- **MyST-Parser**: Enables writing documentation in Markdown (.md) while retaining Sphinx features (tables, cross-references).
- **Sphinx RTD Theme**: Provides the responsive, premium look and feel.

## Local Build Workflow

To build and preview the documentation locally:

1. **Install Dependencies**:
   ```bash
   pip install ".[docs]"
   ```
2. **Execute Build**:
   ```bash
   cd doc && make html
   ```
3. **Review**: Open `doc/_build/html/index.html` in your browser.

## Docstring Standards

To ensure automated API extraction works correctly, follow the **Google Style Python Docstrings**.
- Always document parameters, return values, and types.
- Include a "Notes" section for mathematical formulas ($S^2$, $O(N)$).

## ReadTheDocs (RTD) Strategy

The project is configured with GitHub webhooks.
- **Automation**: Every push to `main` triggers a build of the "Latest" version.
- **Versioning**: Branch tags (e.g., `release_0_99_4`) can be specifically enabled on the RTD dashboard to preserve documentation for older releases.

## SourceForge Mirroring

While **ReadTheDocs** is our primary documentation host, we maintain a secondary mirror on **SourceForge** to ensure high-availability and persistence for research citations.

## Automated Quality Safeguards

To prevent documentation drift and common technical errors, PyEQSP includes a suite of automated checks integrated into `verify_all.py`:

*   **Link & Citation Check** (`validation/check_links.py`): Validates all internal anchors, cross-file `{ref}` targets, and citation links.
*   **Function Existence Check** (`validation/quality_check.py`): Scans all guides and `README.md` to ensure any code snippet referencing `eqsp.<func>` actually exists in the current package version. This prevents "ghost" references like the legacy `plot_regions_2d`.
*   **Coordinate Convention Check**: Validates that array shape descriptions in documentation follow the **column-major (dim+1, N)** convention rather than the common Row-major error.
*   **Matplotlib Initialization Check**: Enforces that `matplotlib.use('Agg')` is called before any `pyplot` imports in examples to ensure headless environment compatibility.
*   **Configuration Type Check**: Validates that `doc/conf.py` variables use the correct data types expected by Sphinx extensions.
*   **Orthography Check** (`validation/quality_check.py`): Enforces the project's linguistic standard (**Australian -ize English**), flagging non-compliant spellings (e.g., "-ise" suffixes) to ensure global academic consistency.
*   **Readability Check** (`validation/compute_readability.py`): Calculates Flesch-Kincaid and Gunning-Fog scores to monitor prose complexity and prevent academic drift.

## Optional Linting: Vale

The project is configured for **Vale**, a prose linter that checks for style, clarity, and readability.

- **Configuration**: Uses `.vale.ini` and a custom `.vale/styles` directory.
- **Rulesets**: Integrates `proselint`, `write-good`, and `Readability`.
- **Manual Execution**: To run a manual prose audit, use:
  ```bash
  vale README.md doc/*.md
  ```
- **Optional Status**: Vale is **not** currently part of the mandatory `verify_all.py` suite. If you do not have Vale installed, the standard verification process will still pass.

To ensure stability across CI/CD and diverse local environments, these scripts are designed with **architectural isolation** (using independent `sys.path` setup) and **headless environment support** (automatically forcing `matplotlib.use('Agg')` if executed on a machine without a display).

## Guide Lifecycle

- **Volume 1 (User)**: Should be updated whenever a new public feature or visualization method is added.
- **Volume 2 (Maintenance)**: Should be updated when internal architecture changes (e.g., transitioning from Mayavi to a new 3D engine) or when benchmarks are re-run.

### Automated Sourceforge Upload
The project includes a utility script to automate the Sphinx build and print the exact transfer command for the Sourceforge mirror:
```bash
python release/upload_sourceforge.py
```

### Manual Deployment
If you need to manually push a documentation update, follow these steps:

1. **Build HTML locally**:
   ```bash
   cd doc && make html
   ```
2. **Transfer via SCP**:
   (Replace `USER` with your SourceForge username)
   ```bash
   scp -r doc/_build/html/* USER@web.sourceforge.net:/home/project-web/pyeqsp/htdocs/
   ```

3. **Verify**: The documentation will be instantly live at `http://pyeqsp.sourceforge.io`.
