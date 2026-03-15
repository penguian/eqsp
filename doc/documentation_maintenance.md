# Documentation Maintenance Guide

PyEQSP uses **Sphinx** and **ReadTheDocs (RTD)** to deliver both the User and Maintenance guides in a unified book series format.

## Technology Stack

- **Sphinx**: The core documentation generator.
- **MyST-Parser**: Enables writing documentation in MarkDown (.md) while retaining Sphinx features (tables, cross-references).
- **Sphinx RTD Theme**: Provides the responsive, premium look and feel.

## Local Build Workflow

To build and preview the documentation locally:

1. **Install Dependencies**:
   ```bash
   pip install .[docs]
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

## Guide Lifecycle

- **Volume 1 (User)**: Should be updated whenever a new public feature or visualization method is added.
- **Volume 2 (Maintenance)**: Should be updated when internal architecture changes (e.g., transitioning from Mayavi to a new 3D engine) or when benchmarks are re-run.
