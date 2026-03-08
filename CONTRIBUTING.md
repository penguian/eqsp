# Contributing to EQSP

Thank you for helping us refine the Recursive Zonal Equal Area Sphere Partitioning (EQSP) library! This project is currently in Beta testing, and your feedback is invaluable.

## How to Provide Feedback

### Reporting Bugs
If you find a bug, please [open a new issue](https://github.com/penguian/eqsp/issues/new/choose) using the **Bug Report** template. Please include:
- Steps to reproduce the issue.
- Your environment details (OS, Python version, etc.).
- Any relevant plots or screenshots.

### Suggesting Enhancements
We welcome ideas for new features or improvements to the partitioning algorithms or visualizations. Please use the **Feature Request** template when [opening an issue](https://github.com/penguian/eqsp/issues/new/choose).

## Technical Contributions

If you would like to contribute code fixes or improvements, please follow the forking workflow:

1. **Fork the Repository**: Create your own copy of the `penguian/eqsp` repository on GitHub.
2. **Clone and Setup**:
   We recommend using a **virtual environment** to avoid dependency conflicts:
   ```bash
   # Create and activate a virtual environment
   python3 -m venv .venvs/.venv
   source .venvs/.venv/bin/activate  # On Windows use `.venvs\.venv\Scripts\activate`

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/eqsp.git
   cd eqsp

   # Install in "editable" mode with development tools
   pip install -e ".[dev]"
   ```
   > **What is "editable" mode (`-e`)?** This creates a link between your local code and your Python environment. Any changes you make to the code in this folder will take effect immediately without needing to reinstall. The `[dev]` extra installs linting and testing tools (`ruff`, `pylint`, `pytest`, `coverage`).

3. **Troubleshooting Installation**:
   If the `pip install` command fails, please:
   - Ensure your `pip` is up to date (`pip install --upgrade pip`).
   - Check the detailed environment setup guide in [INSTALL.md](INSTALL.md).
   - If you are still stuck, [open an issue](https://github.com/penguian/eqsp/issues/new) and we will help you!

4. **Draft your Changes**: We recommend creating a new branch for your fix or feature.
5. **Coding Standards**:
   To maintain high code quality, we require:
   - **Linting**: Code must satisfy `ruff` and `pylint`.
   - **Docstrings**: Use the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
   - **Code Coverage**: We maintain a strict **100% test coverage** policy.
       - All new features must include comprehensive tests.
       - `# pragma: no cover` should be used sparingly and only for truly unreachable code, platform-specific blocks, or debugging branches.
       - Every pragma usage must be justified and reviewed.
   - **Minimalism**: Keep changes focused and brief.

6. **Run Tests & Linters**: Ensure that your changes satisfy all quality checks and do not break existing functionality. We provide a single verification script that runs Ruff, Pylint, and Pytest with coverage:
   ```bash
   python3 verify_all.py
   ```
   For more details on the relationship between `doctests` and `pytest`, see the [Testing Guide](doc/testing_guide.md).

7. **Submit a Pull Request**: Once your changes are ready, submit a Pull Request (PR) from your fork to our `main` branch.

### Syncing your Fork
If the main repository has been updated, you can sync your fork using:
```bash
git fetch upstream
git merge upstream/main
```

## Community Standards
Please be respectful and constructive in your feedback. Our goal is to build a robust and mathematically sound library for the sphere partitioning community.
