# PyEQSP: Python Equal Area Sphere Partitioning Library

**Release 1.0b3** (2026-08-30): Copyright 2026 Paul Leopardi

# Installation

This guide covers the installation of **PyEQSP**,
the Python Equal Area Sphere Partitioning Library.

## Prerequisites

-   Python 3.11 or later
-   `pip` (Python package installer)

The package depends on:
-   `numpy`
-   `scipy`
-   `matplotlib`
-   `pyvista` (optional)

Installing **PyEQSP** via `pip` automatically installs these dependencies.

### Python Virtual Environments

We recommend installing and using **PyEQSP** within a Python
virtual environment. A virtual environment isolates project
dependencies from your system Python, preventing version conflicts.
You can locate the environment anywhere accessible in the
file system; it doesn't need to be in the project directory.

To create and activate a virtual environment:

```bash
python3 -m venv VENV
source VENV/bin/activate
```

If you need to use system-installed packages such as PyVista, create
the environment with the `--system-site-packages` flag instead:

```bash
python3 -m venv --system-site-packages VENV_SYS
source VENV_SYS/bin/activate
```

For details on configuring Qt environment variables and using virtual
environments with Jupyter, see the [Installation Guide](doc/user/installation.md).

## 1. Installation from Source (Git Clone)

If you want to use the latest development version or change the
code, install from the source repository.

### Step 1: Clone the repository

```bash
git clone https://github.com/penguian/pyeqsp.git
cd pyeqsp
```

> **Naming Distinction**: While the project name is **PyEQSP** and you install it via `pip install pyeqsp`, you import the package as **eqsp**.

### Step 2: Install the package

Activate your virtual environment before running
these commands.

To install the package:

```bash
pip install .
```

To install with PyVista support:
```bash
pip install ".[pyvista]"
```

### Step 3: Install in Editable Mode (For Developers)

If you intend to change the code and want changes to take effect
immediately without reinstalling:

```bash
pip install -e .
```

To also install development tools (`ruff`, `pylint`, `pytest`,
`coverage`):

```bash
pip install -e ".[dev]"
pre-commit install
```

## 2. Installation via Pip

Because the current PyPI releases of **PyEQSP** are beta pre-releases, install with the `--pre` flag (this will not be necessary once a stable 1.0.0+ release is published):

```bash
pip install --pre pyeqsp
```

To upgrade an existing installation:

```bash
pip install --upgrade --pre pyeqsp
```

## Verification

To verify the installation, start a Python shell and try
importing the package:

```python
import eqsp
print(eqsp.__version__)
```

You can also run the illustration verification script if you have
the source code:

```bash
python tests/src/inspect_illustrations.py
```

## Building Documentation

To build the HTML documentation locally, ensure you have the
`docs` dependencies installed:

```bash
pip install ".[docs]"
cd doc
make html
```

The rendered documentation will be available in
`doc/_build/html/index.html`.

## Uninstalling

To remove the package:

```bash
pip uninstall pyeqsp
```
