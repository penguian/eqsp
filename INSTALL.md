# Installation

This guide covers the installation of the Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox for Python.

## Prerequisites

-   Python 3.11 or later
-   `pip` (Python package installer)

The package depends on:
-   `numpy`
-   `scipy`
-   `matplotlib`
-   `mayavi` (optional)
-   `PyQt5` (optional)

These dependencies will be installed automatically when installing via `pip`.

## 1. Installation via Pip (Recommended)

If the package is available on PyPI or a package repository, you can install it directly:

```bash
pip install eqsp
```

To upgrade an existing installation:

```bash
pip install --upgrade eqsp
```

## 2. Installation from Source (Git Clone)

If you want to use the latest development version or modify the code, install from the source repository.

### Step 1: Clone the repository

```bash
git clone https://github.com/penguian/eqsp.git
cd eqsp
```
*(Replace the URL with the actual repository URL if different)*

### Step 2: Install the package

To install the package in the current environment:

```bash
pip install .
```

To install with Mayavi support:
```bash
pip install ".[mayavi]"
```

### Step 3: Install in Editable Mode (For Developers)

If you intend to modify the code and want changes to be reflected immediately without reinstalling:

```bash
pip install -e .
```

### Using System-Installed Mayavi (e.g., via apt on Ubuntu)

If you have Mayavi installed via your system package manager (e.g., `sudo apt install python3-mayavi`), you can use it with `eqsp` by creating a virtual environment that accesses system site packages.

For detailed instructions on setting up this environment (`venv_sys`) and using it with Jupyter Notebook, please refer to [doc/python_environments.md](doc/python_environments.md).

> **Note:** The `venv_sys` configuration and associated environment variables (like `QT_API` and `QT_QPA_PLATFORM`) described in this documentation were specifically tested on **Kubuntu Linux 25.10**. Other Linux distributions or versions may require different environment variable values or additional configuration.

Quick setup summary:
```bash
python3 -m venv --system-site-packages venv_sys
source venv_sys/bin/activate
pip install .
```

## Verification

To verify the installation, start a Python shell and try importing the package:

```python
import eqsp
print(eqsp.__version__)
```

You can also run the illustration verification script if you have the source code:

```bash
python tests/src/inspect_illustrations.py
```

## Building Documentation

To build the HTML documentation locally, ensure you have the `docs` dependencies installed:

```bash
pip install ".[docs]"
cd doc
make html
```

The rendered documentation will be available in `doc/_build/html/index.html`.

## Uninstalling

To remove the package:

```bash
pip uninstall eqsp
```
