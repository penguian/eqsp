# Installation

This guide covers the installation of the EQSP (Equal Area
Sphere Partitioning) library for Python.

## Prerequisites

-   Python 3.11 or later
-   `pip` (Python package installer)

The package depends on:
-   `numpy`
-   `scipy`
-   `matplotlib`
-   `mayavi` (optional)
-   `PyQt5` (optional)

These dependencies will be installed automatically when installing
via `pip`.

### Python Virtual Environments

It is recommended that you install and use `eqsp` within a Python
virtual environment. A virtual environment isolates project
dependencies from your system Python, preventing version conflicts
between projects. It can be located anywhere accessible in the
file system; it does not need to be in the project directory.

To create and activate a virtual environment:

```bash
python3 -m venv /path/to/your/venv
source /path/to/your/venv/bin/activate
```

If you need to use system-installed packages such as Mayavi, create
the environment with the `--system-site-packages` flag instead:

```bash
python3 -m venv --system-site-packages /path/to/your/venv_sys
source /path/to/your/venv_sys/bin/activate
```

For detailed guidance on choosing between these two approaches,
configuring Qt environment variables, and using virtual
environments with Jupyter, see
[doc/python_environments.md](doc/python_environments.md).

## 1. Installation via Pip (Recommended)

If the package is available on PyPI or a package repository, you
can install it directly:

```bash
pip install eqsp
```

> **Note:** The `eqsp` package is not yet published to PyPI. This
> command will work once a stable release is published. For now,
> install from source (see below).

To upgrade an existing installation:

```bash
pip install --upgrade eqsp
```

## 2. Installation from Source (Git Clone)

If you want to use the latest development version or modify the
code, install from the source repository.

### Step 1: Clone the repository

```bash
git clone https://github.com/penguian/eqsp.git
cd eqsp
```
*(Replace the URL with the actual repository URL if different.)*

### Step 2: Install the package

Ensure your virtual environment is activated before running
these commands.

To install the package:

```bash
pip install .
```

To install with Mayavi support:
```bash
pip install ".[mayavi]"
```

### Step 3: Install in Editable Mode (For Developers)

If you intend to modify the code and want changes to be reflected
immediately without reinstalling:

```bash
pip install -e .
```

To also install development tools (`ruff`, `pylint`, `pytest`,
`coverage`):

```bash
pip install -e ".[dev]"
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
pip uninstall eqsp
```
