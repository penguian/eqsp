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

Two virtual environment strategies are supported:

| Strategy | Command | Typical Use Case |
|:---|:---|:---|
| **`VENV`** | `python3 -m venv VENV` | Standard isolated environment; installs dependencies via PyPI wheels |
| **`VENV_SYS`** | `python3 -m venv --system-site-packages VENV_SYS` | System-integrated environment; uses system packages (e.g. system VTK) |

To create and activate a standard virtual environment:

```bash
python3 -m venv VENV
source VENV/bin/activate
```

If you need to use system-installed packages (such as system VTK), create the environment with the `--system-site-packages` flag instead:

```bash
python3 -m venv --system-site-packages VENV_SYS
source VENV_SYS/bin/activate
```

For further details and Jupyter integration, see the [Installation Guide](doc/user/installation.md).

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

To install with PyVista support (standard `VENV` path):
```bash
pip install ".[pyvista]"
```

#### PyVista with System VTK (VENV_SYS)

If you need PyVista 3D visualizations and want to use the system-installed VTK
(either to avoid large PyPI downloads on x86-64, or because you are on ARM64
where the PyPI VTK wheel is incompatible — see below), install the system VTK
package first:

```bash
# Fedora / RHEL:
sudo dnf install python3-vtk

# Ubuntu / Debian:
sudo apt install python3-vtk9
```

Then create a system-site-packages virtual environment and install PyVista
without its bundled VTK:

```bash
python3 -m venv --system-site-packages VENV_SYS
source VENV_SYS/bin/activate
pip install --no-deps pyvista
pip install pyvista-validation scooby pillow pooch cyclopts
pip install -e ".[dev]"
```

> [!NOTE]
> Because `--no-deps` is used to prevent pip from downloading the incompatible PyPI `vtk` wheel on ARM64, PyVista's non-VTK Python dependencies (`pyvista-validation`, `scooby`, `pillow`, `pooch`, `cyclopts`) are installed explicitly.

##### Note on ARM64 / Fedora Asahi Remix (Apple Silicon)

On ARM64 systems running Fedora Asahi Remix (e.g. Apple M1/M2), the `VENV_SYS`
path is **required** rather than optional. The PyPI `vtk` wheel is built for
4 KB memory page alignment (x86-64 default) and segfaults on the Apple M-series
kernel, which requires 16 KB page alignment. The Fedora `python3-vtk` RPM is
built correctly for 16 KB page alignment on ARM64.

PyVista is the officially supported 3D backend on ARM64 / Asahi Linux.

##### Confirmed Working Configurations

| Architecture | OS | Python | VTK | Venv strategy |
|:---|:---|:---|:---|:---|
| x86-64 | Fedora / Ubuntu | 3.11+ | PyPI wheel | `VENV` |
| x86-64 | Fedora / Ubuntu | 3.11+ | System RPM/deb | `VENV_SYS` |
| aarch64 | Fedora Asahi Remix 45 | 3.14.x (system) | System RPM | `VENV_SYS` (required) |

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
