# Python Environments and Setup Guide

This document provides detailed instructions on creating and
managing Python virtual environments for **PyEQSP**.

## Overview

A Python virtual environment is an isolated directory that
contains its own Python interpreter and installed packages,
separate from your system Python. This prevents version conflicts
between projects and ensures a reproducible setup.

Python provides the `venv` module in the standard library for
creating virtual environments. There are two main approaches:

-   **Self-contained (`venv`)**: All dependencies are installed
    fresh into the environment. This is the simplest option and
    is suitable for most users.
-   **System-integrated (`venv_sys`)**: The environment can also
    access packages installed via your OS package manager (e.g.,
    `apt`). This is useful when you want to use system-provided
    builds of heavy packages like Mayavi, VTK, or PyQt5, which
    can be difficult to compile from source.

The recommended convention is to store your environments in a hidden subdirectory within the project root:

-   **`.venvs/.venv`**: For the standard development environment.
-   **`.venvs/.venv_sys`**: For the system-integrated environment (Mayavi).

Using a hidden `.venvs/` directory keeps the project root clean while ensuring that automated scripts and IDEs can easily discover your environments.

### Creating a Self-Contained Environment

```bash
python3 -m venv .venvs/.venv
source .venvs/.venv/bin/activate
pip install -e ".[dev]"
```

This installs **PyEQSP** and all its dependencies entirely within
the environment. System-installed Python packages are not
visible.

### Creating a System-Integrated Environment

```bash
python3 -m venv --system-site-packages .venvs/.venv_sys
source .venvs/.venv_sys/bin/activate
pip install -e ".[dev]"
```

The `--system-site-packages` flag allows the environment to
"see" and import packages installed in
`/usr/lib/python3/dist-packages`. Packages installed via `pip`
within the environment take priority over system versions.

## System-Integrated Environment Details

### Why Use This Method?

-   **Avoids Build Issues**: System packages are pre-compiled
    and tested for your OS, avoiding complex compilation errors
    often seen when pip-installing `mayavi` or `vtk`.
-   **Saves Space**: Reuses large libraries installed via `apt`
    instead of duplicating them.
-   **Isolation**: Still allows you to install other Python
    packages effectively isolated from the system Python, and
    to install **PyEQSP** in editable mode.

### Prerequisites

Ensure you have the necessary system packages installed. On
Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-venv python3-mayavi python3-numpy python3-scipy python3-matplotlib
```

### Create the Environment

Create a virtual environment with the `--system-site-packages`
flag. This flag is the key: it allows the virtual environment
to "see" and import packages installed in
`/usr/lib/python3/dist-packages`.

```bash
python3 -m venv --system-site-packages .venvs/.venv_sys
```

### Activate and Configure

Activate the environment:

```bash
source .venvs/.venv_sys/bin/activate
export QT_API="pyqt5"
export QT_QPA_PLATFORM="xcb"
```

> **Note:** The `QT_API` and `QT_QPA_PLATFORM` exports are
> often necessary on modern Linux systems to ensure Mayavi uses
> the correct Qt bindings and display platform. This specific
> configuration was tested on **Kubuntu Linux 25.10**. Other
> environments may require different values or additional
> variables altogether.

Now, install **PyEQSP**. For developers, we recommend "editable"
mode so changes to source code are reflected immediately:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

> **What does `".[dev]"` install?** In addition to **PyEQSP**
> itself, this installs the development tools defined in
> `pyproject.toml`: `ruff`, `pylint`, `pytest`, and `coverage`.
> These are the same tools used by CI and by `verify_all.py`.

> **How does this interact with `venv_sys`?** When using a
> `--system-site-packages` virtual environment, pip installs
> the dev tools into the venv's local `site-packages` directory,
> which takes priority over any system-installed versions of the
> same packages. System packages like `numpy`, `scipy`,
> `matplotlib`, and `mayavi` remain visible through the
> `--system-site-packages` flag.

### Verification

Verify that you can import both the system package (Mayavi) and
your local package (EqSP):

```bash
python3 -c "import mayavi.mlab; print('Mayavi OK')"
python3 -c "import eqsp; print('EqSP OK')"
```

## Using with Jupyter Notebook

To use a virtual environment within Jupyter Notebook or
JupyterLab, you need to register it as a kernel.

### Install ipykernel

Ensure `ipykernel` is installed. It might be available from the
system, but installing it in the venv ensures compatibility.

```bash
source .venvs/.venv_sys/bin/activate
pip install ipykernel ipyevents
```

> **Note:** `ipyevents` is required for Mayavi's interactive
> features in Jupyter notebooks.

### Register the Kernel

Run the following command to make this environment available in
Jupyter:

```bash
python3 -m ipykernel install --user --name=venv_sys --display-name "Python (venv_sys)"
```

-   `--name=venv_sys`: The internal machine-readable name.
-   `--display-name "Python (venv_sys)"`: The name you will see
    in the Jupyter menu.

### Select the Kernel

Open Jupyter Notebook or Lab:

1.  Open your `.ipynb` file.
2.  Go to the **Kernel** menu -> **Change Kernel**.
3.  Select **Python (venv_sys)**.

Now, your notebook can import `mayavi` (from system) and `eqsp`
(from your local editable install).

## Known Issues

### 3D Visualization in Jupyter

While `test_illustrations` (Matplotlib 2D) typically works in
Jupyter, `test_visualizations` (Mayavi 3D) may fail to display
plots in some environments (e.g., Kubuntu with QT env vars set),
even when command-line execution works.

**Workaround:**
If 3D plots do not appear in Jupyter:
1.  Run the verify scripts from the terminal, e.g.
    `python3 tests/verify/verify_illustrations.py` or
    `python3 tests/verify/verify_mayavi_extensions_color.py`.
2.  Or check for specific browser/extension console errors
    related to `ipyevents` or WebGL.
