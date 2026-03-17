# Installation Guide

**PyEQSP: Python Equal Area Sphere Partitioning Library**

PyEQSP requires Python 3.11 or later. We recommend using a virtual environment to manage dependencies locally.

## Basic Installation

The easiest way to install **PyEQSP** is via `pip` from PyPI:

```bash
pip install pyeqsp
```

To install with development and testing dependencies (recommended for reproducing research):

```bash
pip install "pyeqsp[dev]"
```

## Creating a Virtual Environment

Using a virtual environment prevents version conflicts between your scientific projects.

```bash
# Create a hidden environment directory
python3 -m venv .venvs/.venv

# Activate it
source .venvs/.venv/bin/activate

# Install PyEQSP in the environment
pip install pyeqsp
```

## 3D Plotting Requirements

While 2D illustrations work with standard Matplotlib, **3D interactive visualizations** require **Mayavi**.

> [!IMPORTANT]
> Mayavi and its underlying engine (VTK) can be complex to install from source on some systems. If `pip install mayavi` fails, please refer to the [System-Integrated Environment (venv_sys)](internal/venv_sys_setup.md) section in the Maintenance Guide for a reliable alternative using system-provided packages.

## Migration from MATLAB

If you are a user of the original MATLAB EQ Sphere Partitioning Toolbox, please check the [User Migration Guide](user_migration_guide.md) for function mappings and architectural differences.
