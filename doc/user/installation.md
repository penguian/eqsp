# Appendix B: Installation & Requirements

**PyEQSP: Python Equal Area Sphere Partitioning Library**

PyEQSP requires Python 3.11 or later. We recommend using a virtual environment to manage dependencies locally.

## Basic Installation

The easiest way to install **PyEQSP** is via `pip` from PyPI. Because the current PyPI releases are beta pre-releases, you must specify the `--pre` flag:

```bash
pip install --pre pyeqsp
```

To install with development and testing dependencies (recommended for reproducing research):

```bash
pip install --pre "pyeqsp[dev]"
```

## Creating a Virtual Environment

Using a virtual environment prevents version conflicts between your scientific projects.

```bash
# Create a hidden environment directory
python3 -m venv .venvs/.venv

# Activate it
source .venvs/.venv/bin/activate

# Install PyEQSP in the environment
pip install --pre pyeqsp
```

(venv-sys-setup)=
## 3D Plotting & Visualizations Setup

While 2D illustrations work with standard Matplotlib, **3D interactive visualizations** require **PyVista**.

### 1. Install PyVista

```bash
pip install --pre "pyeqsp[pyvista]"
```

### 2. Display Calibration & Off-Screen Rendering

PyVista supports both interactive GUI windows and headless off-screen rendering for CI environments or Jupyter notebooks:

```bash
export PYVISTA_OFF_SCREEN=true
```

## Jupyter Notebook Integration

PyVista integrates with Jupyter Notebooks for interactive 3D rendering:

```bash
pip install trame ipywidgets
```

## Troubleshooting

If PyVista fails to open a window:
1. Verify `echo $DISPLAY` is set (or set `export PYVISTA_OFF_SCREEN=true` for headless environments).
2. Try running `python3 tests/src/inspect_visualizations.py` to verify PyVista rendering.
