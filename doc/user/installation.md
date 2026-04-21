# Appendix B: Installation & Requirements

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

(venv-sys-setup)=
## 3D Plotting & System-Integrated Setup (venv_sys)

While 2D illustrations work with standard Matplotlib, **3D interactive visualizations** require **Mayavi**. Heavy mathematical and visualization libraries like **Mayavi**, **VTK**, and **PyQt5** can be difficult to compile from source via `pip`.

For these features, we recommend the `venv_sys` approach, which leverages pre-compiled binaries provided by your OS package manager (e.g., `apt`).

### 1. Install System Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3-venv python3-mayavi python3-numpy python3-scipy python3-matplotlib
```

### 2. Create and Activate

```bash
python3 -m venv --system-site-packages .venvs/.venv_sys
source .venvs/.venv_sys/bin/activate
```

### 3. Display Calibration (Kubuntu/Linux)

For environments using KDE/Plasma or specific Qt versions, you may need to export these variables to ensure Mayavi initializes correctly:

```bash
export QT_API="pyqt5"
export QT_QPA_PLATFORM="xcb"
```

:::{important}
This specific calibration was validated on **Kubuntu Linux 25.10**. Other distributions may require `offscreen` backends for CI or different `QT_API` targets.
:::

## Jupyter Notebook Integration

To use 3D features in Jupyter, you must register `venv_sys` as a kernel:

```bash
pip install ipykernel ipyevents
python3 -m ipykernel install --user --name=venv_sys --display-name "Python (venv_sys)"
```

## Troubleshooting

If Mayavi fails to open a window:
1. Verify `echo $DISPLAY` is set.
2. Check if `QT_QPA_PLATFORM` matches your display server (X11 vs. Wayland).
3. Try running `python3 tests/src/inspect_visualizations.py` to check for specific VTK error messages.
