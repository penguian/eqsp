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
In the commands below, `.venvs/.venv` is the project's conventional path
(see `INSTALL.md` in the repository root for background on virtual environments;
you may use any path that suits your setup).

```bash
# Create a hidden environment directory
python3 -m venv .venvs/.venv

# Activate it
source .venvs/.venv/bin/activate

# Install PyEQSP in the environment
pip install --pre pyeqsp
```

> [!NOTE]
> `.venvs/.venv` is the project's conventional virtual environment path.
> Replace it with your preferred location if you are using a different layout.


(venv-sys-setup)=
## 3D Plotting & Visualizations Setup

While 2D illustrations work with standard Matplotlib, **3D interactive visualizations** require **PyVista**.

### 1. Install PyVista

To install with PyVista support in a standard `VENV` environment:

```bash
pip install --pre "pyeqsp[pyvista]"
```

#### Using System VTK (`VENV_SYS` Path)

If you are using a `VENV_SYS` environment (required on ARM64 / Fedora Asahi Remix; optional on x86-64), install the system `python3-vtk` package first (see `INSTALL.md` in the repository root), then install PyVista and its Python dependencies without bundled VTK:

```bash
pip install --no-deps pyvista
pip install pyvista-validation scooby pillow pooch cyclopts
```

This uses the system VTK rather than downloading the PyPI wheel.

### 2. Display Calibration & Off-Screen Rendering

PyVista supports both interactive GUI windows and headless off-screen rendering for CI
environments or Jupyter notebooks.

Off-screen rendering is controlled in Python by setting:

```python
import pyvista as pv
pv.OFF_SCREEN = True
```

> [!NOTE]
> `eqsp.visualizations` does not set `pv.OFF_SCREEN` automatically; set it in your
> script (e.g. `pv.OFF_SCREEN = True`) before calling 3D functions. The helper script
> `tests/src/inspect_visualizations.py` also honors `PYVISTA_OFF_SCREEN` by setting
> `pv.OFF_SCREEN` before running.

## Jupyter Notebook Integration

PyVista integrates with Jupyter Notebooks for interactive 3D rendering:

```bash
pip install trame ipywidgets
```

## Troubleshooting

If PyVista fails to open a window:
1. Verify `echo $DISPLAY` is set, or enable off-screen mode via `import pyvista as pv; pv.OFF_SCREEN = True` in your script.
2. Try running `python3 tests/src/inspect_visualizations.py` to verify PyVista rendering.
3. **ARM64 segfault on PyVista import**: The PyPI `vtk` wheel is built for 4 KB page alignment and is incompatible with ARM64 / Fedora Asahi Remix (which requires 16 KB page alignment). Use the `VENV_SYS` install path described in `INSTALL.md`.
4. **Interactive window fails on Wayland**: Try setting `QT_QPA_PLATFORM=wayland` or `QT_QPA_PLATFORM=xcb` (XWayland fallback). Not needed for automated testing or headless scripts (`pv.OFF_SCREEN = True`).
