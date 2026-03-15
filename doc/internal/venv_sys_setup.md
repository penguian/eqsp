# System-Integrated Environment (venv_sys)

This document details the configuration of the `venv_sys` environment, which is used for project maintenance, automated testing of 3D features, and high-fidelity research reproduction.

## Rationale: Why venv_sys?

Heavy mathematical and visualization libraries like **Mayavi**, **VTK**, and **PyQt5** can be difficult to compile from source via `pip`. Using a system-integrated environment allows PyEQSP to:

1.  **Avoid Build Failures**: Reuses pre-compiled, tested binaries provided by the OS package manager (e.g., `apt`).
2.  **Ensure Stability**: Leverages system-specific optimizations for OpenGL and GPU acceleration.
3.  **Unified Testing**: Provides a stable baseline for `pytest` runs involving Mayavi mocks and 3D rendering.

## Creation Procedure

The `venv_sys` environment must be created with the `--system-site-packages` flag.

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

> [!NOTE]
> This specific calibration was validated on **Kubuntu Linux 25.10**. Other distributions may require `offscreen` backends for CI or different `QT_API` targets.

## Developer Installation

Once activated, install PyEQSP in editable mode to ensure your changes are reflected in both the venv and the system-linked packages:

```bash
pip install -e ".[dev]"
```

## Jupyter Notebook Integration

To use 3D features in Jupyter, you must register `venv_sys` as a kernel:

```bash
pip install ipykernel ipyevents
python3 -m ipykernel install --user --name=venv_sys --display-name "Python (venv_sys)"
```

## Troubleshooting

### 3D Rendering Fails
If Mayavi fails to open a window:
1. Verify `echo $DISPLAY` is set.
2. Check if `QT_QPA_PLATFORM` matches your display server (X11 vs. Wayland).
3. Try running `python3 tests/src/inspect_visualizations.py` to check for specific VTK error messages.
