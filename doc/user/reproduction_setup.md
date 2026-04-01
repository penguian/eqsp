# Reproduction Setup & Troubleshooting

This document provides the technical configuration needed to execute the high-fidelity research reproduction scripts found in `examples/phd-thesis/`.

## Environment Requirements

Reproduction scripts for 3D figures (e.g., Fig 3.1, Fig 4.10) require the **Mayavi** engine. We strongly recommend using the **`venv_sys`** environment for this work. See [venv_sys_setup.md](venv_sys_setup.md) for more.

### Backend Configuration

Rendering 3D Great Circle arcs and Voronoi cells on the sphere requires specific VTK/Mayavi backends. On Linux systems, calibrate your environment as follows:

```bash
export QT_API="pyqt5"
export QT_QPA_PLATFORM="xcb"
```

> [!WARNING]
> Without these exports, Mayavi may fail to initialize a window or crash with a `Segmentation Fault` when attempting to rasterize GREAT CIRCLE edges.

## Bitwise Reproducibility

PyEQSP aims for identical numerical results to the original thesis. But users should be aware of two potential sources of variance:

1.  **Hardware Rasterization**: 3D plots generated via Mayavi/VTK use GPU hardware. Minor variances (e.g., ~2 pixel differences) can occur between different GPUs or drivers due to non-deterministic anti-aliasing.
2.  **Floating-Point Drift**: While we use `numpy.longdouble` for critical recursions, extreme depths in high dimensions ($d > 8$) may show sub-microscopic differences across different CPU architectures.

## Headless Execution

To run reproduction scripts on a server or in CI without a display:

```bash
export HEADLESS=1
python3 fig_3_4_max_diam_s2.py
```

Numerical scripts (Matplotlib `Agg` backend) will save PNGs directly. 3D scripts (Mayavi) will attempt to use an offscreen buffer if `xvfb` is available.

For the full mapping of scripts to thesis figures, see the [Thesis Research Reproduction](phd-thesis-examples.md) guide in Volume 1.
