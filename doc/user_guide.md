# Volume 1: The User Guide

This guide enables researchers and scientists to set up **PyEQSP** and reproduce published research results.

## Introduction & Setup

PyEQSP requires Python 3.11 or later. We recommend using a virtual environment to manage dependencies like NumPy, SciPy, and Matplotlib.
- **Environment Management**: See the [Installation Guide](user/installation.md) for strategy details (e.g., `venv` vs `venv_sys`).

## Use Cases & Applicability

PyEQSP is the Python successor to the original MATLAB Equal Area Sphere Partitioning toolbox. Researchers use it in many fields:
- **Climate Science**: Spatio-temporal reconstructions.
- **Bio-Medicine**: RNA folding simulations and MRI sampling.
- **Physics**: Correlation energy in Fermi gases.

For a comprehensive review of applications, see the [Use Cases Guide](user/use_cases.md) and the *Journal of Applied Statistics* (JAS) 2024 paper: *"The applicability of equal area partitions of the unit sphere"*.

## Core Concepts & Geometries

The library implements recursive zonal partitioning (the EQ algorithm) for these manifolds:
- **Manifolds**: Operations on $S^2$ (sphere), $S^3$, and higher-dimensional spheres $S^d$.
- **Coordinate Systems**: Support for both Spherical and Euclidean conventions.

For more details on the underlying mathematics, see the [Core Concepts & Geometries](user/core_concepts.md) guide.

## Practical Usage

Generate partitions and analyze their properties using the core API:
- **Partitioning**: Creating regions and point sets via `eq_regions` and `eq_point_set`.
- **Property Analysis**: Measuring min-distance, packing density, and Riesz energy.

For step-by-step examples, see the [Practical Usage Guide](user/practical_usage.md).

## Visualization

PyEQSP supports both 2D and 3D visualizations:
- **2D Projections**: Matplotlib-based schematics.
- **3D Interactive Graphics**: High-fidelity rendering using Mayavi and VTK.

For detailed plotting options, see the [Visualization & Illustration Guide](user/visualization_guide.md).

## Reproducing Research

A primary goal of PyEQSP is the faithful reproduction of research results from the original PhD thesis [Leo07].
- **Thesis Examples**: A dedicated guide to the [Thesis Reproduction Scripts](user/phd-thesis-examples.md) is available. For technical setup details, see the [Reproduction Setup](user/reproduction_setup.md) guide in Volume 2.

## Executable Examples

Standalone Python scripts demonstrating these concepts are available in the [examples/user-guide/src/](https://github.com/penguian/pyeqsp/tree/main/examples/user-guide/src) directory:
- `example_quick_start.py`: Basic partitioning and property analysis.
- `example_visualize_2d.py` & `example_visualize_3d.py`: Plotting and interactive rendering.
- `example_symmetric_partitions.py`: Using the `even_collars` parameter.

## Migration from MATLAB

If you are transitioning from the original MATLAB toolbox, please refer to the [User Migration Guide](user/user_migration_guide.md) for a mapping of functions and major architectural changes.

For the full list of scientific works cited in this volume, see the [References](user/references_vol1.md) chapter.
