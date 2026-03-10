# Book Series Outline: Equal Area Sphere Partitioning in Python

This document provides a structured outline for the planned two-volume documentation series on ReadTheDocs.

---

## Volume 1: The User Guide
**Purpose**: An accessible entry point for researchers and scientists using the **PyEQSP** library.

### Introduction & Setup
- **Background**: Overview of Equal Area Sphere Partitioning (EQ) and its applications.
- **Environment Management**:
  - Python version requirements (3.11+).
  - Virtual environment strategies (`venv` vs `venv_sys`).
  - Handling system-integrated dependencies (Mayavi, VTK, PyQt5).
- **Quick Start**: Generating your first point set in 60 seconds.

### The Applicability of PyEQSP: Use Cases
*Based on: "The applicability of equal area partitions of the unit sphere", JAS 1(2), 2024.*
- **History**: Evolution from the 2005 Matlab Toolbox to the PyEQSP Python library.
- **Cross-Disciplinary Applications**:
  - **Climate & Weather**: Spatio-temporal reconstructions and load balancing for the ECMWF Integrated Forecasting System (IFS).
  - **Biology & Medicine**: RNA folding simulations ($S^3$), 3D MRI sampling trajectories, and brain connectivity parcellation.
  - **Geophysics**: Sampling tectonic patterns on the moon and locating geomagnetic virtual observatories (GVOs).
  - **Engineering & Physics**: Micromechanics of carbon fibre and correlation energy in Fermi gases.
  - **Robotics & Visualization**: Orientation estimation via hyperhemispheres and Google Earth spatial data mapping (R2G2).
- **Evaluating Performance**: Lessons from comparisons with $k$-means clustering, spiral points, and icosahedral partitions.

### Core Concepts & Geometries
- **Coordinate Systems**: Spherical vs Euclidean conventions in **PyEQSP**.
- **The EQ Algorithm**: Intuition behind recursive zonal partitioning.
- **Supported Manifolds**: Operations on $S^2$, $S^3$, and higher-dimensional spheres $S^d$.

### Practical Usage
- **Partitioning**: Working with `eq_regions` and `eq_point_set`.
- **Property Analysis**: Measuring minimum distance, packing density, and Riesz energy.
- **Histograms & Binning**: Vectorized point-in-region lookups for bulk data on $S^2$.

### Visualization & Illustration
- **2D Schematics**: Matplotlib-based algorithm diagrams and projections.
- **3D Interactive Graphics**: Mastering Mayavi for rendering partitions and point sets on $S^2$ and $S^3$.
- **Jupyter Integration**: Using `ipyevents` for interactive research in notebooks.

### Reproducing Research
- **PhD Thesis Examples**: A guide to the `examples/phd-thesis/src` scripts.
- **Calibration**: Matching thesis aesthetics (LaTeX titles, axis formatting, markers).

---

## Volume 2: The Maintenance Guide
**Purpose**: A deep technical manual for developers, contributors, and those extending the library.

### Architecture & Design
- **Module Breakdown**: Internal (`_private`) vs Public API logic.
- **The Composite Strategy**: Integrating Matplotlib LaTeX rendering with Mayavi/VTK scenes.
- **Extensibility**: Design patterns for adding new manifolds or property metrics.

### Algorithmic Optimizations (The Python Shift)
- **Vectorization Patterns**: Replacing Matlab loops with NumPy/masked-array operations.
- **Spatial Indexing**: Using KDTrees for $O(N \log N)$ distance property calculations.
- **Numerical Stability**: Root-finding optimizations for cap radius calculation.

### Performance & Scaling
- **The $O(N^{0.6})$ Proof**: High-fidelity benchmarking of `eq_regions` across dimensions.
- **Symmetry-Aware Energy**: Implementation of block-based summation for Riesz energy.
- **Parallelization**: Multi-process strategies for computationally heavy figures (e.g., Fig 3.7).
- **Suggestions for Improvement**:
    - **Cython/Numba Integration**: Transitioning recursive collar loops to compiled C/LLVM for near-native speed.
    - **Persistent Caching**: Implementing disk-based or memory-resident caches for regional properties across sequential research runs.
    - **Advanced Vectorization**: Researching "ragged array" structures (e.g., via `awkward-array`) to eliminate loops over varying partition sizes $N$.

### Quality Assurance & Verification
- **Static Analysis**: Achieving and maintaining 10.00/10 Pylint and zero-Ruff compliance.
- **Test Infrastructure**: Managing the `pytest` suite and doctest parity.
- **Coverage Strategy**: Analyzing results from the `run_coverage.py` tool.

### Maintaining and Generating the Documentation
- **Docstring Standards**: Leveraging Google-style docstrings for automated API extraction.
- **The `doc/*.md` Library**: Managing the collection of background and technical guides.
- **Sphinx build system**: Orchestrating the `make html` workflow and `index.rst` management.
- **ReadTheDocs (RTD) Strategy**: Webhook integration, versioning (Alpha/Beta/Stable), and public hosting.
- **Guide Lifecycle**: Procedures for periodic updates to the User and Maintenance Guides to reflect library evolution.

### Release & Lifecycle
- **The PyPI Pipeline**: Build systems, metadata, and `twine` uploading.
- **Test-Driven Versions**: Verification workflows on TestPyPI.
- **Roadmap Governance**: Managing the transition between Alpha, Beta, and Production branches.
