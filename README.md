# PyEQSP: Python Equal Area Sphere Partitioning Library

**Release 0.99.4** (2026-03-17): Copyright 2026 Paul Leopardi

PyEQSP is a Python library that implements the **Recursive Zonal
Equal Area (EQ) Sphere Partitioning** algorithm, originally
implemented as a Matlab toolbox by Paul Leopardi.

An EQ partition is a partition of Sᵈ (the unit sphere in
the (d+1)-dimensional Euclidean space ℝᵈ⁺¹) into
a finite number of regions of equal area. The area of each region
is defined using the Lebesgue measure inherited from
ℝᵈ⁺¹.

> [!IMPORTANT]
> **Naming Distinction**: The project and GitHub repository are named **PyEQSP** (or **pyeqsp** on PyPI), but the Python package is imported as **eqsp**.
>
> | **Repository / PyPI** | **Package Import** |
> | :--- | :--- |
> | `pyeqsp` | `import eqsp` |

## What is an EQ partition?

An **EQ partition** is a partition of Sᵈ (the unit sphere in
the (d+1)-dimensional Euclidean space ℝᵈ⁺¹) into
a finite number of regions of equal area. The area of each region
is defined using the Lebesgue measure inherited from
ℝᵈ⁺¹.

The **diameter** of a region is the supremum of the Euclidean
distance between any two points of the region. The regions of an
EQ partition have been proven to have small diameter, in the
sense that there exists a constant C(d) such that the maximum
diameter of the regions of an N-region EQ partition of Sᵈ is
bounded above by C(d)·N⁻¹/ᵈ.

## What is an EQ point set?

An **EQ point set** is the set of centre points of the regions
of an EQ partition. Each region is defined as a product of
intervals in spherical polar coordinates. The centre point of a
region is defined via the centre points of each interval, with
the exception of spherical caps and their descendants, where the
centre point is defined using the centre of the spherical cap.

## Applications

EQ partitions and point sets are useful in a range of
applications that require well-distributed points on a sphere,
including:

- Numerical integration (quadrature) on the sphere
- Sensor, satellite, or antenna placement
- Mesh generation for geophysical and climate models
- Monte Carlo sampling on spherical domains
- Computer graphics and rendering

## Installation

Requires **Python 3.11+**. It is recommended that you install
**PyEQSP** within a Python virtual environment. See
[INSTALL.md](INSTALL.md) for full instructions, including
environment setup and optional dependencies.

## Quick Start

### 1. Create EQ partitions

Create an array in Cartesian coordinates representing the
centre points of an EQ partition of Sᵈ into N regions:

```python
import eqsp

dim = 2
N = 100
points_x = eqsp.eq_point_set(dim, N)
# points_x.shape is (dim+1, N)
```

Create an array in spherical polar coordinates representing
the centre points:

```python
points_s = eqsp.eq_point_set_polar(dim, N)
```

Create an array in polar coordinates representing the regions
of an EQ partition:

```python
regions = eqsp.eq_regions(dim, N)
# regions.shape is (dim, 2, N)
```

### 2. Find properties of EQ partitions

Find the (per-partition) maximum diameter bound of the EQ
partition of $S^d$ into $N$ regions:

```python
from eqsp.region_props import eq_diam_bound

diam_bound = eq_diam_bound(dim, N)
```

### 3. Find properties of EQ point sets

Find the r⁻ˢ energy and min distance of the EQ centre
point sets of Sᵈ for N points:

```python
from eqsp.point_set_props import eq_energy_dist

s = dim - 1  # Standard Riesz energy kernel power
energy, min_dist = eq_energy_dist(dim, N, s)
```

### 4. Produce illustrations

The **PyEQSP** package provides two kinds of plot:

- **2D illustrations** (`eqsp.illustrations`): projections
  rendered with Matplotlib. No extra dependencies required.
- **3D visualizations** (`eqsp.visualizations`): interactive
  plots rendered with Mayavi. Requires the optional `mayavi`
  and `PyQt5` packages.

#### 2D Illustrations (Matplotlib)

Project the EQ partition of S² into N regions onto a
2D plane:

```python
from eqsp.illustrations import project_s2_partition
import matplotlib.pyplot as plt

project_s2_partition(10, proj='stereo')
plt.show()
```

Illustrate the EQ algorithm steps for the partition of Sᵈ
into N regions:

```python
from eqsp.illustrations import illustrate_eq_algorithm

illustrate_eq_algorithm(3, 10)
plt.show()
```

#### 3D Visualizations (Mayavi)

Display a 3D rendering of the EQ partition of S² into N
regions:

```python
from eqsp.visualizations import show_s2_partition

show_s2_partition(10)
# Opens a native Mayavi GUI window.
```

Display a 3D stereographic projection of the EQ partition of
S³ into N regions:

```python
from eqsp.visualizations import project_s3_partition

project_s3_partition(10, proj='stereo')
```

## User Guide Examples

Standalone Python scripts demonstrating core library features:
- **[examples/user-guide/src/](examples/user-guide/src/)**: Contains `example_quick_start.py`, `example_visualize_2d.py`, `example_visualize_3d.py`, and `example_symmetric_partitions.py`.

These examples are fully integrated into the test suite and documentation.

## Thesis Examples

For users interested in reproducing the results from the
original PhD thesis, reproduction scripts are available in the
`examples/phd-thesis/` directory. See
[doc/phd-thesis-examples.md](doc/phd-thesis-examples.md)
for details.

## Performance & Benchmarking

The package includes benchmarks to measure the efficiency of
core partitioning and mathematical operations. See
[doc/benchmarks.md](doc/benchmarks.md) for details.

## Frequently Asked Questions

### Is PyEQSP for S² and S³ only? What is the maximum dimension?

In principle, any function which has `dim` as a parameter will
work for any integer dim ≥ 1 (where S¹ is the circle). In
practice, for large $d$, the functions may be slow or consume
large amounts of memory due to the recursive nature or array
sizes.

### What is the range of the number of points, N?

In principle, any function which takes `N` as an argument will
work with any positive integer value of `N`. In practice, for
very large `N`, the functions may be slow or memory-intensive.

### Visualization options

- `illustrations.project_s2_partition(N, proj=...)`:
  2D projection of S² partition (Matplotlib).
- `illustrations.illustrate_eq_algorithm(dim, N)`:
  Step-by-step visualization (Matplotlib).
- `visualizations.show_s2_partition(N)`:
  3D plot of S² partition (Mayavi).
- `visualizations.project_s3_partition(N, proj=...)`:
  3D projection of S³ partition (Mayavi).

See the docstrings for more details (e.g.
`help(eqsp.visualizations.show_s2_partition)`).

## Package Structure

- `eqsp.partitions`: Core partitioning functions
  (`eq_regions`, `eq_point_set`, `eq_caps`).
- `eqsp.utilities`: Geometric utilities
  (`area_of_cap`, `volume_of_ball`, `polar2cart`, etc.).
- `eqsp.point_set_props`: Properties of point sets
  (energy, min distance).
- `eqsp.region_props`: Properties of regions
  (diameter, vertex max dist).
- `eqsp.illustrations`: 2D visualizations (Matplotlib).
- `eqsp.visualizations`: 3D visualizations (Mayavi).

## Reporting Bugs & Contributing

This project is currently in **Beta testing**. We welcome
your feedback!

- Found a bug? Please
  [open an issue](https://github.com/penguian/pyeqsp/issues/new/choose).
- Want to contribute? See
  [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Citation

If you use this software in research, please cite the original
work:

> Paul Leopardi, "A partition of the unit sphere into regions
> of equal area and small diameter", Electronic Transactions on
> Numerical Analysis, Volume 25, 2006, pp. 309-327.
> http://etna.mcs.kent.edu/vol.25.2006/pp309-327.dir/pp309-327.html

For a recent case study and discussion on the applicability of these
constructions, see:

> Paul Leopardi, "The applicability of equal area partitions of
> the unit sphere", Journal of Approximation Software, Volume 1,
> Issue 2, 2024.
> https://doi.org/10.13135/jas.10248

## License

This software is released under the **MIT License**. See the
`COPYING` file for details.

The original Matlab implementation can be found at:
http://eqsp.sourceforge.net
