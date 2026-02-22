# Recursive Zonal Equal Area Sphere Partitioning Toolbox (Python Port)

This is a Python port of the **Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox**, originally developed in Matlab by Paul Leopardi.

## What is the Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox?

The Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox is a suite of Python functions. These functions are intended for use in exploring different aspects of EQ sphere partitioning.

The functions are grouped into the following groups of tasks:
1.  Create EQ partitions
2.  Find properties of EQ partitions
3.  Find properties of EQ point sets
4.  Produce illustrations
5.  Perform utility functions

## What is an EQ partition?

An **EQ partition** is a partition of $S^d$ (the unit sphere in the $(d+1)$-dimensional Euclidean space $\mathbb{R}^{d+1}$) into a finite number of regions of equal area. The area of each region is defined using the Lebesgue measure inherited from $\mathbb{R}^{d+1}$.

The **diameter** of a region is the supremum of the Euclidean distance between any two points of the region. The regions of an EQ partition have been proven to have small diameter, in the sense that there exists a constant $C(d)$ such that the maximum diameter of the regions of an $N$ region EQ partition of $S^d$ is bounded above by $C(d) \cdot N^{-1/d}$.

## What is an EQ point set?

An **EQ point set** is the set of center points of the regions of an EQ partition. Each region is defined as a product of intervals in spherical polar coordinates. The center point of a region is defined via the center points of each interval, with the exception of spherical caps and their descendants, where the center point is defined using the center of the spherical cap.

## Installation

Requires Python 3.8+ and the following dependencies:
- `numpy`
- `scipy`
- `matplotlib`
- `mayavi`
- `PyQt5`

To install in editable mode (recommended for development):
```bash
pip install -e .
```

To install directly:
```bash
pip install .
```

To install with Mayavi support:
```bash
pip install ".[mayavi]"
```

For detailed step-by-step instructions and environment setup, see [INSTALL.md](INSTALL.md).

## Quick Start & Examples

### Which file to begin with?

You need to find a function which does what you want to do. Here are some examples:

#### 1. Create EQ partitions

Create an array in Cartesian coordinates representing the 'center' points of an EQ partition of $S^d$ into $N$ regions:
```python
import eqsp
dim = 2
N = 100
points_x = eqsp.eq_point_set(dim, N)
# points_x.shape is (dim+1, N)
```

Create an array in spherical polar coordinates representing the 'center' points:
```python
points_s = eqsp.eq_point_set_polar(dim, N)
```

Create an array in polar coordinates representing the regions of an EQ partition:
```python
regions = eqsp.eq_regions(dim, N)
# regions.shape is (dim, 2, N)
```

#### 2. Find properties of EQ partitions

Find the (per-partition) maximum diameter bound of the EQ partition of $S^d$ into $N$ regions:
```python
from eqsp.region_props import eq_diam_bound
diam_bound = eq_diam_bound(dim, N)
```

#### 3. Find properties of EQ point sets

Find the $r^{-s}$ energy and min distance of the EQ 'center' point sets of $S^d$ for $N$ points:
```python
from eqsp.point_set_props import eq_energy_dist
s = dim - 1  # Standard Riesz energy kernel power
energy, min_dist = eq_energy_dist(dim, N, s)
```

#### 4. Produce an illustration

Visualizing partitions requires `matplotlib`.

Use a 3D plot to illustrate the EQ partition of $S^2$ into $N$ regions:
```python
from eqsp.illustrations import show_s2_partition
import matplotlib.pyplot as plt

show_s2_partition(10)
plt.show()
```

Use projection to illustrate the EQ partition of $S^2$ into $N$ regions:
```python
from eqsp.illustrations import project_s2_partition
project_s2_partition(10, proj='stereo')
plt.show()
```

Use projection to illustrate the EQ partition of $S^3$ into $N$ regions:
```python
from eqsp.illustrations import project_s3_partition
project_s3_partition(10, proj='stereo')
plt.show()
```

Illustrate the EQ algorithm steps for the partition of $S^d$ into $N$ regions:
```python
from eqsp.illustrations import illustrate_eq_algorithm
illustrate_eq_algorithm(3, 10)
plt.show()

#### 5. High-Quality 3D Illustrations (Mayavi)

### 3D Visualizations

3D plotting functions have been moved to `eqsp.visualizations`. They require Mayavi.

```python
from eqsp import visualizations
visualizations.show_s2_partition(4)  # Opens a native GUI window
```

> **Note:** Mayavi and PyQt5 are optional dependencies. To use them, install with `pip install .[mayavi]` or install `mayavi` and `PyQt5` separately.

## Thesis Examples

For users interested in reproducing the results from the original PhD thesis, a collection of high-fidelity reproduction scripts is available in the `examples/phd-thesis/` directory.

Each script reproduces a specific figure from the thesis:
- **Numerical Plots**: Diameter bounds, packing density, and energy calculations.
- **3D Visualizations**: High-quality Mayavi renderings of partitions and codes.
- **Argparse Support**: All scripts support `--help` and parameters like `--n-max`.

For more details, see the [Thesis Example Reproductions](doc/phd-thesis-examples.md).

## Performance & Benchmarking

The package includes a collection of benchmarks to measure the efficiency of core partitioning and math logic. To run the full suite:

```bash
python3 benchmarks/src/run_benchmarks.py
```

For detailed instructions on configurable benchmarks and performance analysis, see the [Performance Benchmarks Guide](doc/benchmarks.md).

## Frequently Asked Questions

### Is the toolbox for use with $S^2$ and $S^3$ only? What is the maximum dimension?

In principle, any function which has `dim` as a parameter will work for any integer `dim >= 1` (where $S^1$ is the circle). In practice, for large $d$, the functions may be slow or consume large amounts of memory due to the recursive nature or array sizes.

### What is the range of the number of points, $N$?

In principle, any function which takes `N` as an argument will work with any positive integer value of `N`. In practice, for very large `N`, the functions may be slow or memory-intensive.

### What are the options for visualizing points or equal area regions?

- `show_s2_partition(N)`: 3D plot of $S^2$ partition.
- `project_s2_partition(N, proj='stereo'|'eqarea')`: 2D projection of $S^2$ partition.
- `project_s3_partition(N, proj='stereo'|'eqarea')`: 3D projection of $S^3$ partition.
- `illustrate_eq_algorithm(dim, N)`: Step-by-step visualization.

See the docstrings for these functions for more details (e.g. `help(eqsp.illustrations.show_s2_partition)`).

## Package Structure

- `eqsp.partitions`: Core partitioning functions (`eq_regions`, `eq_point_set`, `eq_caps`).
- `eqsp.utilities`: Geometric utilities (`area_of_cap`, `volume_of_ball`, `polar2cart`, etc.).
- `eqsp.point_set_props`: Properties of point sets (energy, min distance).
- `eqsp.region_props`: Properties of regions (diameter, vertex max dist).
- `eqsp.illustrations`: Logic for 2D visualizations (Matplotlib).
- `eqsp.visualizations`: Logic for 3D visualizations (Mayavi).

## Citation

If you use this software in research, please cite the original work:

> Paul Leopardi, "A partition of the unit sphere into regions of equal area and small diameter",
> Electronic Transactions on Numerical Analysis, Volume 25, 2006, pp. 309-327.
> http://etna.mcs.kent.edu/vol.25.2006/pp309-327.dir/pp309-327.html

## License

This software is released under the **MIT License**. See the `COPYING` file for details.


The original Matlab implementation can be found at:
http://eqsp.sourceforge.net
