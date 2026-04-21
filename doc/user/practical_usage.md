# Practical Usage Guide

This guide provides a hands-on introduction to using the core PyEQSP functions for research and data analysis.

## Step 1: Initialize a Partition

The most common task is generating a set of $N$ points roughly uniformly distributed on a sphere.

### Generating Regions
Use `eq_regions` to find the boundaries of the partition.
```python
import eqsp
regions = eqsp.eq_regions(dim=2, N=1000)
```

### Generating Point Sets
Use `eq_point_set` to get the centre points of each region.
```python
points = eqsp.eq_point_set(dim=2, N=1000)
# points is a (3, 1000) NumPy array (column-major convention)
```

## Step 2: Analyze Geometric Properties

Once you have a point set, you can measure its quality using metrics.

### Min-Distance
Measuring the separation between points helps check packing efficiency.
```python
# Measure min distance from existing points
min_dist = eqsp.point_set_min_dist(points)
print(f"Min-distance: {min_dist}")
```

### Riesz Energy
Calculate the energy of the point set to test its uniformity from a physical perspective.
```python
# Measure energy from existing points
energy, _ = eqsp.point_set_energy_dist(points, s=2)
print(f"Riesz s-energy: {energy}")
```

:::{tip}
A complete standalone version of this workflow can be found in [examples/user-guide/src/example_quick_start.py](https://github.com/penguian/pyeqsp/blob/main/examples/user-guide/src/example_quick_start.py).
:::

## Step 3: Advanced Partitioning ($S^3$ and SO(3))

For applications involving rotations or orientations, you can partition the higher-dimensional $S^3$.
```python
# Partition into 5000 regions on S^3
points_s3 = eqsp.eq_point_set(dim=3, N=5000)
```

## Step 4: Fast Vectorized Binning

If you have a large set of data points (e.g., climate observations) and need to bin them into equal-area regions:
1. Generate the regions using `eq_regions`.
2. Map your data points to the regions.

*Note: PyEQSP uses high-performance NumPy vectorization to handle large datasets efficiently.*
