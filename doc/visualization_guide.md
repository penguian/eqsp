# Visualization & Illustration Guide

PyEQSP provides powerful tools for visualizing partitions and point sets in both 2D and 3D.

## Simple 2D Illustrations

For quick analysis, 2D projections of the EQ algorithm are highly effective. The standard illustration shows how the sphere is partitioned into collars and regions.

### EQ Algorithm Illustration
The `eqsp.illustrations` module provides functions to visualize the "igloo" partitioning scheme.

```python
import matplotlib.pyplot as plt
from eqsp import illustrations

# Show the partitioning for 100 regions on S2
plt.figure()
illustrations.illustrate_eq_algorithm(dim=2, N=100)
plt.show()
```

![EQ Algorithm 2D](_static/images/eq_algorithm_2d.png)

## Interactive 3D Visualizations

To truly understand the geometry of a partition or to inspect point sets on $S^2$ and $S^3$, PyEQSP leverages **Mayavi** for interactive 3D rendering.

### Partitioned Spheres
You can rotate, zoom, and inspect the individual regions of a partition in 3D space.

```python
from eqsp import visualizations

# Show a 3D partition of 100 regions with center points
visualizations.show_s2_partition(N=100, show_points=True)
```

![S2 Partition 3d](_static/images/s2_partition_3d.png)

## Advanced Projections

For specific research or mapping needs, the library also supports more advanced projections via `eqsp.visualizations.plot_regions_2d`.

*   **Mollweide**: Equal-area projection of the entire sphere.
*   **Lambert**: Azimuthal equal-area projection.
*   **Polar**: For focusing on the North/South poles.

## Jupyter Integration

PyEQSP is designed to work seamlessly in Jupyter Notebooks.
- **Inline Matplotlib**: Use `%matplotlib inline` for static plots.
- **Interactive Widgets**: Use `%matplotlib widget` for interactive research. Mayavi can be configured via `mlab.init_notebook()`.
