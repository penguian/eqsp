# Symmetric EQ Partitions (`even_collars`)

PyEQSP provides a specialized mode for generating partitions that are perfectly symmetric about the sphere's equator. This is controlled via the `even_collars` parameter in the core API.

## Motivation

Many research applications require the equator ($\theta = \pi/2$) to fall exactly on a region boundary:

*   **Robotics & Orientation Estimation**: Hyperhemispherical grid filters ({ref}`Pfaff et al., 2020 <pfa20>`) construct one half of a partition and mirror it.
*   **Biomedical RNA Folding**: Sampling the rotation groups $SO(3)$ via quaternions on $S^3$. Due to double-covering, researchers often only need to sample the upper hyperhemisphere, requiring an exact equatorial split.

## Using the API

To force a symmetric partition, set `even_collars=True` in `eq_regions` or `eq_point_set`.

```python
from eqsp.partitions import eq_regions

# N must be even for even_collars=True
regions = eq_regions(dim=2, N=100, even_collars=True)
```

:::tip
See [examples/user-guide/src/example_symmetric_partitions.py](https://github.com/penguian/pyeqsp/blob/main/examples/user-guide/src/example_symmetric_partitions.py) for a complete example including symmetry verification.
:::

:::important
If $N$ is odd, setting `even_collars=True` will raise a `ValueError`. This is because for odd $N$, no single hyperplane can perfectly divide the partition into two equal hemispheres.
:::

## Effectiveness and Research Basis

The symmetric modification preserves the **Equal Area** property exactly and maintains the $O(N^{-1/d})$ **Diameter Bound** established in {ref}`Leopardi (2009) <leo09>`.
 While the fitting angle may deviate slightly from the "asymmetric ideal," the difference becomes negligible as $N$ increases.

For technical details on the underlying optimizations (KDTrees, Symmetry exploitation, and Process Pools), see the [Algorithmic Optimizations](../maintainer/algorithmic_optimizations.md) guide.
