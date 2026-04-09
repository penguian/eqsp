# Performance Expectations

This guide helps researchers and scientists understand the computational resources required when working with large-scale PyEQSP partitions.

## Scaling Laws

PyEQSP is optimized to handle millions of regions on standard hardware.

| Operation | Complexity | Expectation |
| :--- | :--- | :--- |
| **Partition Generation** | $O(N^{0.6})$ | Millions of regions in minutes on $S^2$. |
| **Min-Distance** | $O(N \log N)$ | Seconds for $N=100,000$. |
| **Packing Density** | $O(N \log N)$ | Near-instant results for research-grade $N$. |
| **Riesz Energy** | $O(N^2)$ | Under 15 seconds for $N=50,000$. |

## Memory Safety

PyEQSP uses **tiling** and **block-based summation** to ensure that large calculations do not exhaust your system RAM.

*   **Energy Calculations**: Calculating interaction energy for 20,000 points would normally require ~3.2 GB of RAM for the distance matrix. PyEQSP processes these in small "blocks," keeping the peak memory footprint low and linear.
*   **Vectorization**: Most operations are performed using NumPy's vectorized paths, ensuring high-speed execution even when $N$ exceeds one million.

## High-Dimensional Considerations ($S^3$ and above)

As the dimension increases, the recursive depth of the algorithm grows.
- **Symmetry Speed-up**: Using `even_collars=True` can speed up calculations on $S^3$ by enabling a 100% cache hit rate for the southern hemisphere.
- **Parallelization**: Scripts like `fig_3_7_max_diam_multi_dim.py` are already parallelized to handle the increased complexity of higher dimensions (up to $S^8$).

For technical details on the underlying optimizations (KDTrees, Symmetry exploitation, and Process Pools), see the [Algorithmic Optimizations](../maintainer/algorithmic_optimizations.md) guide.
