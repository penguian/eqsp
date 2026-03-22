# Performance Highlights & Algorithmic Optimizations

This document provides a technical summary of the key algorithmic optimizations implemented in the **PyEQSP** Python package to ensure scalability and efficiency for large-scale sphere partitioning analysis.

## Algorithmic Improvements

### Min-Distance Optimization
**Status:** Improved from $O(N^2)$ to $O(N \log N)$.

- **Previous Bottleneck:** Pairwise distance matrices created using `scipy.spatial.distance.pdist` or NumPy broadcasting consumed $O(N^2)$ memory and time.
- **Approach:** We leverage **KDTrees** (`scipy.spatial.KDTree`) for $S^d$ ($d \le 4$) to perform localized neighbour searches. For higher dimensions or specific partition types, we use **structure-aware searches** that exploit the recursive nature of the EQ algorithm to bound the search space.
- **Result:** Calculating the min-distance for $N=100,000$ points now takes seconds rather than minutes, and memory usage remains linear.

### Recursive Partitioning Scaling
**Status:** Verified $O(\mathcal{N}^{0.6})$ scaling ({ref}`[Leo07] <v2-leo07>`, Section 3.10.2).

- **Approach:** The `eq_regions` function implements the Recursive Zonal Equal Area Partitioning algorithm. We performed a high-fidelity verification sweep for $d$ up to 11 and $\mathcal{N}$ up to $2^{22}$ ($\approx 4.2 \times 10^6$).
- **Result:** The performance follows the theoretical $O(\mathcal{N}^{0.6})$ scaling for $S^2$, ensuring that even millions of regions can be calculated in minutes.

### Riesz Energy Calculations
**Status:** Memory-efficient $O(N^2)$ with Symmetry Exploitation ($0.5 \times$ work).

- **Previous Bottleneck:** Creating a full $N \times N$ distance matrix to sum $d_{ij}^{-s}$ lead to $O(N^2)$ memory exhaustion (e.g., 3.2 GB for $N=20,000$).
- **Approach:**
    - **Block-based Summation (Tiling):** Calculations are performed in blocks of size $M \times M$, keeping the peak memory footprint at $O(N \times M)$ instead of $O(N^2)$.
    - **Symmetry Exploitation:** Since $d_{ij} = d_{ji}$, we only compute the upper triangle of the interaction matrix, effectively doubling the performance.
- **Result:** $N=20,000$ energy calculations complete in 5–10 minutes on standard hardware.

### Histogram-Based Region Lookup ($S^2$)
**Status:** Fully Vectorized.

- **Approach:** Assigning points to regions on $S^2$ used a recursive Python loop. The new implementation uses **logarithmic searching** (`np.searchsorted`) across vectorized "collar" boundaries.
- **Result:** Billions of sample points can be binned into partitions in a single vectorized pass.

### Symmetric Partition Performance (`even_collars`)
**Status:** Vectorized support in all properties functions.

- **Approach:** The symmetric partitioning logic (`even_collars=True`) ensures an even number of collars. All property calculation functions (`eq_area_error`, `eq_min_dist`, `eq_energy_dist`, `eq_diam_coeff`) are fully vectorized to support this parameter.
- **Result:** Symmetry calculation adds negligible overhead compared to standard partitions. In some cases, the simplified collar recurrence in symmetric mode can lead to slight performance improvements for high $N$.

## Optimized NumPy & SciPy Patterns

During development, many "hot" paths were refactored to use more efficient library patterns:

### Vectorized Root Finding
In `sradius_of_cap`, we replaced a Python loop over `scipy.optimize.root_scalar` with a vectorized Newton-Raphson implementation using `scipy.optimize.newton` on arrays. This provides a 10–50x speedup for high-dimensional spherical cap radius calculations.

### Efficient Coordinate Transforms
Functions in `utilities.py` (like `cart2polar2`) were refactored to use:
- `np.atleast_2d` to handle both single points and large arrays uniformly.
- Direct array operations instead of `for` loops.
- `np.arctan2` and `np.arccos` for numerically stable conversions.

### Avoiding Massive Intermediate Arrays
Common patterns like `np.linalg.norm(a[:, None] - b, axis=2)` were replaced with more memory-conscious implementations or block-based loops where the intermediate broadcasting would exceed available RAM.

### Parallel Dimension Calculations ({ref}`[Leo07] <v2-leo07>`, Figure 3.7)

`fig_3_7_max_diam_multi_dim.py` calculates max diameter coefficients for EQ partitions across dimensions $d=2$ to $d=8$. The dimension-8 calculation alone accounts for approximately **81%** of the total CPU time, making it the dominant bottleneck.

- **Approach:** Uses `concurrent.futures.ProcessPoolExecutor` with `max_workers=2`. Dimensions are dispatched in **decreasing order** so that `dim=8` begins immediately, while the second worker handles `dim=7` and then the remaining smaller dimensions sequentially.
- **Result:** For a full-fidelity thesis run ($N=2^{20}$), wall-clock time is reduced (e.g., from ~1h 45m to ~1h 24m). This matches the theoretical max improvement for a 2-worker strategy given the Amdahl's Law limit imposed by the serial `dim=8` task.

---

For detailed benchmarks and instructions on running performance tests yourself, see the [Performance Benchmarks](../benchmarks.md).
