# Migration from MATLAB

This guide provides a technical bridge for users transitioning from the original MATLAB `eq_sphere_partitions` toolbox to the Python **PyEQSP** library. While the core mathematical algorithms for recursive partitioning remain identical, the implementation has been modernized to exploit NumPy's vectorization and Python's efficient memory management.

## Performance Comparison (Baseline)

The following benchmark demonstrates the significant performance improvements in the Python implementation when executed on the same hardware (**AMD Ryzen 7 8840HS**, 2.4 GHz).

| Operation | Scale ($N$ or points) | MATLAB (s) | Python (s) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **`eq_area_error`** | 1,000,000 | 0.247 | 0.099 | **2.5x** |
| **`sradius_of_cap`** | 1,000,000 | 35.906 | 0.362 | **99x** |
| **`eq_find_s2_region`** | 2,000,000 | 28.322 | 0.220 | **129x** |
| **`eq_min_dist`** | 50,000 | 23.877 | 0.068 | **350x** |
| **`eq_energy_dist`** | 10,000 | 1.504 | 0.437 | **3.4x** |

> **Note:** The dramatic speedups in `sradius_of_cap` and `eq_min_dist` are due to the transition from MATLAB loops to vectorized Newton-Raphson solvers and KDTree-based spatial searches, respectively.

## Performance Features & Optimizations

The Python port introduces several algorithmic optimizations and internal logic improvements compared to the original MATLAB toolbox:

- **Index Rotation (Histogram Logic)**: Solving the longitude wrap-around issue involved the implementation of an **Index Rotation** (Domain Translation) step.
    - a. Points and boundaries are shifted by the collar's first boundary ($\phi_0$) using a modulo $2\pi$ operation.
    - b. The final boundary is explicitly set to $2\pi$ to ensure strict monotonicity.
    - c. The legacy `lookup_table()` utility was removed in favor of this direct, domain-translated `np.searchsorted()` approach for 100% test coverage and better performance.
- **Min-Distance Optimization**: Optimized to **O(N log N)** using **KDTrees**. Calculating **d_min** for **N=100,000** points is now nearly instantaneous.
- **Riesz Energy**: Uses a **block-based symmetry-aware summation** ($d_{ij} = d_{ji}$). Peak memory remains **O(N)** and total work is halved compared to naive **O(N²)** implementations.
- **Precision Rounding**: Latitude (band) lookups include $10^{-12}$ rounding to prevent points on a boundary from jumping bands due to floating-point variance.

## Quick Reference: Function Name Mapping

Most core functions keep their names. The main differences are in coordinate conversion utilities and illustration functions.

| Matlab Function | Python Function (`eqsp`) | Notes |
| :--- | :--- | :--- |
| **Partitions** | | |
| `eq_point_set` | `eq_point_set` | Identical usage. |
| `eq_regions` | `eq_regions` | Identical usage. |
| `eq_min_dist` | `eq_min_dist` | Identical usage; optimized (**O(N log N)**) in Python. |
| `eq_energy_dist` | `eqsp.eq_energy_dist` | Optimized for **O(N)** memory and symmetry. |
| **Region & Point Properties** | | |
| `eq_area_error` | `region_props.eq_area_error` | Direct port. |
| `eq_vertex_diam` | `region_props.eq_vertex_diam` | Direct port. |
| `eq_vertex_diam_coeff` | `region_props.eq_vertex_diam_coeff` | Direct port. |
| `eq_packing_density` | `point_set_props.eq_packing_density` | Direct port. |
| `point_set_packing_density`| `point_set_props.point_set_packing_density` | Direct port. |
| **Utilities** | | |
| `pol2cart` | `utilities.polar2cart` | Renamed for clarity. |
| `cart2pol` | `utilities.cart2polar2` | Same name. NumPy vectorization replaces the loop-based Matlab implementation. |
| `area_of_sphere` | `utilities.area_of_sphere` | Direct port. |
| `area_of_cap` | `utilities.area_of_cap` | Direct port. |
| `area_of_collar` | `utilities.area_of_collar` | Direct port. |
| `euc2sph_dist` | `utilities.euc2sph_dist` | Direct port. |
| `ideal_collar_angle` | `utilities.ideal_collar_angle` | Direct port. |
| `sradius_of_cap` | `utilities.sradius_of_cap` | Direct port. |
| `volume_of_ball` | `utilities.volume_of_ball` | Direct port. |
| `spherical_dist` | `utilities.spherical_dist` | Direct port. |
| **Histograms** | | |
| `eq_count_points_by_s2_region` | `histograms.eq_count_points_by_s2_region` | Direct port. |
| `eq_find_s2_region` | `histograms.eq_find_s2_region` | Direct port. |
| `in_s2_region` | `histograms.in_s2_region` | Direct port. |
| **2D Illustrations** | | |
| `illustrate_eq_algorithm` | `illustrations.illustrate_eq_algorithm` | Matplotlib. |
| `project_s2_partition` | `illustrations.project_s2_partition` | Matplotlib, 2D projection. |
| `project_point_set` | `illustrations.project_point_set` | Matplotlib, 2D projection. |
| **3D Visualizations** | | |
| `plot_s2_partition` | `visualizations.show_s2_partition` | Mayavi (optional). |
| `project_s3_partition` | `visualizations.project_s3_partition` | Mayavi (optional). |

> **Note:** Internal-only utilities from the original Matlab code (like `fatcurve`) are not exposed in the public Python API.

## API & Usage Differences

### Keyword Arguments
Matlab functions often used "Name, Value" pairs for options. Python uses standard keyword arguments.

**Matlab:**
```matlab
eq_point_set(2, 10, 'offset', 'extra')
```

**Python:**
```python
eq_point_set(2, 10, extra_offset=True)
```

> **Note:** The Matlab `partition_options` object is replaced by keyword arguments in Python. No same configuration object exists in **PyEQSP**.

> **Python Exclusive:** The Python port introduces an `even_collars=True` boolean parameter to `eq_caps` (and downstream functions like `eq_regions` and `eq_point_set`). This forces the partition to have an even number of collars, ensuring the equatorial hyperplane cleanly splits the partition into two equal hemispheres. This parameter does not exist in the Matlab toolbox.

Furthermore, some Python parameters are entirely new to **PyEQSP** and did not exist in the Matlab toolbox:

*   **`even_collars`**: A new boolean parameter passed to partition functions (e.g., `eq_caps(..., even_collars=True)`). This forces an even number of collars, ensuring the equatorial hyperplane perfectly aligns with a cap boundary. This allows for mathematically precise **S²** hemisphere splitting and **S³ → SO(3)** quaternion sampling (for more details see [Symmetric EQ Partitions](even_collar_partitions.md)).
*   **Vectorized Properties**: All property evaluation functions (like `eq_min_dist`, `eq_energy_dist`, `eq_area_error`, and `eq_diam_coeff`) also accept the `even_collars` parameter to test symmetric partitions.

### Return Values
Some functions have been refactored to return consistent types, avoiding fragile dependence on the number of output arguments (`nargout`).

*   **`eqsp.region_props.eq_diam_coeff`**: Always returns a tuple `(bound_coeff, vertex_coeff)`.
    *   *Matlab*: Behaviour varied; often returned only one value if `nargout` was 1.
    *   *Python*: Unpack the result: `bound, vertex = eq_diam_coeff(...)`.

### Coordinate Conventions
*   **Spherical Coordinates**: `eqsp` uses `(phi, theta)` where:
    *   `phi`: Longitude in `[0, 2*pi)`.
    *   `theta`: Colatitude in `[0, pi]` (0 is North Pole).
    *   This matches the standard mathematical convention used in the original paper and the Matlab toolbox.

### Array Handling
*   Input arrays are generally handled as Numpy arrays.
*   Functions are vectorized where appropriate, as in Matlab.

### Indexing: 0-based vs 1-based
Perhaps the most significant difference for Matlab users is that **Python uses 0-based indexing**.
- **Matlab**: `A(1)` is the first element.
- **Python**: `A[0]` is the first element.

This impacts loops and range-based operations:
- `for i in range(N):` iterates from `0` to `N-1`.
- `A[0:k]` selects elements from index `0` up to (but not including) index `k`.

### Array Orientation and Shape
Matlab and NumPy differ in their default memory layout (Column-major vs Row-major).
- **Default Shape**: Most `eqsp` coordinate functions return arrays of shape **(d+1, N)**. This matches the original Matlab convention.
- **Interoperability**: Many other Python libraries (like `scikit-learn` or `pandas`) expect data in "long" format: **(N, features)**.
- **The Solution**: Use the transpose operator `.T` to swap axes efficiently:
  ```python
  points = eqsp.eq_point_set(2, 10)  # Shape: (3, 10)
  points_T = points.T                # Shape: (10, 3)
  ```

### 3D Plotting: `illustrations` vs. `visualizations`
The Python port uses two separate modules for plotting, unlike the single Matlab illustration module:

*   **`eqsp.illustrations`** (Matplotlib, always available): Handles 2D projections (`project_s2_partition`) and algorithm step diagrams (`illustrate_eq_algorithm`).
*   **`eqsp.visualizations`** (Mayavi, optional): Handles all 3D interactive rendering — `show_s2_partition`, `project_s3_partition`, `show_r3_point_set`, etc. Requires Mayavi.

### Documentation Philosophy: Two Volumes
Starting with 0.99.4, the documentation is divided into a **User Guide (Volume 1)** for researchers and a **Maintenance Guide (Volume 2)** for developers. This ensures that technical implementation details (like CI setup or release procedures) do not clutter the practical usage guides.

## Module Structure
The package is organized into logical modules:

*   `eqsp.partitions`: Core partition algorithms (`eq_regions`, `eq_point_set`).
*   `eqsp.utilities`: Mathematics helpers and coordinate conversions.
*   `eqsp.region_props`: Properties of regions (area, diameter).
*   `eqsp.point_set_props`: Properties of point sets (energy, distance).
*   `eqsp.histograms`: Point-in-region lookup and counting for S^2.
*   `eqsp.illustrations`: 2D Matplotlib plotting and algorithm diagrams.
*   `eqsp.visualizations`: 3D Mayavi visualizations (optional dependency).



## Key Features of PyEQSP

Compared to the original MATLAB toolbox, **PyEQSP** provides some distinct advantages:

- **Symmetry Support**: The `even_collars` parameter enables symmetric partitions for $S^3 \to SO(3)$ sampling.
- **High Performance**: Vectorized mathematical operations and $O(N \log N)$ spatial lookups.


## Platform Compatibility

- **Robustness**: 0.99.4 introduces **case-insensitive backend guards** and **environment isolation**, ensuring scripts run warning-free in both interactive and headless/CI environments.

## Performance Features

The Python port includes many algorithmic optimizations that were not included in the original Matlab toolbox:

- **Min-Distance**: Optimized to **O(N log N)** using KDTrees. Calculating **d_min** for **N=100,000** points is now nearly instantaneous.
- **Riesz Energy**: Uses a **block-based symmetry-aware summation**. Peak memory remains **O(N)** and total work is halved compared to naive **O(N²)** implementations.
- **Histogram Lookups**: Fully vectorized point-in-region assignment on **S²** for bulk processing of points.

## Common Matlab-to-Python "Gotchas"

| Feature | Matlab | Python / **eqsp** (Package) |
| :--- | :--- | :--- |
| **Indexing** | 1, 2, 3… | 0, 1, 2… |
| **Loops** | `for i=1:N` (inclusive) | `for i in range(N)` (exclusive of N) |
| **Functions** | No `import` needed | `import eqsp` |
| **Logic** | `&&`, `||`, `~` | `and`, `or`, `not` |
| **Equality** | `==` | `==` (for values), `is` (for identity) |
| **Slicing** | `A(start:end)` | `A[start:end]` (exclusive of end) |

## Learning from Examples

For a deep dive into how the Python API corresponds to the original Matlab implementation, see the [PhD Thesis Example Reproductions](phd-thesis-examples.md)
 document.
