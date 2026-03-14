# User Migration Guide: Matlab Toolbox to PyEQSP

This guide helps users of the original Matlab `eq_sphere_partitions` toolbox transition to the Python **PyEQSP** project (`eqsp` package). While most core functionality remains identical, there are some differences in naming conventions, API structure, and usage patterns.

## Quick Reference: Function Name Mapping

Most core functions retain their names. The main differences are in coordinate conversion utilities and illustration functions.

| Matlab Function | Python Function (`eqsp`) | Notes |
| :--- | :--- | :--- |
| **Partitions** | | |
| `eq_point_set` | `eq_point_set` | Identical usage. |
| `eq_regions` | `eq_regions` | Identical usage. |
| `eq_min_dist` | `eq_min_dist` | Identical usage; optimized (**O(N log N)**) in Python. |
| `eq_energy_dist` | `point_set_props.eq_energy_dist` | Optimized for **O(N)** memory and symmetry. |
| **Utilities** | | |
| `pol2cart` | `utilities.polar2cart` | Renamed for clarity. |
| `cart2pol` | `utilities.cart2polar2` | Renamed. Handles arrays. |
| `area_of_sphere` | `utilities.area_of_sphere` | |
| **Histograms** | | |
| `eq_count_points_by_s2_region` | `histograms.eq_count_points_by_s2_region` | New in Python port. |
| `eq_find_s2_region` | `histograms.eq_find_s2_region` | New in Python port. |
| **2D Illustrations** | | |
| `illustrate_eq_algorithm` | `illustrations.illustrate_eq_algorithm` | Matplotlib. |
| `project_s2_partition` | `illustrations.project_s2_partition` | Matplotlib, 2D projection. |
| **3D Visualizations** | | |
| `plot_s2_partition` | `visualizations.show_s2_partition` | Mayavi (optional). |
| `project_s3_partition` | `visualizations.project_s3_partition` | Mayavi (optional). |

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

> **Python Exclusive:** The Python port introduces an `even_collars=True` boolean parameter to `eq_caps` (and downstream functions like `eq_regions` and `eq_point_set`). This forces the partition to have an even number of collars, ensuring the equatorial hyperplane cleanly splits the partition into two equal hemispheres. This parameter does not exist in the Matlab toolbox.

Furthermore, some Python parameters are entirely new to **PyEQSP** and did not exist in the Matlab toolbox:

*   **`even_collars`**: A new boolean parameter passed to partition functions (e.g., `eq_caps(..., even_collars=True)`). This forces an even number of collars, ensuring the equatorial hyperplane perfectly aligns with a cap boundary. This allows for mathematically precise **S²** hemisphere splitting and **S³ → SO(3)** quaternion sampling (for more details see [doc/even_collar_partitions.md](even_collar_partitions.md)).
*   **Vectorized Properties**: All property evaluation functions (like `eq_min_dist`, `eq_energy_dist`, `eq_area_error`, and `eq_diam_coeff`) also accept the `even_collars` parameter to evaluate symmetric partitions.

### Return Values
Some functions have been refactored to return consistent types, avoiding fragile dependence on the number of output arguments (`nargout`).

*   **`eqsp.region_props.eq_diam_coeff`**: Always returns a tuple `(bound_coeff, vertex_coeff)`.
    *   *Matlab*: Behaviour varied; often returned only one value if `nargout` was 1.
    *   *Python*: Unpack the result: `bound, vertex = eq_diam_coeff(...)`.

### Coordinate Conventions
*   **Spherical Coordinates**: `eqsp` uses `(phi, theta)` where:
    *   `phi`: Longitude in `[0, 2*pi)`.
    *   `theta`: Colatitude in `[0, pi]` (0 is North Pole).
    *   This matches the standard mathematical convention used in the original paper.

### Array Handling
*   Input arrays are generally handled as Numpy arrays.
*   Functions are vectorized where appropriate, similar to Matlab.

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
*   **`eqsp.visualizations`** (Mayavi, optional): Handles all 3D interactive rendering — `show_s2_partition`, `project_s3_partition`, `show_r3_point_set`, etc. Requires Mayavi; see [System Packages (Advanced)](#system-packages-advanced) for installation notes.

## Module Structure
The package is organized into logical modules:

*   `eqsp.partitions`: Core partition algorithms (`eq_regions`, `eq_point_set`).
*   `eqsp.utilities`: Math helpers and coordinate conversions.
*   `eqsp.region_props`: Properties of regions (area, diameter).
*   `eqsp.point_set_props`: Properties of point sets (energy, distance).
*   `eqsp.histograms`: Point-in-region lookup and counting for S^2.
*   `eqsp.illustrations`: 2D Matplotlib plotting and algorithm diagrams.
*   `eqsp.visualizations`: 3D Mayavi visualizations (optional dependency).

## Installation & Getting Started
Install the package from PyPI via:
```bash
pip install pyeqsp
```

### Naming Distinction

Note that while the project is branded **PyEQSP** and the installation name is **`pyeqsp`**, the Python package name remains **`eqsp`** to preserve compatibility and standard naming conventions:

| Context | Name |
| :--- | :--- |
| **Official Branding** | PyEQSP |
| **PyPI / Install Name** | `pyeqsp` |
| **Python Import Name** | `eqsp` |
| **Repository Name** | `pyeqsp` |

Basic usage:
```python
import eqsp
import numpy as np

# Generate 10 points on S^2
points = eqsp.eq_point_set(2, 10)

# 2D projected view (Matplotlib, no extra dependencies)
from eqsp import illustrations as ill
ill.project_s2_partition(10, proj='eqarea')

# 3D interactive view (requires Mayavi)
from eqsp import visualizations as vis
vis.show_s2_partition(10)
```

### System Packages (Advanced)
If you rely on system-installed packages like `mayavi` (via `apt`), see [doc/python_environments.md](python_environments.md) for instructions on setting up a compatible virtual environment (`venv_sys`).

> **Note:** This configuration was specifically tested on **Kubuntu Linux 25.10**. Different environments may require different values for environment variables like `QT_API`.

## Performance "Killer Features"

The Python port includes several algorithmic optimizations that significantly outperform the original Matlab toolbox:

- **Minimum Distance**: Optimized to **O(N log N)** using KDTrees. Calculating **d_min** for **N=100,000** points is now nearly instantaneous.
- **Riesz Energy**: Uses a **block-based symmetry-aware summation**. Peak memory remains **O(N)** and total work is halved compared to naive **O(N²)** implementations.
- **Histogram Lookups**: Fully vectorized point-in-region assignment on **S²** for bulk processing of billions of points.

## Common Matlab-to-Python "Gotchas"

| Feature | Matlab | Python / **eqsp** (Package) |
| :--- | :--- | :--- |
| **Indexing** | 1, 2, 3... | 0, 1, 2... |
| **Loops** | `for i=1:N` (inclusive) | `for i in range(N)` (exclusive of N) |
| **Functions** | No `import` needed | `import eqsp` |
| **Logic** | `&&`, `||`, `~` | `and`, `or`, `not` |
| **Equality** | `==` | `==` (for values), `is` (for identity) |
| **Slicing** | `A(start:end)` | `A[start:end]` (exclusive of end) |

## Learning from Examples

For a deep dive into how the Python API corresponds to the original Matlab implementation, see the [PhD Thesis Example Reproductions](phd-thesis-examples.md)
 document.
