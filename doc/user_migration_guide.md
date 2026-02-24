# User Migration Guide: EQSP Toolbox (Matlab to Python)

This guide helps users of the original Matlab `eq_sphere_partitions` toolbox transition to the Python `eqsp` package. While most core functionality remains identical, there are some differences in naming conventions, API structure, and usage patterns.

## 1. Quick Reference: Function Name Mapping

Most core functions retain their names. The main differences are in coordinate conversion utilities and illustration functions.

| Matlab Function | Python Function (`eqsp`) | Notes |
| :--- | :--- | :--- |
| **Partitions** | | |
| `eq_point_set` | `eq_point_set` | Identical usage. |
| `eq_regions` | `eq_regions` | Identical usage. |
| `eq_min_dist` | `eq_min_dist` | Identical usage; optimized ($O(N \log N)$) in Python. |
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

## 2. API & Usage Differences

### 2.1 Keyword Arguments
Matlab functions often used "Name, Value" pairs for options. Python uses standard keyword arguments.

**Matlab:**
```matlab
eq_point_set(2, 10, 'offset', 'extra')
```

**Python:**
```python
eq_point_set(2, 10, extra_offset=True)
```

### 2.2 Return Values
Some functions have been refactored to return consistent types, avoiding fragile dependence on the number of output arguments (`nargout`).

*   **`eqsp.region_props.eq_diam_coeff`**: Always returns a tuple `(bound_coeff, vertex_coeff)`.
    *   *Matlab*: Behavior varied; often returned only one value if `nargout` was 1.
    *   *Python*: Unpack the result: `bound, vertex = eq_diam_coeff(...)`.

### 2.3 Coordinate Conventions
*   **Spherical Coordinates**: `eqsp` uses `(phi, theta)` where:
    *   `phi`: Longitude in `[0, 2*pi)`.
    *   `theta`: Colatitude in `[0, pi]` (0 is North Pole).
    *   This matches the standard mathematical convention used in the original paper.

### 2.4 Array Handling
*   Input arrays are generally handled as Numpy arrays.
*   Functions are vectorized where appropriate, similar to Matlab.

### 2.5 3D Plotting: `illustrations` vs. `visualizations`
The Python port uses two separate modules for plotting, unlike the single Matlab illustration module:

*   **`eqsp.illustrations`** (Matplotlib, always available): Handles 2D projections (`project_s2_partition`) and algorithm step diagrams (`illustrate_eq_algorithm`). Functions that require 3D rendering raise `NotImplementedError` and direct you to `eqsp.visualizations`.
*   **`eqsp.visualizations`** (Mayavi, optional): Handles all 3D interactive rendering — `show_s2_partition`, `project_s3_partition`, `show_r3_point_set`, etc. Requires Mayavi; see §4.1 for installation notes.

## 3. Module Structure
The package is organized into logical modules:

*   `eqsp.partitions`: Core partition algorithms (`eq_regions`, `eq_point_set`).
*   `eqsp.utilities`: Math helpers and coordinate conversions.
*   `eqsp.region_props`: Properties of regions (area, diameter).
*   `eqsp.point_set_props`: Properties of point sets (energy, distance).
*   `eqsp.histograms`: Point-in-region lookup and counting for S^2.
*   `eqsp.illustrations`: 2D Matplotlib plotting and algorithm diagrams.
*   `eqsp.visualizations`: 3D Mayavi visualizations (optional dependency).

## 4. Installation & Getting Started
Install via:
```bash
pip install eqsp
```

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

### 4.1 System Packages (Advanced)
If you rely on system-installed packages like `mayavi` (via `apt`), see [doc/python_environments.md](python_environments.md) for instructions on setting up a compatible virtual environment (`venv_sys`).

> **Note:** This configuration was specifically tested on **Kubuntu Linux 25.10**. Different environments may require different values for environment variables like `QT_API`.

## 5. Learning from Examples

For a deep dive into how the Python API corresponds to the original Matlab implementation, see the [PhD Thesis Example Reproductions](phd-thesis-examples.md)
 document.
