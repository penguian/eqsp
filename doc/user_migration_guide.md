# User Migration Guide: EQSP Toolbox (Matlab to Python)

This guide helps users of the original Matlab `eq_sphere_partitions` toolbox transition to the Python `eqsp` package. While most core functionality remains identical, there are some differences in naming conventions, API structure, and usage patterns.

## 1. Quick Reference: Function Name Mapping

Most core functions retain their names. The main differences are in coordinate conversion utilities and illustration functions.

| Matlab Function | Python Function (`eqsp`) | Notes |
| :--- | :--- | :--- |
| **Partitions** | | |
| `eq_point_set` | `eq_point_set` | Identical usage. |
| `eq_regions` | `eq_regions` | Identical usage. |
| `eq_min_dist` | `eq_min_dist` | Identical usage. |
| **Utilities** | | |
| `pol2cart` | `utilities.polar2cart` | Renamed for clarity. |
| `cart2pol` | `utilities.cart2polar2` | Renamed. Handles arrays. |
| `area_of_sphere` | `utilities.area_of_sphere` | |
| **Illustrations** | | |
| `illustrate_eq_algorithm` | `illustrations.illustrate_eq_algorithm` | Moved to `illustrations` module. |
| `plot_s2_partition` | `illustrations.show_s2_partition` | 3D Plotting. |
| `project_s2_partition` | `illustrations.project_s2_partition` | 2D Projection. |

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

## 3. Module Structure
The package is organized into logical modules:
*   `eqsp.partitions`: Core partition algorithms (`eq_regions`, `eq_point_set`).
*   `eqsp.utilities`: Math helpers and coordinate conversions.
*   `eqsp.region_props`: Properties of regions (area, diameter).
*   `eqsp.point_set_props`: Properties of point sets (energy, distance).
*   `eqsp.illustrations`: Plotting tools (using Matplotlib).

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

# View them
eqsp.illustrations.show_r3_point_set(points)
```
