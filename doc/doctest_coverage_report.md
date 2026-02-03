# Test vs Doctest Correspondence Report

This report analyzes whether all doctests in the Python source code have corresponding tests in the `tests/` directory.

## Summary

**Status:** ✅ **100% Covered**

All major functions and their doctests are now covered by the `pytest` suite.

## Detailed Analysis

### 1. `eqsp.utilities`
| Function | Doctest Exists? | Pytest Exists? | Status |
| :--- | :--- | :--- | :--- |
| `asfloat` | ✅ | ✅ | Covered |
| `cart2polar2` | ✅ | ✅ (`test_polar2cart_cart2polar_inversion`) | Covered |
| `polar2cart` | ✅ | ✅ | Covered |
| `euc2sph_dist` | ✅ | ✅ | Covered |
| `sph2euc_dist` | ✅ | ✅ | Covered |
| `euclidean_dist` | ✅ | ✅ | Covered |
| `spherical_dist` | ✅ | ✅ | Covered |
| `area_of_sphere` | ✅ | ✅ | Covered |
| `volume_of_ball` | ✅ | ✅ | Covered |
| `area_of_ideal_region` | ✅ | ✅ | Covered |
| `ideal_collar_angle` | ✅ | ✅ | Covered |
| `area_of_cap` | ✅ | ✅ | Covered |
| `sradius_of_cap` | ✅ | ✅ | Covered |
| `area_of_collar` | ✅ | ✅ | Covered |

### 2. `eqsp.region_props`
| Function | Doctest Exists? | Pytest Exists? | Status |
| :--- | :--- | :--- | :--- |
| `eq_area_error` | ✅ | ✅ | Covered |
| `eq_diam_bound` | ✅ | ✅ | Covered |
| `eq_vertex_diam` | ✅ | ✅ | Covered |
| `eq_diam_coeff` | ✅ | ✅ | Covered |
| `eq_vertex_diam_coeff` | ✅ | ✅ | Covered |
| `eq_regions_property` | ✅ | ✅ | Covered |
| `area_of_region` | ✅ | ✅ | Covered |

### 3. `eqsp.point_set_props`
| Function | Doctest Exists? | Pytest Exists? | Status |
| :--- | :--- | :--- | :--- |
| `eq_min_dist` | ✅ | ✅ | Covered |
| `calc_dist_coeff` | ✅ | ✅ | Covered |
| `eq_energy_coeff` | ✅ | ✅ | Covered |
| `eq_energy_dist` | ✅ | ✅ | Covered |
| `eq_packing_density` | ✅ | ✅ | Covered |
| `sphere_int_energy` | ✅ | ✅ | Covered |
| `point_set_dist_coeff` | ✅ | ✅ (via `test_point_set_dist_and_energy`) | Covered |
| `point_set_energy_coeff` | ✅ | ✅ (via `test_point_set_dist_and_energy`) | Covered |
| `point_set_energy_dist` | ✅ | ✅ | Covered |
| `eq_dist_coeff` | ✅ | ✅ | Covered |
| `eq_point_set_property` | ✅ | ✅ | Covered |

### 4. `eqsp.partitions`
| Function | Doctest Exists? | Pytest Exists? | Status |
| :--- | :--- | :--- | :--- |
| `eq_caps` | ✅ | ✅ | Covered |
| `eq_point_set` | ✅ | ✅ | Covered |
| `eq_point_set_polar` | ✅ | ✅ | Covered |
| `eq_regions` | ✅ | ✅ | Covered |
| `partition_options` | ✅ | ✅ | Covered |

## Conclusion

The test suite now has full parity with the doctests provided in the source code.
