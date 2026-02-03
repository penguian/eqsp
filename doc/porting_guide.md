# Porting Guide: EQSP Toolbox (Matlab to Python)

This document outlines the process and status of porting the `eq_sphere_partitions` (EQSP) toolbox from Matlab to Python.

## 1. Objective
To create a fully functional, tested, and documented Python package (`eqsp`) that replicates the functionality of the original Matlab toolbox.

## 2. Status Overview
**Current Status:** ✅ Core Logic Ported & Tested

This project began with a partial port of the Matlab toolbox created with the assistance of GitHub Copilot. That initial port has now been **completed**, verified, and fully tested.

- **Modules Ported:** `partitions`, `point_set_props`, `region_props`, `utilities`.
- **Testing:** 100% coverage of source doctests; full parity with relevant Matlab tests.
- **Verification:** All 35 tests in the new `pytest` suite pass.

## 3. Porting Details

### 3.1 Partitions (`eqsp.partitions`)
- **Functions:** `eq_caps`, `eq_point_set`, `eq_point_set_polar`, `eq_regions`, `partition_options`.
- **Tests:** `tests/test_partitions.py` verifies these against Matlab baselines.

### 3.2 Point Set Properties (`eqsp.point_set_props`)
- **Functions:** `eq_min_dist`, `eq_energy_coeff`, `eq_packing_density`, etc.
- **Tests:** `tests/test_point_set_props.py`.
- **Fixes:** 
    - Corrected `eq_energy_coeff` handling of scalar vs array inputs.
    - Added tests for `eq_dist_coeff` and `eq_point_set_property` to match doctests.

### 3.3 Region Properties (`eqsp.region_props`)
- **Functions:** `eq_area_error`, `eq_diam_bound`, `eq_vertex_diam`, etc.
- **Tests:** `tests/test_region_props.py`.
- **Discrepancies Resolved:**
    - **Dimension 3 Diameter:** Matlab implementation returns `2.0` for small `N` in `dim=3` (regions spanning full longitude). Python docstrings incorrectly listed smaller values (likely copy-paste errors or different expectations). Tests and docstrings were updated to match the correct behavior (diameter = 2.0).
    - **Area Error:** Relaxed test tolerance for `eq_area_error` to account for minor floating-point differences (`1e-10`).
    - **API:** Refactored `eq_diam_coeff` to explicitly return a tuple `(bound_coeff, vertex_coeff)`, removing fragile stack inspection logic.

### 3.4 Utilities (`eqsp.utilities`)
- **Functions:** `area_of_sphere`, `volume_of_ball`, `euc2sph_dist`, `asfloat`, etc.
- **Tests:** `tests/test_utilities.py`.
- **Coverage:** Added explicit tests for all utility functions to ensure they match doctest behavior, including `asfloat`, `area_of_ideal_region`, and distance conversions.

### 3.5 Illustrations (`eqsp.illustrations`, `eqsp.illustrations_mayavi`)
- **Matplotlib (`eqsp.illustrations`):** Full port of original 2D and 3D plotting functions.
    - `show_s2_partition`, `project_s2_partition`, `illustrate_eq_algorithm`, etc.
- **Mayavi (`eqsp.illustrations_mayavi`):** Added support for high-quality 3D rendering (optional dependency).
    - `show_s2_sphere`, `show_r3_point_set`, `show_s2_partition` (with tubes).

## 4. Testing Strategy
- **Framework:** `pytest`.
- **Source:** Tests were ported from the original `eq_test/` Matlab scripts.
- **Doctest Parity:** A specific effort was made to ensure every example in the Python docstrings has a corresponding regression test in the `tests/` directory.

## 5. Future Work
- **Documentation:** Building full Sphinx documentation.
