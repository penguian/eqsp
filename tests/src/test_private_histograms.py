"""
EQSP Tests: Private Histograms features

Copyright Paul Leopardi 2026
"""

# pylint: disable=import-outside-toplevel

import doctest
from math import pi

import numpy as np

import eqsp
from eqsp._private import _histograms
from eqsp._private._histograms import lookup_s2_region, lookup_table


def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(_histograms)
    assert results.failed == 0


def test_lookup_table_boundaries():
    """Test function test_lookup_table_boundaries."""
    table = [-100.0, -70.0, 2.5, 75.0, 125.7]

    # Normal points between table values
    y_normal = [-80.0, 0.0, 50.0, 100.0]
    expected_normal = [1, 2, 3, 4]

    # Points exactly hitting the table values
    y_exact = [-100.0, -70.0, 2.5, 75.0, 125.7]
    # np.searchsorted(side='right') means exactly hits go into the next bin index
    # So if value is exactly table[0] (-100.0), searchsorted returns 1
    # If exactly table[4] (125.7), searchsorted returns 5, now capped at 4.
    expected_exact = [1, 2, 3, 4, 4]

    # Out of bounds points
    y_out = [-200.0, 200.0]
    # Below minimum gets capped to 0.
    # Above maximum gets capped to len(table) - 1 (4).
    # lookup_s2_region relies on this behavior to map points near pi to the last cap.
    expected_out = [0, 4]

    y_all = y_normal + y_exact + y_out
    expected_all = expected_normal + expected_exact + expected_out

    np.testing.assert_array_equal(lookup_table(table, y_all), expected_all)


def test_lookup_s2_region_vectorization_fidelity():
    """Test function test_lookup_s2_region_vectorization_fidelity."""
    N = 33
    dim = 2

    s_regions = eqsp.eq_regions(dim, N)
    s_cap, n_regions = eqsp.eq_caps(dim, N)
    c_regions = np.cumsum(n_regions)

    # Test 1000 randomly scattered points
    rng = np.random.default_rng(1234)
    points_1 = rng.uniform(0, 2 * pi, 1000)
    points_2 = np.arccos(rng.uniform(-1, 1, 1000))
    random_points = np.vstack((points_1, points_2))

    # Also explicitly test points falling directly on boundaries
    # s_cap boundaries (colatitudes)
    cap_points = np.zeros((2, len(s_cap)))
    cap_points[1, :] = s_cap

    # longitude boundaries
    long_points = np.zeros((2, s_regions.shape[2]))
    long_points[0, :] = s_regions[0, 0, :]
    long_points[1, :] = s_regions[1, 0, :] + 0.1  # inside the colatitude band

    all_points = np.hstack((random_points, cap_points, long_points))

    # 1. Run the original sequential logic
    r_idx_original = np.zeros(all_points.shape[1], dtype=int)

    n_caps = len(s_cap)
    n_regions_total = s_regions.shape[2]

    for p_idx in range(all_points.shape[1]):
        c_idx = lookup_table(s_cap, all_points[1, p_idx])
        if 0 < c_idx < n_caps - 1:
            min_r_idx = int(c_regions[c_idx - 1]) + 1
            max_r_idx = int(c_regions[c_idx])
            s_longs = s_regions[0, :, min_r_idx - 1 : max_r_idx].copy()
            if s_longs[0, 0] >= 2 * np.pi:
                s_longs[:, 0] -= 2 * np.pi
            n_longs = s_longs.shape[1]
            l_idx = lookup_table(s_longs[1, :], all_points[0, p_idx]) % n_longs
            if all_points[0, p_idx] < s_longs[0, 0]:
                l_idx = n_longs - 1
            r_idx_original[p_idx] = min_r_idx + l_idx
        elif c_idx == 0:
            r_idx_original[p_idx] = 1
        elif c_idx >= n_caps - 1:
            r_idx_original[p_idx] = n_regions_total
        else:
            r_idx_original[p_idx] = 0

    # 2. Run the (potentially vectorized) module logic
    r_idx_module = lookup_s2_region(all_points, s_regions, s_cap, c_regions)

    np.testing.assert_array_equal(r_idx_original, r_idx_module)
