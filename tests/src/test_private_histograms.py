"""
PyEQSP Tests: Private Histograms

Copyright Paul Leopardi 2026
"""

# pylint: disable=import-outside-toplevel

import doctest
from math import pi

import numpy as np

import eqsp
from eqsp._private import _histograms
from eqsp._private._histograms import lookup_s2_region


def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(_histograms)
    assert results.failed == 0


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

    # 1. Run the reference Brute Force logic
    # (Checking every point against every region for membership)
    r_idx_brute = np.zeros(all_points.shape[1], dtype=int)
    all_regions = eqsp.eq_regions(dim, N)

    for p_idx in range(all_points.shape[1]):
        point = all_points[:, p_idx : p_idx + 1]
        for r_num in range(N):
            region = all_regions[:, :, r_num]
            if eqsp.histograms.in_s2_region(point, region)[0]:
                r_idx_brute[p_idx] = r_num + 1
                break

    # 2. Run the (potentially vectorized) module logic
    r_idx_module = lookup_s2_region(all_points, s_regions, s_cap, c_regions)

    np.testing.assert_array_equal(r_idx_brute, r_idx_module)
