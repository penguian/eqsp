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

    # 1. Run the reference Translated Domain logic
    cap_bound_lats = np.round(s_cap, 12)
    c_starts = np.concatenate(([0], c_regions[:-1]))
    n_detection = len(cap_bound_lats)

    r_idx_original = np.zeros(all_points.shape[1], dtype=int)

    for p_idx in range(all_points.shape[1]):
        pts_lat = np.round(all_points[1, p_idx], 12)
        pts_long = all_points[0, p_idx]

        c_idx = np.searchsorted(cap_bound_lats, pts_lat, side="left")
        min_r_offset = int(c_starts[c_idx])

        if c_idx < n_detection - 1:
            n_longs = int(c_starts[c_idx + 1]) - min_r_offset
        else:
            n_longs = s_regions.shape[2] - min_r_offset

        if n_longs > 1:
            s_longs = s_regions[0, :, min_r_offset : min_r_offset + n_longs]
            phi0 = s_longs[0, 0]
            two_pi = 2 * np.pi

            pts_long_translated = (pts_long - phi0) % two_pi
            ends_translated = (s_longs[1, :] - phi0) % two_pi
            if ends_translated[-1] <= 1e-15:
                ends_translated[-1] = two_pi

            l_idx = np.searchsorted(ends_translated, pts_long_translated, side="left")
            if l_idx >= n_longs:
                l_idx = 0
            r_idx_original[p_idx] = min_r_offset + 1 + l_idx
        else:
            r_idx_original[p_idx] = min_r_offset + 1

    # 2. Run the (potentially vectorized) module logic
    r_idx_module = lookup_s2_region(all_points, s_regions, s_cap, c_regions)

    np.testing.assert_array_equal(r_idx_original, r_idx_module)
