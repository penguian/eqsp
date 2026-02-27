"""
EQSP Tests: Histograms features

Copyright Paul Leopardi 2026
"""

import doctest
from math import pi

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from eqsp import histograms, partitions


def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(histograms)
    assert results.failed == 0


def test_eq_count_points_by_s2_region_docstrings():
    """Test function test_eq_count_points_by_s2_region_docstrings."""
    points_s = partitions.eq_point_set_polar(2, 8)

    # Test N=8
    counts = histograms.eq_count_points_by_s2_region(points_s, 8)
    assert_array_equal(counts, [1, 1, 1, 1, 1, 1, 1, 1])

    # Test N=5
    counts = histograms.eq_count_points_by_s2_region(points_s, 5)
    assert_array_equal(counts, [1, 2, 2, 2, 1])

    # Larger set
    points_s = partitions.eq_point_set_polar(2, 128)

    counts = histograms.eq_count_points_by_s2_region(points_s, 8)
    assert_array_equal(counts, [19, 15, 14, 17, 15, 14, 15, 19])

    counts = histograms.eq_count_points_by_s2_region(points_s, 5)
    assert_array_equal(counts, [19, 29, 32, 29, 19])


def test_eq_find_s2_region_docstrings():
    """Test function test_eq_find_s2_region_docstrings."""
    points_s = partitions.eq_point_set_polar(2, 8)

    regions = histograms.eq_find_s2_region(points_s, 8)
    assert_array_equal(regions, [1, 2, 3, 4, 5, 6, 7, 8])

    regions = histograms.eq_find_s2_region(points_s, 5)
    assert_array_equal(regions, [1, 2, 2, 3, 3, 4, 4, 5])


def test_in_s2_region_docstrings():
    """Test function test_in_s2_region_docstrings."""
    points_s = partitions.eq_point_set_polar(2, 8)
    s_regions = partitions.eq_regions(2, 5)

    # Check region 3 (index 2)
    region = s_regions[:, :, 2]
    in_region = histograms.in_s2_region(points_s, region)
    # Expected: points 3 and 4 (indices 3, 4) should be in region 3 (1-based)
    # eq_find_s2_region(points_s, 5) -> [1, 2, 2, 3, 3, 4, 4, 5]
    # Indices 3 and 4 correspond to '3' in the find output.
    expected = np.array([False, False, False, True, True, False, False, False])
    assert_array_equal(in_region, expected)


def test_consistency_find_and_in_region():
    """Test function test_consistency_find_and_in_region."""
    N = 10
    num_points = 50
    # Generate random points on S^2
    rng = np.random.default_rng(42)
    dim = 2
    points_s = np.zeros((dim, num_points))
    points_s[0, :] = rng.uniform(0, 2 * pi, num_points)  # Longitude
    points_s[1, :] = np.arccos(rng.uniform(-1, 1, num_points))  # Colatitude

    region_indices = histograms.eq_find_s2_region(points_s, N)
    regions = partitions.eq_regions(dim, N)

    for i in range(num_points):
        r_idx = region_indices[i]  # 1-based index from find

        # Check the reported region
        region = regions[:, :, r_idx - 1]  # 0-based index for array
        assert histograms.in_s2_region(points_s[:, i : i + 1], region)[0], (
            f"Point {i} assigned to region {r_idx} but in_s2_region mismatch"
        )

        # Check a different region (should be false, unless boundary case)
        # Choosing a region far away
        other_idx = (r_idx + N // 2) % N
        if other_idx == 0:
            other_idx = N

        # Note: strictly speaking a point could be on boundary of two regions,
        # but with random float points it's unlikely.
        other_region = regions[:, :, other_idx - 1]
        assert not histograms.in_s2_region(points_s[:, i : i + 1], other_region)[0], (
            f"Point {i} assigned to region {r_idx} but also claims "
            f"to be in region {other_idx}"
        )


def test_boundary_conditions():
    """Test function test_boundary_conditions."""
    N = 4

    # North Pole: [phi, 0]
    points_np = np.array([[0.0], [0.0]])
    r_idx = histograms.eq_find_s2_region(points_np, N)
    # Should always be region 1 (top cap)
    assert r_idx[0] == 1

    # South Pole: [phi, pi]
    points_sp = np.array([[0.0], [pi]])
    r_idx = histograms.eq_find_s2_region(points_sp, N)
    # Should always be region N (bottom cap)
    assert r_idx[0] == N


def test_invalid_inputs():
    """Test function test_invalid_inputs."""
    points_s = np.array([[0.0], [0.0]])
    # N must be positive
    with pytest.raises(ValueError):
        histograms.eq_find_s2_region(points_s, 0)

    # Dimension mismatch (regions expects dim=2 for histograms)
    bad_regions = np.zeros((3, 2, 8))  # Wrong dimension (3 instead of 2)
    with pytest.raises(ValueError):
        histograms.in_s2_region(points_s, bad_regions[:, :, 0])
