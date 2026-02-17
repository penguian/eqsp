import numpy as np
import pytest
from numpy.testing import assert_allclose
from eqsp import partitions
from math import pi


def test_eq_caps():
    # Test dim=2, N=10 (from docstring/Matlab)
    s_cap, n_regions = partitions.eq_caps(2, 10)
    expected_s_cap = np.array([0.6435, 1.5708, 2.4981, 3.1416])
    expected_n_regions = np.array([1.0, 4.0, 4.0, 1.0])

    assert_allclose(s_cap, expected_s_cap, atol=1e-4)
    assert_allclose(n_regions, expected_n_regions)
    assert np.sum(n_regions) == 10

    # Test dim=3, N=6
    s_cap, n_regions = partitions.eq_caps(3, 6)
    expected_s_cap = np.array([0.9845, 2.1571, 3.1416])
    expected_n_regions = np.array([1.0, 4.0, 1.0])

    assert_allclose(s_cap, expected_s_cap, atol=1e-4)
    assert_allclose(n_regions, expected_n_regions)
    assert np.sum(n_regions) == 6

    # Test edge case dim=1 (circle)
    # Should divide into N equal sectors
    N = 4
    s_cap, n_regions = partitions.eq_caps(1, N)
    # n_regions should be all 1s
    assert_allclose(n_regions, np.ones(N))
    # s_cap should be 2pi/N * sector
    expected_s_cap = np.arange(1, N + 1) * 2 * pi / N
    assert_allclose(s_cap, expected_s_cap)

    # Test edge case N=1
    s_cap, n_regions = partitions.eq_caps(2, 1)
    assert_allclose(s_cap, [pi])
    assert_allclose(n_regions, [1])


def test_eq_point_set_polar():
    # Test dim=2, N=4
    # Expected shape (2, 4)
    dim = 2
    N = 4
    points_s = partitions.eq_point_set_polar(dim, N)
    assert points_s.shape == (dim, N)

    # Check bounds
    # theta (last row) should be in [0, pi]
    assert np.all(points_s[-1, :] >= 0)
    assert np.all(points_s[-1, :] <= pi)

    # Test dim=3, N=10
    dim = 3
    N = 10
    points_s = partitions.eq_point_set_polar(dim, N)
    assert points_s.shape == (dim, N)


def test_eq_point_set():
    # Test dim=2, N=4
    dim = 2
    N = 4
    points_x = partitions.eq_point_set(dim, N)
    assert points_x.shape == (dim + 1, N)

    # Check unit norm
    norms = np.linalg.norm(points_x, axis=0)
    assert_allclose(norms, 1.0, atol=1e-10)


def test_eq_regions():
    # Test dim=2, N=4
    dim = 2
    N = 4
    regions = partitions.eq_regions(dim, N)
    # shape: (dim, 2, N)
    assert regions.shape == (dim, 2, N)

    # Test N=1
    regions = partitions.eq_regions(dim, 1)
    assert regions.shape == (dim, 2, 1)
    # Should correspond to whole sphere
    # dim=2: [0, 2pi], [0, pi]
    assert_allclose(regions[0, :, 0], [0, 2 * pi])
    assert_allclose(regions[1, :, 0], [0, pi])
