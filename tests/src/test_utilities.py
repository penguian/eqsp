"""
PyEQSP Tests: Utilities features

Copyright Paul Leopardi 2026
"""

import doctest

import numpy as np
from numpy.testing import assert_allclose

from eqsp import utilities


def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(utilities)
    assert results.failed == 0


def test_polar2cart_cart2polar_inversion():
    """Test function test_polar2cart_cart2polar_inversion."""
    # cart2polar2 is only for S^2 (3D Cartesian coordinates)

    # Test S^2 round trip
    # Random points in S^2
    rng = np.random.default_rng(42)
    dim_s = 2
    num_points = 5

    # Generate random polar coordinates for S^2
    # s[0] (phi) in [0, 2pi), s[1] (theta) in [0, pi]
    s_polar = np.zeros((dim_s, num_points))
    s_polar[0, :] = rng.random(num_points) * 2 * np.pi
    s_polar[1, :] = rng.random(num_points) * np.pi

    x_cart = utilities.polar2cart(s_polar)

    # Check cart2polar2 inversion
    s_polar_back = utilities.cart2polar2(x_cart)

    # Handle wrap around for phi close to 0/2pi
    diff = np.abs(s_polar - s_polar_back)
    diff[0] = np.minimum(diff[0], 2 * np.pi - diff[0])

    assert_allclose(diff, 0, atol=1e-7)


def test_polar2cart_general():
    """Test function test_polar2cart_general."""
    # For generic dimensions, we only test that polar2cart produces points
    # on sphere (norm=1)
    for dim in range(1, 5):
        rng = np.random.default_rng(42)
        n_points = 10
        s = rng.random((dim, n_points)) * np.pi
        # First angle usually [0, 2pi)
        s[0, :] *= 2.0

        x = utilities.polar2cart(s)
        # Check shape
        assert x.shape == (dim + 1, n_points)
        # Check unit norm
        norms = np.linalg.norm(x, axis=0)
        assert_allclose(norms, 1.0, atol=1e-10)
