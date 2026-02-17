import numpy as np
import pytest
from numpy.testing import assert_allclose
from eqsp import utilities


def test_area_of_cap():
    # Matlab: a = area_of_cap(2,pi/2) -> 2*pi for hemisphere in 3D (S^2)
    # area_of_cap takes (dim_s, angle). dim_s=2 means S^2.
    # area of S^2 is 4pi. hemisphere is 2pi.
    a = utilities.area_of_cap(2, np.pi / 2)
    assert_allclose(a, 2 * np.pi)

    # Matlab: a = area_of_cap(3,0:pi/4:pi)
    # Check simple values for S^3 (dim=3)
    # area of S^3 is 2*pi^2.
    assert_allclose(utilities.area_of_cap(3, 0), 0.0, atol=1e-10)
    assert_allclose(utilities.area_of_cap(3, np.pi), 2 * np.pi**2)


def test_area_of_collar():
    # Matlab: area = area_of_collar(2,0:2,1:3)
    # This implies a_top and a_bot arrays.
    # Check simpler case first.
    # S^2 collar from 0 to pi/2 -> 2pi
    area = utilities.area_of_collar(2, 0.0, np.pi / 2)
    assert_allclose(area, 2 * np.pi)

    # Vectorized inputs
    # S^2, collar1: 0->pi/2, collar2: pi/2->pi
    a_top = np.array([0.0, np.pi / 2])
    a_bot = np.array([np.pi / 2, np.pi])
    areas = utilities.area_of_collar(2, a_top, a_bot)
    assert_allclose(areas, [2 * np.pi, 2 * np.pi])


def test_volume_of_ball():
    # Matlab: volume = volume_of_ball(1:7)
    # Check known values
    # dim=1 (line segment [-1,1]): vol=2
    assert_allclose(utilities.volume_of_ball(1), 2.0)
    # dim=2 (disk): vol=pi
    assert_allclose(utilities.volume_of_ball(2), np.pi)
    # dim=3 (ball): vol=4/3 pi
    assert_allclose(utilities.volume_of_ball(3), 4 / 3 * np.pi)
    # dim=4: vol=1/2 pi^2
    assert_allclose(utilities.volume_of_ball(4), 0.5 * np.pi**2)


def test_area_of_sphere():
    # dim=1 (S^1, circle): 2pi
    assert_allclose(utilities.area_of_sphere(1), 2 * np.pi)
    # dim=2 (S^2, sphere): 4pi
    assert_allclose(utilities.area_of_sphere(2), 4 * np.pi)
    # dim=3 (S^3): 2pi^2
    assert_allclose(utilities.area_of_sphere(3), 2 * np.pi**2)


def test_polar2cart_cart2polar_inversion():
    # cart2polar2 is only for S^2 (3D Cartesian coordinates)

    # Test S^2 round trip
    # Random points in S^2
    rng = np.random.default_rng(42)
    dim_s = 2
    dim_x = 3
    num_points = 5

    # Generate random polar coordinates for S^2
    # s[0] (phi) in [0, 2pi), s[1] (theta) in [0, pi]
    s_polar = np.zeros((dim_s, num_points))
    s_polar[0, :] = rng.random(num_points) * 2 * np.pi
    s_polar[1, :] = rng.random(num_points) * np.pi

    x_cart = utilities.polar2cart(s_polar)

    # Check cart2polar2 inversion
    s_polar_back = utilities.cart2polar2(x_cart)

    # Note: angles might differ by 2pi or be close to boundaries
    # Using small epsilon for finite precision
    # Also handle standard range differences if any.
    # utilities.cart2polar2 returns phi in [0, 2pi), theta in [0, pi]
    # which matches our generation.

    # Handle wrap around for phi close to 0/2pi
    diff = np.abs(s_polar - s_polar_back)
    # If diff[0] ~ 2pi, it's fine.
    diff[0] = np.minimum(diff[0], 2 * np.pi - diff[0])

    assert_allclose(diff, 0, atol=1e-7)


def test_polar2cart_general():
    # For generic dimensions, we only test that polar2cart produces points on sphere (norm=1)
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


def test_euc2sph_dist():
    # Test scalar
    # e = 2 -> s = pi
    assert_allclose(utilities.euc2sph_dist(2.0), np.pi)
    # e = 0 -> s = 0
    assert_allclose(utilities.euc2sph_dist(0.0), 0.0)
    # e = sqrt(2) -> s = pi/2 (chord of 90 degrees)
    assert_allclose(utilities.euc2sph_dist(np.sqrt(2)), np.pi / 2)

    # Test array
    e = np.array([0, np.sqrt(2), 2.0])
    s = utilities.euc2sph_dist(e)
    assert_allclose(s, [0, np.pi / 2, np.pi])


def test_sph2euc_dist():
    # Test scalar
    # s = pi -> e = 2
    assert_allclose(utilities.sph2euc_dist(np.pi), 2.0)
    # s = 0 -> e = 0
    assert_allclose(utilities.sph2euc_dist(0.0), 0.0)
    # s = pi/2 -> e = sqrt(2)
    assert_allclose(utilities.sph2euc_dist(np.pi / 2), np.sqrt(2))

    # Test array
    s = np.array([0, np.pi / 2, np.pi])
    e = utilities.sph2euc_dist(s)
    assert_allclose(e, [0, np.sqrt(2), 2.0])

    # Round trip test
    rng = np.random.default_rng(42)
    s_orig = rng.random(10) * np.pi
    e = utilities.sph2euc_dist(s_orig)
    s_new = utilities.euc2sph_dist(e)
    assert_allclose(s_new, s_orig, atol=1e-10)


def test_euclidean_dist():
    # Points on unit sphere
    # North pole (0,0,1) and South pole (0,0,-1) -> dist 2
    x = np.array([[0], [0], [1]])
    y = np.array([[0], [0], [-1]])
    assert_allclose(utilities.euclidean_dist(x, y), [2.0])

    # Same point -> dist 0
    assert_allclose(utilities.euclidean_dist(x, x), [0.0])

    # Orthogonal points (1,0,0) and (0,1,0) -> dist sqrt(2)
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [1], [0]])
    assert_allclose(utilities.euclidean_dist(x, y), [np.sqrt(2)])

    # Vectorized
    # Col 1: same, Col 2: orth, Col 3: anti-podal
    x = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # dists: 0, sqrt(2), 2
    expected = [0.0, np.sqrt(2), 2.0]
    assert_allclose(utilities.euclidean_dist(x, y), expected)


def test_spherical_dist():
    # Points on unit sphere
    # North pole (0,0,1) and South pole (0,0,-1) -> dist pi
    x = np.array([[0], [0], [1]])
    y = np.array([[0], [0], [-1]])
    assert_allclose(utilities.spherical_dist(x, y), [np.pi])

    # Same point -> dist 0
    assert_allclose(utilities.spherical_dist(x, x), [0.0])

    # Orthogonal points (1,0,0) and (0,1,0) -> dist pi/2
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [1], [0]])
    assert_allclose(utilities.spherical_dist(x, y), [np.pi / 2])

    # Vectorized
    # Col 1: same, Col 2: orth, Col 3: anti-podal
    x = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # dists: 0, pi/2, pi
    expected = [0.0, np.pi / 2, np.pi]
    assert_allclose(utilities.spherical_dist(x, y), expected, atol=1e-10)


def test_asfloat():
    # Test scalar
    x0 = 12.789
    a0 = utilities.asfloat(x0)
    assert isinstance(a0, float)
    assert_allclose(a0, 12.789)

    # Test single element list/array -> float
    x1 = [[22.546]]
    a1 = utilities.asfloat(x1)
    assert isinstance(a1, float)
    assert_allclose(a1, 22.546)

    # Test array
    x2 = [12.789, 22.546]
    a2 = utilities.asfloat(x2)
    assert isinstance(a2, np.ndarray)
    assert_allclose(a2, [12.789, 22.546])


def test_area_of_ideal_region():
    # Example from doctest
    # dim=3, N=[1..6]
    areas = utilities.area_of_ideal_region(3, np.arange(1, 7))
    expected = np.array([19.7392, 9.8696, 6.5797, 4.9348, 3.9478, 3.2899])
    assert_allclose(areas, expected, atol=1e-4)

    # Check trivial case: dim=2, N=1 -> 4pi
    assert_allclose(utilities.area_of_ideal_region(2, 1), 4 * np.pi)


def test_ideal_collar_angle():
    # Example from doctest
    # dim=2, N=10
    angle = utilities.ideal_collar_angle(2, 10)
    assert_allclose(angle, 1.121, atol=1e-3)

    # dim=3, N=[1..6]
    angles = utilities.ideal_collar_angle(3, np.arange(1, 7))
    expected = np.array([2.7026, 2.145, 1.8739, 1.7025, 1.5805, 1.4873])
    assert_allclose(angles, expected, atol=1e-4)


def test_sradius_of_cap():
    # Example from doctest
    # dim=2, area = 2pi -> s_cap = pi/2 (hemisphere)
    area = utilities.area_of_sphere(2) / 2
    s_cap = utilities.sradius_of_cap(2, area)
    assert_allclose(s_cap, np.pi / 2)

    # dim=3
    areas = np.linspace(0, 4, 5) * utilities.area_of_sphere(3) / 4
    s_caps = utilities.sradius_of_cap(3, areas)
    expected = np.array([0.0, 1.1549, 1.5708, 1.9867, 3.1416])
    assert_allclose(s_caps, expected, atol=1e-4)
