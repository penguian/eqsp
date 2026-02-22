import numpy as np
import pytest
import doctest
from numpy.testing import assert_allclose
from eqsp import point_set_props


def test_doctests():
    """Run doctests for point_set_props."""
    import doctest
    results = doctest.testmod(point_set_props)
    assert results.failed == 0


def test_eq_min_dist():
    # Test based on doctest
    dist = point_set_props.eq_min_dist(2, 10)
    assert_allclose(dist, 1.0515, atol=1e-4)

    N = np.arange(1, 7)
    dist = point_set_props.eq_min_dist(3, N)
    expected_dist = np.array([2.0, 2.0, 1.4142, 1.4142, 1.4142, 1.4142])
    assert_allclose(dist, expected_dist, atol=1e-4)


def test_calc_dist_coeff():
    # Example from doctest
    N = np.arange(2, 7)
    # Reconstruct dist manually or assume eq_min_dist works
    dist = point_set_props.eq_min_dist(2, N)
    coeff = point_set_props.calc_dist_coeff(2, N, dist)
    expected_coeff = np.array(
        [2.82842712, 2.44948974, 2.82842712, 3.16227766, 3.46410162]
    )
    assert_allclose(coeff, expected_coeff, atol=1e-5)


def test_eq_energy_coeff():
    # Example from doctest
    coeff = point_set_props.eq_energy_coeff(2, 10)
    assert_allclose(coeff, -0.5460877923347524, atol=1e-5)

    N = np.arange(1, 7)
    coeff = point_set_props.eq_energy_coeff(3, N)
    expected_coeff = np.array([-0.5, -0.5512, -0.5208, -0.5457, -0.5472, -0.5679])
    assert_allclose(coeff, expected_coeff, atol=1e-4)

    # With s=0
    coeff = point_set_props.eq_energy_coeff(2, N, 0)
    expected_coeff_s0 = np.array([0.0, -0.2213, -0.1569, -0.2213, -0.2493, -0.2569])
    assert_allclose(coeff, expected_coeff_s0, atol=1e-4)


def test_eq_energy_dist():
    # Example from doctest
    energy = point_set_props.eq_energy_dist(2, 10)
    assert_allclose(energy, 32.7312, atol=1e-4)

    # With returning dist too
    N = np.arange(1, 7)
    energy, dist = point_set_props.eq_energy_dist(3, N, 0)
    expected_energy = np.array([0.0, -0.6931, -1.3863, -2.7726, -4.1589, -6.2383])
    expected_dist = np.array([2.0, 2.0, 1.4142, 1.4142, 1.4142, 1.4142])

    assert_allclose(energy, expected_energy, atol=1e-4)
    assert_allclose(dist, expected_dist, atol=1e-4)


def test_eq_packing_density():
    # Example from doctest
    density = point_set_props.eq_packing_density(2, 10)
    assert_allclose(density, 0.7467459582397998, atol=1e-6)

    N = np.arange(1, 7)
    density = point_set_props.eq_packing_density(3, N)
    expected_density = np.array([1.0, 1.0, 0.2725, 0.3634, 0.4542, 0.5451])
    assert_allclose(density, expected_density, atol=1e-4)


def test_sphere_int_energy():
    # Example from doctest
    energy = point_set_props.sphere_int_energy(2, 0)
    assert_allclose(energy, -0.1931471805599453, atol=1e-6)


def test_point_set_dist_and_energy():
    # Example from doctest
    x = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [1, 0, 0, -1]])

    # point_set_energy_dist
    e, d = point_set_props.point_set_energy_dist(x, 2)
    assert_allclose(e, 2.5)
    assert_allclose(d, 1.41421356)

    e0, d0 = point_set_props.point_set_energy_dist(x, 0)
    assert_allclose(e0, -2.77258872)
    assert_allclose(d0, 1.41421356)

    # point_set_dist_coeff
    coeff = point_set_props.point_set_dist_coeff(x)
    assert_allclose(coeff, 2.8284271247)


def test_eq_dist_coeff():
    N = np.arange(2, 7)
    coeff = point_set_props.eq_dist_coeff(2, N)
    expected_coeff = np.array([2.8284, 2.4495, 2.8284, 3.1623, 3.4641])
    assert_allclose(coeff, expected_coeff, atol=1e-4)


def test_eq_point_set_property():
    res = point_set_props.eq_point_set_property(
        point_set_props.point_set_min_dist, 2, 10
    )
    assert_allclose(res, 1.0515, atol=1e-4)
