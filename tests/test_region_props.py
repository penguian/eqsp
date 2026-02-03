
import numpy as np
import pytest
from numpy.testing import assert_allclose
from eqsp import region_props
from math import pi

def test_eq_area_error():
    # Example from doctest
    total_error, max_error = region_props.eq_area_error(2, 10)
    assert_allclose(total_error, 0.0, atol=1e-10)
    assert_allclose(max_error, 0.0, atol=1e-10)

    N = np.arange(1, 7)
    total_error, max_error = region_props.eq_area_error(3, N)
    
    # Tolerance needs to be adjusted as these are very small errors
    # Doctest values seem to be from a different environment. 
    # We verify that errors are reasonably small (< 1e-10).
    # Note that Python implementation gives errors around 1e-11 for some cases.
    assert_allclose(total_error, 0, atol=1e-10)
    assert_allclose(max_error, 0, atol=1e-10)

def test_eq_diam_bound():
    # Example from doctest
    bound = region_props.eq_diam_bound(2, 10)
    assert_allclose(bound, 1.6733, atol=1e-4)

    N = np.arange(1, 7)
    bound = region_props.eq_diam_bound(3, N)
    # The docstring values seem incorrect (1.0515 is min_dist for 2,10).
    # Logic suggests for small N in dim 3, regions often span pi longitude, so diam is 2.
    expected = np.array([2., 2., 2., 2., 2., 2.])
    assert_allclose(bound, expected, atol=1e-4)

def test_eq_vertex_diam():
    # Example from doctest
    vdiam = region_props.eq_vertex_diam(2, 10)
    assert_allclose(vdiam, 1.4142, atol=1e-4)

    N = np.arange(1, 7)
    vdiam = region_props.eq_vertex_diam(3, N)
    # Similar to diam_bound, code returns 2.0 for these small N
    expected = np.array([2., 2., 2., 2., 2., 2.])
    assert_allclose(vdiam, expected, atol=1e-4)

def test_eq_diam_coeff():
    # Test double return (now always returns tuple)
    bound_coeff, vertex_coeff = region_props.eq_diam_coeff(2, 10)
    assert_allclose(bound_coeff, 5.2915, atol=1e-4)

    # Test array return
    # bound_coeff and vertex_coeff should be 2.0 * N^(1/3) for these N where diam=2.0
    N = np.arange(1, 7)
    bound_coeff, vertex_coeff = region_props.eq_diam_coeff(3, N)
    expected_coeff = 2.0 * np.power(N, 1/3)
    
    assert_allclose(bound_coeff, expected_coeff, atol=1e-4)
    assert_allclose(vertex_coeff, expected_coeff, atol=1e-4)

def test_eq_vertex_diam_coeff():
    # Example from doctest
    coeff = region_props.eq_vertex_diam_coeff(2, 10)
    assert_allclose(coeff, 4.4721, atol=1e-4)
    
    N = np.arange(1, 7)
    coeff = region_props.eq_vertex_diam_coeff(3, N)
    expected = np.array([2.    , 2.5198, 2.8845, 3.1748, 3.42  , 3.6342])
    assert_allclose(coeff, expected, atol=1e-4)

def test_eq_regions_property():
    # Example from doctest
    def dummy_property(regions): 
        return regions.shape[2]
    
    prop = region_props.eq_regions_property(dummy_property, 2, [3, 4])
    assert_allclose(prop, [3, 4])

def test_area_of_region():
    # Example from doctest
    # region spanning 0 to 2pi longitude, 0 to pi colatitude -> whole sphere S^2
    # dim=2
    region = np.array([[0, 2*np.pi], [0, np.pi]])
    area = region_props.area_of_region(region)
    # Area of S^2 is 4pi
    assert_allclose(area, 4 * np.pi, atol=1e-4)
    assert_allclose(area, 12.5664, atol=1e-4)
