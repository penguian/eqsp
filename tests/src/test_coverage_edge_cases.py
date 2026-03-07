# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=import-outside-toplevel

import numpy as np
import pytest

from eqsp._private._histograms import lookup_s2_region, lookup_table
from eqsp._private._partitions import circle_offset, rot3
from eqsp._private._partitions import polar_colat as priv_polar_colat
from eqsp.partitions import eq_caps, eq_point_set_polar, eq_regions
from eqsp.point_set_props import (
    calc_energy_coeff,
    point_set_energy_dist,
    point_set_min_dist,
    sphere_int_energy,
)
from eqsp.region_props import area_of_region, eq_area_error, eq_diam_coeff
from eqsp.utilities import area_of_cap, sradius_of_cap


def test_histogram_errors():
    # lookup_s2_region: Mismatch between s_cap and c_regions
    with pytest.raises(
        ValueError, match="Mismatch between length of s_cap and c_regions"
    ):
        lookup_s2_region(np.zeros((2,1)), np.zeros((2,2,1)), [1, 2], [1])

    # lookup_s2_region: Mismatch between c_regions[-1] and N
    with pytest.raises(ValueError, match="Mismatch between c_regions"):
        lookup_s2_region(np.zeros((2,1)), np.zeros((2,2,10)), [1], [5])

    # lookup_s2_region: Empty points
    pts = np.zeros((2,0))
    res = lookup_s2_region(pts, np.zeros((2,2,1)), [1], [1])
    assert res.size == 0

    # lookup_table: Decreasing table
    with pytest.raises(NotImplementedError):
        lookup_table([10, 5, 1], 7)

def test_partition_errors():
    # circle_offset: n_top/bot positive
    with pytest.raises(ValueError, match="positive integers"):
        circle_offset(0, 4)

    # rot3: axis 4
    with pytest.raises(ValueError, match="axis must be 1, 2, or 3"):
        rot3(4, 0.1)

    # eq_caps: invalid dim/N
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        eq_caps(0, 10)
    with pytest.raises(ValueError, match="N must be a positive integer"):
        eq_caps(2, 0)

    # eq_point_set_polar: invalid dim/N
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        eq_point_set_polar(0.5, 10)
    with pytest.raises(ValueError, match="N must be a positive integer"):
        eq_point_set_polar(2, -1)

def test_props_errors():
    # eq_area_error: None
    with pytest.raises(ValueError):
        eq_area_error(None, None)

    # eq_diam_coeff: None
    with pytest.raises(ValueError):
        eq_diam_coeff(None, None)

def test_s1_logic():
    # dim=1 is circle partitioning
    # area_of_cap(1, cap) -> s_cap
    assert np.isclose(area_of_cap(1, 0.5), 1.0)
    # sradius_of_cap(1, area) -> area
    assert np.isclose(sradius_of_cap(1, 0.7), 0.35)

    # eq_caps for dim=1
    s_cap, n_regions = eq_caps(1, 4)
    assert len(s_cap) == 4
    assert np.all(n_regions == 1)

    # area_of_region for dim=1
    # region shape (1, 2)
    reg = np.array([[0, 1.0]])
    assert np.isclose(area_of_region(reg), 1.0)
    # wraps
    reg_wrap = np.array([[0, 0]]) # s_bot=0 -> 2*pi
    assert np.isclose(area_of_region(reg_wrap), 2*np.pi)
    # top == bot != 0
    reg_zero = np.array([[1.0, 1.0]])
    assert np.isclose(area_of_region(reg_zero), 2*np.pi)

def test_high_dim_and_n1():
    # polar_colat N=1
    assert np.isclose(priv_polar_colat(2, 1), np.pi)

    # point_set_energy_dist N=1
    e, d = point_set_energy_dist(np.zeros((3,1)), s=2)
    assert e == 0.0
    assert d == 2.0

    # point_set_min_dist N=1
    assert point_set_min_dist(np.zeros((3,1))) == 2.0

    # eq_min_dist dim=1, N>1
    from eqsp.point_set_props import eq_min_dist
    assert np.isclose(eq_min_dist(1, 4), 2 * np.sin(np.pi / 4))

    # sphere_int_energy dim=1 path
    assert sphere_int_energy(1, 0) == 0
    assert sphere_int_energy(1, 0.5) != 0
    assert np.isclose(
        calc_energy_coeff(1, 4, 0, 10), 10.0 / (4.0 * np.log(4.0))
    )  # internal path

def test_extra_offset_paths():
    # Exercise extra_offset in eq_regions for dim=2/3
    # This hits lines 269, 428, etc.
    regs = eq_regions(3, 8, extra_offset=True)
    assert isinstance(regs, tuple)
    assert regs[0].shape == (3, 2, 8)

    # dim > 3 extra_offset=True (should be ignored)
    pts = eq_point_set_polar(4, 16, extra_offset=True)
    assert pts.shape == (4, 16)

def test_multicollar_cache():
    # Exercise use_cache=False path by using dim=2
    # Let's ensure we hit the True/False paths
    eq_point_set_polar(2, 100) # dim=2 -> use_cache=True (line 226)
    eq_regions(2, 50)          # dim=2 -> use_cache=False (line 383)

def test_show_progress():
    # Hit all the progress print statements taking N as an array > 1 length
    n_arr = np.array([2, 3])
    eq_area_error(2, n_arr, show_progress=True)
    eq_diam_coeff(2, n_arr, show_progress=True)
    from eqsp.region_props import eq_diam_bound, eq_vertex_diam, eq_vertex_diam_coeff
    eq_diam_bound(2, n_arr, show_progress=True)
    eq_vertex_diam_coeff(2, n_arr, show_progress=True)
    eq_vertex_diam(2, n_arr, show_progress=True)

    from eqsp.point_set_props import (
        eq_dist_coeff,
        eq_energy_coeff,
        eq_energy_dist,
        eq_min_dist,
        eq_packing_density,
        eq_point_set_property,
    )
    eq_energy_dist(2, n_arr, show_progress=True)
    eq_min_dist(2, n_arr, show_progress=True)
    eq_packing_density(2, n_arr, show_progress=True)
    eq_energy_coeff(2, n_arr, show_progress=True)
    eq_dist_coeff(2, n_arr, show_progress=True)
    def f(_x):
        return 0.0
    eq_point_set_property(f, 2, n_arr, show_progress=True)

def test_spherical_dist_error():
    from eqsp.utilities import euclidean_dist, spherical_dist
    with pytest.raises(ValueError, match="x and y must both have shape"):
        spherical_dist(np.zeros((2,1)), np.zeros((3,2)))
    with pytest.raises(
        ValueError, match="Input arrays x and y must have the same shape"
    ):
        euclidean_dist(np.zeros((2,1)), np.zeros((3,2)))

def test_point_set_energy_blocks():
    # Hit block_size processing lines (679-680) and default s=None
    pts = np.random.rand(3, 10)
    e, d = point_set_energy_dist(pts, block_size=2)
    assert e > 0
    assert d > 0
