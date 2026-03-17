"""
PyEQSP: Python Equal Area Sphere Partitioning Library.

This module provides functions for equal area sphere partitioning, including:
- Creating partitions (`eq_regions`, `eq_point_set`, `eq_caps`)
- Calculating properties (`point_set_props`, `region_props`)
- Utilities (`utilities`)
- Illustrations (`illustrations`)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from eqsp.partitions import eq_caps, eq_point_set, eq_point_set_polar, eq_regions
from eqsp.point_set_props import (
    eq_dist_coeff,
    eq_energy_coeff,
    eq_energy_dist,
    eq_min_dist,
    point_set_dist_coeff,
    point_set_energy_coeff,
    point_set_energy_dist,
    point_set_min_dist,
)
from eqsp.region_props import eq_diam_bound
from eqsp.utilities import (
    area_of_cap,
    area_of_collar,
    area_of_sphere,
    cart2polar2,
    polar2cart,
    volume_of_ball,
)

__all__ = [
    "eq_regions",
    "eq_point_set",
    "eq_point_set_polar",
    "eq_caps",
    "volume_of_ball",
    "area_of_sphere",
    "area_of_cap",
    "area_of_collar",
    "polar2cart",
    "cart2polar2",
    "eq_min_dist",
    "eq_dist_coeff",
    "eq_energy_dist",
    "eq_energy_coeff",
    "eq_diam_bound",
    "point_set_min_dist",
    "point_set_energy_dist",
    "point_set_dist_coeff",
    "point_set_energy_coeff",
]

try:
    __version__ = _pkg_version("pyeqsp")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
