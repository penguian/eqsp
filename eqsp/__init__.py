
"""
Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox.

This module provides functions forEQ sphere partitioning, including:
- Creating partitions (`eq_regions`, `eq_point_set`, `eq_caps`)
- Calculating properties (`point_set_props`, `region_props`)
- Utilities (`utilities`)
- Illustrations (`illustrations`)
"""
from eqsp.partitions import eq_regions, eq_point_set, eq_point_set_polar, eq_caps
from eqsp.utilities import (
    volume_of_ball,
    area_of_sphere,
    area_of_cap,
    area_of_collar,
    polar2cart,
    cart2polar2,
)
from eqsp.point_set_props import (
    eq_min_dist,
    eq_dist_coeff,
    eq_energy_dist,
    eq_energy_coeff,
)
from eqsp.region_props import eq_diam_bound

# Define version
__version__ = "0.98.0"
