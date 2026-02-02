"""
EQSP Package
"""

from .partitions import (
    eq_caps,
    eq_point_set,
    eq_point_set_polar,
    eq_regions,
)

from .utilities import (
    asfloat,
    cart2polar2,
    polar2cart,
    euc2sph_dist,
    sph2euc_dist,
    euclidean_dist,
    spherical_dist,
    area_of_sphere,
    volume_of_ball,
    area_of_ideal_region,
    ideal_collar_angle,
    area_of_cap,
    sradius_of_cap,
    area_of_collar,
)

from .partition_options import partition_options

from ._private._partitions import (
   polar_colat,
   num_collars,
   ideal_region_list,
   cap_colats,
   round_to_naturals,
)

# illustration_options might be missing too?
# Let's check where illustration_options is.
