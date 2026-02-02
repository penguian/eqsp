"""
Recursive Zonal Equal Area Sphere Partitioning: Utilities

EQSP: Recursive Zonal Equal Area Sphere Partitioning.

Functions:
==========
- area_of_cap:            Area of spherical cap
- area_of_collar:         Area of spherical collar
- area_of_ideal_region:   Area of one region of an EQ partition
- area_of_sphere:         Area of sphere
- cart2polar2:            Convert Cartesian to spherical polar coordinates on S^2
- euc2sph_dist:           Convert Euclidean to spherical distance
- euclidean_dist:         Euclidean distance between two points
- fatcurve:               Create a parameterized cylindrical surface
- ideal_collar_angle:     Ideal angle for spherical collars of an EQ partition
- polar2cart:             Convert spherical polar to Cartesian coordinates
- sph2euc_dist:           Convert spherical to Euclidean distance
- spherical_dist:         Spherical distance between two points on the sphere
- sradius_of_cap:         Spherical radius of spherical cap of given area
- volume_of_ball:         Volume of the unit ball

Copyright 2025 Paul Leopardi.
For licensing, see LICENSE.
For references, see AUTHORS.
"""

from .area_of_cap import area_of_cap
from .area_of_collar import area_of_collar
from .area_of_ideal_region import area_of_ideal_region
from .area_of_sphere import area_of_sphere
from .cart2polar2 import cart2polar2
from .euc2sph_dist import euc2sph_dist
from .euclidean_dist import euclidean_dist
from .fatcurve import fatcurve
from .ideal_collar_angle import ideal_collar_angle
from .polar2cart import polar2cart
from .sph2euc_dist import sph2euc_dist
from .spherical_dist import spherical_dist
from .sradius_of_cap import sradius_of_cap
from .volume_of_ball import volume_of_ball
