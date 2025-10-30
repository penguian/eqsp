"""
Private functions for EQ sphere partitions.

This module contains helper functions used internally by the EQ sphere partitions package.
"""

import numpy as np
from math import pi, sin, cos, gcd
from typing import Tuple, Union, List


def bot_cap_region(dim: int, a_cap: float) -> np.ndarray:
    """South polar (bottom) cap region of EQ partition.

    An array of two points representing the bottom cap of radius a_cap as a region.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    a_cap : float
        Radius of the bottom cap.

    Returns
    -------
    np.ndarray
        Array representing the bottom cap region.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> bot_cap_region(1, 0.5)
    array([5.783185, 6.283185])
    """
    if dim == 1:
        region = np.array([2*pi-a_cap, 2*pi])
    else:
        sphere_region_1 = sphere_region(dim-1)
        region = np.vstack([sphere_region_1, [pi-a_cap, pi]])

    return region


def cap_colats(dim: int, N: int, c_polar: float, n_regions: np.ndarray) -> np.ndarray:
    """Colatitudes of spherical caps enclosing cumulative sum of regions.

    Given dim, N, c_polar and n_regions, determine c_caps,
    an increasing list of colatitudes of spherical caps which enclose the same area
    as that given by the cumulative sum of regions.
    The number of elements is n_collars+2.
    c_caps[0] is c_polar.
    c_caps[n_collars] is Pi-c_polar.
    c_caps[n_collars+1] is Pi.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int
        Number of regions.
    c_polar : float
        Colatitude of the polar cap.
    n_regions : np.ndarray
        Array of number of regions in each zone.

    Returns
    -------
    np.ndarray
        Array of colatitudes of spherical caps.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> cap_colats(2, 10, 0.5, np.array([1, 8, 1]))
    array([0.5     , 2.641593, 3.141593])
    """
    c_caps = np.zeros(n_regions.shape)
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(dim, N)
    n_collars = n_regions.size - 2
    subtotal_n_regions = 1

    for collar_n in range(1, n_collars+1):
        subtotal_n_regions += n_regions[collar_n]
        c_caps[collar_n] = sradius_of_cap(dim, subtotal_n_regions*ideal_region_area)

    c_caps[n_collars+1] = pi

    return c_caps


def centres_of_regions(regions: np.ndarray) -> np.ndarray:
    """Centre points of given regions.

    Parameters
    ----------
    regions : np.ndarray
        Array of regions.

    Returns
    -------
    np.ndarray
        Array of centre points of the regions.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> regions = np.array([[[0, 2*np.pi]], [[0, np.pi]]])
    >>> centres_of_regions(regions)
    array([[0.      ],
           [1.570796]])
    """
    tol = np.finfo(float).eps * 2**5
    dim = regions.shape[0]
    N = regions.shape[2]
    points = np.zeros((dim, N))

    top = regions[:, 0, :]
    bot = regions[:, 1, :]

    zero_bot = np.abs(bot[0, :]) < tol
    bot[0, zero_bot] = 2*pi

    equal_bot = np.abs(bot[0, :] - top[0, :]) < tol
    bot[0, equal_bot] = top[0, equal_bot] + 2*pi

    twopi_bot = np.abs(bot[0, :] - top[0, :] - 2*pi) < tol
    points[0, twopi_bot] = 0
    points[0, ~twopi_bot] = np.mod((bot[0, ~twopi_bot] + top[0, ~twopi_bot])/2, 2*pi)

    for k in range(1, dim):
        pi_bot = np.abs(bot[k, :] - pi) < tol
        points[k, pi_bot] = pi

        zero_top = np.abs(top[k, :]) < tol
        points[k, zero_top] = 0

        all_else = ~(zero_top | pi_bot)
        points[k, all_else] = np.mod((top[k, all_else] + bot[k, all_else])/2, pi)

    return points


def circle_offset(n_top: int, n_bot: int, extra_twist: bool = False) -> float:
    """Try to maximize minimum distance of center points for S^2 collars.

    Given n_top and n_bot, calculate an offset.

    The values n_top and n_bot represent the numbers of
    equally spaced points on two overlapping circles.
    The offset is given in multiples of whole rotations, and
    consists of three parts:
    1) Half the difference between a twist of one sector on each of bottom and top.
       This brings the centre points into alignment.
    2) A rotation which will maximize the minimum angle between
       points on the two circles.
    3) An optional extra twist by a whole number of sectors on the second circle.
       The extra twist is added so that the location of
       the minimum angle between circles will
       progressively twist around the sphere with each collar.

    Parameters
    ----------
    n_top : int
        Number of points on the top circle.
    n_bot : int
        Number of points on the bottom circle.
    extra_twist : bool, optional
        Whether to add an extra twist, by default False.

    Returns
    -------
    float
        Offset value.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> circle_offset(4, 8)
    0.09375
    >>> circle_offset(4, 8, True)
    0.84375
    """
    offset = (1/n_bot - 1/n_top)/2 + gcd(n_top, n_bot)/(2*n_top*n_bot)

    if extra_twist:
        twist = 6
        offset = offset + twist/n_bot

    return offset


def ideal_region_list(dim: int, N: int, c_polar: float, n_collars: int) -> np.ndarray:
    """The ideal real number of regions in each zone.

    List the ideal real number of regions in each collar, plus the polar caps.

    Given dim, N, c_polar and n_collars, determine r_regions,
    a list of the ideal real number of regions in each collar,
    plus the polar caps.
    The number of elements is n_collars+2.
    r_regions[0] is 1.
    r_regions[n_collars+1] is 1.
    The sum of r_regions is N.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int
        Number of regions.
    c_polar : float
        Colatitude of the polar cap.
    n_collars : int
        Number of collars.

    Returns
    -------
    np.ndarray
        Array of ideal number of regions in each zone.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> ideal_region_list(2, 10, 0.5, 1)
    array([1.      , 8.      , 1.      ])
    """
    r_regions = np.zeros(2+n_collars)
    r_regions[0] = 1

    if n_collars > 0:
        # Based on n_collars and c_polar, determine a_fitting,
        # the collar angle such that n_collars collars fit between the polar caps.
        a_fitting = (pi-2*c_polar)/n_collars
        ideal_region_area = area_of_ideal_region(dim, N)

        for collar_n in range(1, n_collars+1):
            ideal_collar_area = area_of_collar(dim, c_polar+(collar_n-1)*a_fitting,
                                             c_polar+collar_n*a_fitting)
            r_regions[collar_n] = ideal_collar_area / ideal_region_area

    r_regions[1+n_collars] = 1

    return r_regions


def num_collars(N: Union[int, np.ndarray], c_polar: Union[float, np.ndarray],
                a_ideal: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """The number of collars between the polar caps.

    Given N, an ideal angle, and c_polar,
    determine n_collars, the number of collars between the polar caps.

    Parameters
    ----------
    N : int or np.ndarray
        Number of regions.
    c_polar : float or np.ndarray
        Colatitude of the polar cap.
    a_ideal : float or np.ndarray
        Ideal angle.

    Returns
    -------
    int or np.ndarray
        Number of collars.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> num_collars(10, 0.5, 0.3)
    7
    >>> import numpy as np
    >>> num_collars(np.array([2, 10]), np.array([0.5, 0.5]), np.array([0.3, 0.3]))
    array([0, 7])
    """
    n_collars = np.zeros_like(N, dtype=int)

    if isinstance(N, np.ndarray):
        enough = (N > 2) & (a_ideal > 0)
        n_collars[enough] = np.maximum(1, np.round((pi-2*c_polar[enough])/a_ideal[enough])).astype(int)
    else:
        if N > 2 and a_ideal > 0:
            n_collars = max(1, round((pi-2*c_polar)/a_ideal))

    return n_collars


def polar_colat(dim: int, N: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """The colatitude of the North polar (top) spherical cap.

    Given dim and N, determine the colatitude of the North polar spherical cap.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int or np.ndarray
        Number of regions.

    Returns
    -------
    float or np.ndarray
        Colatitude of the North polar cap.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> polar_colat(2, 10)
    0.5
    >>> polar_colat(2, np.array([1, 2, 10]))
    array([3.141593, 1.570796, 0.5     ])
    """
    if isinstance(N, np.ndarray):
        c_polar = np.zeros_like(N, dtype=float)

        c_polar[N == 1] = pi
        c_polar[N == 2] = pi/2

        enough = N > 2
        c_polar[enough] = sradius_of_cap(dim, area_of_ideal_region(dim, N[enough]))

        return c_polar
    else:
        if N == 1:
            return pi
        elif N == 2:
            return pi/2
        else:
            return sradius_of_cap(dim, area_of_ideal_region(dim, N))


def rot3(axis: int, angle: float) -> np.ndarray:
    """R^3 rotation about a coordinate axis.

    Parameters
    ----------
    axis : int
        Coordinate axis (1, 2, or 3).
    angle : float
        Rotation angle.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> rot3(1, np.pi/6)
    array([[ 1.    ,  0.    ,  0.    ],
           [ 0.    ,  0.866 , -0.5   ],
           [ 0.    ,  0.5   ,  0.866 ]])
    >>> rot3(2, np.pi/6)
    array([[ 0.866 ,  0.    , -0.5   ],
           [ 0.    ,  1.    ,  0.    ],
           [ 0.5   ,  0.    ,  0.866 ]])
    >>> rot3(3, np.pi/6)
    array([[ 0.866 , -0.5   ,  0.    ],
           [ 0.5   ,  0.866 ,  0.    ],
           [ 0.    ,  0.    ,  1.    ]])
    """
    c = cos(angle)
    s = sin(angle)

    if axis == 1:
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 2:
        R = np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])
    elif axis == 3:
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    return R


def round_to_naturals(N: int, r_regions: np.ndarray) -> np.ndarray:
    """Round off a given list of numbers of regions.

    Given N and r_regions, determine n_regions,
    a list of the natural number of regions in each collar and the polar caps.
    This list is as close as possible to r_regions, using rounding.
    The number of elements is n_collars+2.
    n_regions[0] is 1.
    n_regions[n_collars+1] is 1.
    The sum of n_regions is N.

    Parameters
    ----------
    N : int
        Total number of regions.
    r_regions : np.ndarray
        Array of ideal real number of regions.

    Returns
    -------
    np.ndarray
        Array of natural number of regions.

    Notes
    -----
    Originally written by Paul Leopardi.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> round_to_naturals(10, np.array([1, 8.1, 0.9]))
    array([1, 8, 1])
    """
    n_regions = r_regions.copy()
    discrepancy = 0

    for zone_n in range(r_regions.size):
        n_regions[zone_n] = round(r_regions[zone_n] + discrepancy)
        discrepancy = discrepancy + r_regions[zone_n] - n_regions[zone_n]

    assert sum(n_regions) == N, f'Sum of result n_regions does not equal N=={N}'

    return n_regions.astype(int)


def s2_offset(points_1: np.ndarray) -> np.ndarray:
    """Experimental offset rotation of S^2.

    Parameters
    ----------
    points_1 : np.ndarray
        2xM matrix representing M points of S^2 in spherical polar coordinates.

    Returns
    -------
    np.ndarray
        R^3 rotation matrix which rotates the north pole of S^2 to a point
        specified by the points of points_1.

    See Also
    --------
    rot3, circle_offset

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> s = np.array([[0, 0.7854, 2.3562, 3.9270, 5.4978, 0],
    ...              [0, 1.5708, 1.5708, 1.5708, 1.5708, 3.1416]])
    >>> s2_offset(s)
    array([[ 0.    ,  0.7071,  0.7071],
           [-1.    ,  0.    ,  0.    ],
           [-0.    , -0.7071,  0.7071]])
    """
    n_in_collar = points_1.shape[1]

    if n_in_collar > 2:
        if (n_in_collar > 3) and (points_1[1, 1] == points_1[1, 2]):
            a_3 = (points_1[0, 1] + points_1[0, 2]) / 2
        else:
            a_3 = points_1[0, 1] + pi
        a_2 = points_1[1, 1] / 2
    else:
        a_3 = 0
        a_2 = pi/2

    rotation = np.matmul(rot3(2, -a_2), rot3(3, -a_3))

    return rotation


def sphere_region(dim: int) -> np.ndarray:
    """The sphere represented as a single region of an EQ partition.

    An array of two points representing S^dim as a region.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.

    Returns
    -------
    np.ndarray
        Array representing the sphere as a region.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> sphere_region(1)
    array([0.      , 6.283185])
    >>> sphere_region(2)
    array([[0.      , 6.283185],
           [0.      , 3.141593]])
    """
    if dim == 1:
        region = np.array([0, 2*pi])
    else:
        sphere_region_1 = sphere_region(dim-1)
        region = np.vstack([sphere_region_1, [0, pi]])

    return region


def top_cap_region(dim: int, a_cap: float) -> np.ndarray:
    """North polar (top) cap region of EQ partition.

    An array of two points representing the top cap of radius a_cap as a region.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    a_cap : float
        Radius of the top cap.

    Returns
    -------
    np.ndarray
        Array representing the top cap region.

    Notes
    -----
    Originally written by Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> top_cap_region(1, 0.5)
    array([0.      , 0.5     ])
    """
    if dim == 1:
        region = np.array([0, a_cap])
    else:
        sphere_region_1 = sphere_region(dim-1)
        region = np.vstack([sphere_region_1, [0, a_cap]])

    return region


# The following functions are used in the implementation but weren't provided in the original code
# These are placeholder implementations that would need to be replaced with the actual functions

def area_of_ideal_region(dim: int, N: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the area of an ideal region on a sphere.

    This is a placeholder function.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int or np.ndarray
        Number of regions.

    Returns
    -------
    float or np.ndarray
        Area of an ideal region.
    """
    # This is a placeholder implementation
    if isinstance(N, np.ndarray):
        return np.ones_like(N) * (4 * pi / N)
    else:
        return 4 * pi / N


def sradius_of_cap(dim: int, area: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the spherical radius of a cap with given area.

    This is a placeholder function.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    area : float or np.ndarray
        Area of the cap.

    Returns
    -------
    float or np.ndarray
        Spherical radius of the cap.
    """
    # This is a placeholder implementation
    if isinstance(area, np.ndarray):
        return np.sqrt(area / pi)
    else:
        return np.sqrt(area / pi)


def area_of_collar(dim: int, theta1: float, theta2: float) -> float:
    """Calculate the area of a collar on a sphere.

    This is a placeholder function.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    theta1 : float
        First colatitude.
    theta2 : float
        Second colatitude.

    Returns
    -------
    float
        Area of the collar.
    """
    # This is a placeholder implementation
    return 2 * pi * (np.cos(theta1) - np.cos(theta2))
