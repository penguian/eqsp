import numpy as np
from math import (
    cos,
    gcd,
    isclose,
    pi,
    sin)

from utilities import (
    area_of_ideal_region,
    area_of_collar,
    sradius_of_cap)

def bot_cap_region(dim, a_cap):
    """
    BOT_CAP_REGION South polar (bottom) cap region of EQ partition

    An array of two points representing the bottom cap of radius a_cap as a region.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    a_cap : float
        Cap angle in radians.

    Returns
    -------
    region : ndarray
        Array representing the bottom cap region.

    See Also
    --------
    sphere_region

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> bot_cap_region(1, 0.5)
    array([5.78318531, 6.28318531])
    """
    if dim == 1:
        return np.array([2 * pi - a_cap, 2 * pi])
    else:
        sphere_region_1 = sphere_region(dim - 1)
        return np.vstack([np.append(sphere_region_1[:, 0], pi - a_cap),
                          np.append(sphere_region_1[:, 1], pi)]).T

def cap_colats(dim, N, c_polar, n_regions):
    """
    CAP_COLATS Colatitudes of spherical caps enclosing cumulative sum of regions

    Given dim, N, c_polar and n_regions, determine c_caps,
    an increasing list of colatitudes of spherical caps which enclose the same area
    as that given by the cumulative sum of regions.
    The number of elements is n_collars+2.
    c_caps[1] is c_polar.
    c_caps[n_collars+1] is Pi-c_polar.
    c_caps[n_collars+2] is Pi.

    Parameters
    ----------
    dim : int
    N : int
    c_polar : float
    n_regions : ndarray

    Returns
    -------
    c_caps : ndarray

    See Also
    --------
    sradius_of_cap, area_of_ideal_region

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> n_regions = np.array([1, 2, 1])
    >>> cap_colats(2, 4, 0.5, n_regions)
    array([0.5, 3.14159265])
    """
    c_caps = np.zeros_like(n_regions, dtype=float)
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(dim, N)
    n_collars = n_regions.size - 2
    subtotal_n_regions = 1
    for collar_n in range(1, n_collars + 1):
        subtotal_n_regions += n_regions[collar_n]
        c_caps[collar_n] = sradius_of_cap(dim, subtotal_n_regions * ideal_region_area)
    c_caps[n_collars + 1] = pi
    return c_caps

def centres_of_regions(regions):
    """
    CENTRES_OF_REGIONS Centre points of given regions

    Parameters
    ----------
    regions : ndarray
        Shape (dim, 2, N)

    Returns
    -------
    points : ndarray
        Shape (dim, N)

    See Also
    --------
    None

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> regions = np.array([[[0, 1], [0, 1]], [[2, 2], [2, 2]]])
    >>> centres_of_regions(regions)
    array([[0.5, 0.5],
           [2. , 2. ]])
    """
    tol = np.finfo(float).eps * 2 ** 5
    dim = regions.shape[0]
    N = regions.shape[2]
    points = np.zeros((dim, N))
    top = regions[:, 0, :]
    bot = regions[:, 1, :]
    zero_bot = np.abs(bot[0, :]) < tol
    bot[0, zero_bot] = 2 * pi
    equal_bot = np.abs(bot[0, :] - top[0, :]) < tol
    bot[0, equal_bot] = top[0, equal_bot] + 2 * pi
    twopi_bot = np.abs(bot[0, :] - top[0, :] - 2 * pi) < tol
    points[0, twopi_bot] = 0
    other = ~twopi_bot
    points[0, other] = np.mod((bot[0, other] + top[0, other]) / 2, 2 * pi)
    for k in range(1, dim):
        pi_bot = np.abs(bot[k, :] - pi) < tol
        points[k, pi_bot] = pi
        zero_top = np.abs(top[k, :]) < tol
        points[k, zero_top] = 0
        all_else = ~(zero_top | pi_bot)
        points[k, all_else] = np.mod((top[k, all_else] + bot[k, all_else]) / 2, pi)
    return points

def circle_offset(n_top, n_bot, extra_twist=False):
    """
    CIRCLE_OFFSET Try to maximize minimum distance of center points for S^2 collars

    Given n_top and n_bot, calculate an offset.

    Parameters
    ----------
    n_top : int
    n_bot : int
    extra_twist : bool, optional

    Returns
    -------
    offset : float

    See Also
    --------
    None

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> circle_offset(5, 7)
    0.02857142857142857
    >>> circle_offset(5, 7, True)
    0.8857142857142857
    """
    offset = (1 / n_bot - 1 / n_top) / 2 + gcd(n_top, n_bot) / (2 * n_top * n_bot)
    if extra_twist:
        twist = 6
        offset += twist / n_bot
    return offset

def ideal_region_list(dim, N, c_polar, n_collars):
    """
    IDEAL_REGION_LIST The ideal real number of regions in each zone

    List the ideal real number of regions in each collar, plus the polar caps.

    Parameters
    ----------
    dim : int
    N : int
    c_polar : float
    n_collars : int

    Returns
    -------
    r_regions : ndarray

    See Also
    --------
    area_of_ideal_region, area_of_collar

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> ideal_region_list(2, 4, 0.5, 2)
    array([1., 1., 1., 1.])
    """
    r_regions = np.zeros(2 + n_collars)
    r_regions[0] = 1
    if n_collars > 0:
        a_fitting = (pi - 2 * c_polar) / n_collars
        ideal_region_area = area_of_ideal_region(dim, N)
        for collar_n in range(1, n_collars + 1):
            ideal_collar_area = area_of_collar(
                    dim,
                    c_polar + (collar_n - 1) * a_fitting,
                    c_polar + collar_n * a_fitting)
            r_regions[collar_n] = ideal_collar_area / ideal_region_area
    r_regions[-1] = 1
    return r_regions

def num_collars(N, c_polar, a_ideal):
    """
    NUM_COLLARS The number of collars between the polar caps

    Given N, an ideal angle, and c_polar,
    determine n_collars, the number of collars between the polar caps.

    Parameters
    ----------
    N : int or ndarray
    c_polar : float or ndarray
    a_ideal : float or ndarray

    Returns
    -------
    n_collars : ndarray

    See Also
    --------
    None

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> num_collars(np.array([4]), np.array([0.5]), np.array([0.5]))
    array([5])
    """
    N = np.array(N)
    c_polar = np.array(c_polar)
    a_ideal = np.array(a_ideal)
    n_collars = np.zeros_like(N, dtype=int)
    enough = (N > 2) & (a_ideal > 0)
    n_collars[enough] = np.maximum(
        1,
        np.round((pi - 2 * c_polar[enough]) / a_ideal[enough])).astype(int)
    return n_collars

def polar_colat(dim, N):
    """
    POLAR_COLAT The colatitude of the North polar (top) spherical cap

    Given dim and N, determine the colatitude of the North polar spherical cap.

    Parameters
    ----------
    dim : int
    N : int or ndarray

    Returns
    -------
    c_polar : ndarray

    See Also
    --------
    sradius_of_cap, area_of_ideal_region

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> polar_colat(2, np.array([1, 2, 4]))
    array([3.14159265, 1.57079633, 2.])
    """
    N = np.array(N)
    c_polar = np.zeros_like(N, dtype=float)
    enough = N > 2
    c_polar[N == 1] = pi
    c_polar[N == 2] = pi / 2
    c_polar[enough] = sradius_of_cap(dim, area_of_ideal_region(dim, N[enough]))
    return c_polar

def rot3(axis, angle):
    """
    ROT3 R^3 rotation about a coordinate axis

    Parameters
    ----------
    axis : int
        1, 2, or 3
    angle : float

    Returns
    -------
    R : ndarray
        3x3 rotation matrix

    See Also
    --------
    None

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> np.round(rot3(1, np.pi/6), 4)
    array([[ 1.    ,  0.    ,  0.    ],
           [ 0.    ,  0.866 , -0.5   ],
           [ 0.    ,  0.5   ,  0.866 ]])
    >>> np.round(rot3(2, np.pi/6), 4)
    array([[ 0.866 ,  0.    , -0.5   ],
           [ 0.    ,  1.    ,  0.    ],
           [ 0.5   ,  0.    ,  0.866 ]])
    >>> np.round(rot3(3, np.pi/6), 4)
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
    else:
        raise ValueError("Axis must be 1, 2, or 3.")
    return R

def round_to_naturals(N, r_regions):
    """
    ROUND_TO_NATURALS Round off a given list of numbers of regions

    Given N and r_regions, determine n_regions,
    a list of the natural number of regions in each collar and the polar caps.
    This list is as close as possible to r_regions, using rounding.

    Parameters
    ----------
    N : int
    r_regions : ndarray

    Returns
    -------
    n_regions : ndarray

    See Also
    --------
    None

    Notes
    -----
    Copyright 2025 Paul Leopardi.
    Add an assertion.
    Copyright 2024 Paul Leopardi.
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> round_to_naturals(4, np.array([1.1, 1.9, 1.0]))
    array([1, 2, 1])
    """
    n_regions = np.copy(r_regions)
    discrepancy = 0
    for zone_n in range(r_regions.size):
        n_regions[zone_n] = round(r_regions[zone_n] + discrepancy)
        discrepancy += r_regions[zone_n] - n_regions[zone_n]
    n_regions = n_regions.astype(int)
    assert np.sum(n_regions) == N, f'Sum of result n_regions does not equal N=={N}'
    return n_regions

def s2_offset(points_1):
    """
    S2_OFFSET Experimental offset rotation of S^2

    ROTATION = S2_OFFSET(POINTS_1) sets ROTATION to be an R^3 rotation matrix which
    rotates the north pole of S^2 to a point specified by the points of POINTS_1.

    Parameters
    ----------
    points_1 : ndarray
        2 by M matrix, representing M points of S^2 in spherical polar coordinates.

    Returns
    -------
    rotation : ndarray
        3x3 rotation matrix.

    See Also
    --------
    rot3, circle_offset

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> s = np.array([[0, 0.78539816, 2.35619449, 3.927, 5.4978, 0],
    ...               [0, 1.57079633, 1.57079633, 1.57079633, 1.57079633, 3.1416]])
    >>> np.round(s2_offset(s), 4)
    array([[ 0.    ,  0.7071,  0.7071],
           [-1.    ,  0.    ,  0.    ],
           [-0.    , -0.7071,  0.7071]])
    """
    n_in_collar = points_1.shape[1]
    if n_in_collar > 2:
        if n_in_collar > 3 and isclose(points_1[1, 1], points_1[1, 2]):
            a_3 = (points_1[0, 1] + points_1[0, 2]) / 2
        else:
            a_3 = points_1[0, 1] + pi
        a_2 = points_1[1, 1] / 2
    else:
        a_3 = 0
        a_2 = pi / 2
    return np.dot(rot3(2, -a_2), rot3(3, -a_3))

def sphere_region(dim):
    """
    SPHERE_REGION The sphere represented as a single region of an EQ partition

    An array of two points representing S^dim as a region.

    Parameters
    ----------
    dim : int

    Returns
    -------
    region : ndarray

    See Also
    --------
    None

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> sphere_region(1)
    array([0., 6.28318531])
    """
    if dim == 1:
        return np.array([0., 2 * pi])
    else:
        sphere_region_1 = sphere_region(dim - 1)
        return np.vstack([np.append(sphere_region_1[:, 0], 0),
                          np.append(sphere_region_1[:, 1], pi)]).T

def top_cap_region(dim, a_cap):
    """
    TOP_CAP_REGION North polar (top) cap region of EQ partition

    An array of two points representing the top cap of radius a_cap as a region.

    Parameters
    ----------
    dim : int
    a_cap : float

    Returns
    -------
    region : ndarray

    See Also
    --------
    sphere_region

    Notes
    -----
    Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

    Examples
    --------
    >>> top_cap_region(1, 0.5)
    array([0. , 0.5])
    """
    if dim == 1:
        return np.array([0., a_cap])
    else:
        sphere_region_1 = sphere_region(dim - 1)
        return np.vstack([np.append(sphere_region_1[:, 0], 0),
                          np.append(sphere_region_1[:, 1], a_cap)]).T
