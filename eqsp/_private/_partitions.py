import numpy as np
from math import cos, gcd, pi, sin

from ..utilities import (
    area_of_collar,
    area_of_ideal_region,
    sradius_of_cap,
)


def bot_cap_region(dim, a_cap):
    """
    South polar (bottom) cap region of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of the sphere S^dim.
    a_cap : float
        Spherical radius of the bottom cap.

    Returns
    -------
    region : ndarray
        (dim, 2) array of two points in spherical polar coordinates
        representing the South polar cap region.

    See Also
    --------
    top_cap_region, sphere_region

    Raises
    ------
    ValueError
        If inputs are not appropriate numeric values.

    Notes
    -----
    The returned array shape is (dim, 2), columns represent bounds.

    Examples
    --------
    >>> np.set_printoptions(precision=4)
    >>> bot_cap_region(1, pi/6)
    array([[5.7596, 6.2832]])
    >>> bot_cap_region(2, pi/6)
    array([[0.    , 6.2832],
           [2.618 , 3.1416]])
    >>> bot_cap_region(3, pi/6)
    array([[0.    , 6.2832],
           [0.    , 3.1416],
           [2.618 , 3.1416]])
    """
    if dim == 1:
        return np.array([[2 * pi - a_cap, 2 * pi]])
    sphere_region_1 = sphere_region(dim - 1)
    first_col = np.append(sphere_region_1[:, 0], pi - a_cap)
    second_col = np.append(sphere_region_1[:, 1], pi)
    return np.column_stack([first_col, second_col])


def cap_colats(dim, N, c_polar, n_regions):
    """
    Colatitudes of spherical caps enclosing cumulative sum of regions.

    Parameters
    ----------
    dim : int
        Dimension of the sphere S^dim.
    N : int
        Number of regions in the partition.
    c_polar : float
        Colatitude of the North polar cap.
    n_regions : array_like
        List of region counts per collar and cap.

    Returns
    -------
    c_caps : ndarray
        Increasing array of colatitudes enclosing cumulative region sums.

    See Also
    --------
    polar_colat, sradius_of_cap, area_of_ideal_region

    Raises
    ------
    ValueError
        If inputs are inconsistent in length or type.

    Notes
    -----
    Length equals n_collars + 2. c_caps[0] = c_polar, last = pi.

    Examples
    --------
    >>> np.set_printoptions(precision=4)
    >>> dim = 2
    >>> N = 4
    >>> c_polar = polar_colat(dim, N)
    >>> n_regions = np.array([1, 2, 1])
    >>> cap_colats(dim, N, c_polar, n_regions)
    array([1.0472, 2.0944, 3.1416])
    """
    n_regions = np.asarray(n_regions)
    c_caps = np.zeros_like(n_regions, dtype=float)
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(dim, N)
    n_collars = n_regions.size - 2
    subtotal_n_regions = 1
    for collar_n in range(1, n_collars + 1):
        subtotal_n_regions = subtotal_n_regions + n_regions[collar_n]
        c_caps[collar_n] = sradius_of_cap(dim, subtotal_n_regions * ideal_region_area)
    c_caps[1 + n_collars] = pi
    return c_caps


def centres_of_regions(regions):
    """
    Centre points of given regions.

    Parameters
    ----------
    regions : ndarray
        (dim, 2, N) array of region bounds in spherical polar coordinates.

    Returns
    -------
    points : ndarray
        (dim, N) array of region center points.

    See Also
    --------
    sphere_region

    Raises
    ------
    AssertionError
        If input dimensions are inconsistent.

    Notes
    -----
    Uses floating point tolerance for equality. Azimuth mod 2*pi, others mod pi.

    Examples
    --------
    >>> regions = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    >>> centres_of_regions(regions)
    array([[1., 2.],
           [0., 2.]])
    """
    tol = np.finfo(float).eps * 2**5
    regions = np.asarray(regions, dtype=float)
    if regions.ndim == 2:
        regions = regions[:, :, np.newaxis]
    dim = regions.shape[0]
    N = regions.shape[2]
    points = np.zeros((dim, N), dtype=float)
    top = regions[:, 0, :]
    bot = regions[:, 1, :]
    zero_bot = np.abs(bot[0, :]) < tol
    bot[0, zero_bot] = 2 * pi
    equal_bot = np.abs(bot[0, :] - top[0, :]) < tol
    bot[0, equal_bot] = top[0, equal_bot] + 2 * pi
    twopi_bot = np.abs(bot[0, :] - top[0, :] - 2 * pi) < tol
    points[0, twopi_bot] = 0.0
    mask_other = ~twopi_bot
    points[0, mask_other] = np.mod(
        (bot[0, mask_other] + top[0, mask_other]) / 2.0, 2 * pi
    )
    for k in range(1, dim):
        pi_bot = np.abs(bot[k, :] - pi) < tol
        points[k, pi_bot] = pi
        zero_top = np.abs(top[k, :]) < tol
        points[k, zero_top] = 0.0
        all_else = ~(zero_top | pi_bot)
        points[k, all_else] = np.mod((top[k, all_else] + bot[k, all_else]) / 2.0, pi)
    return points


def circle_offset(n_top, n_bot, extra_twist=False):
    """
    Maximize minimum distance of center points for S^2 collars.

    Parameters
    ----------
    n_top : int
        Number of points in top circle.
    n_bot : int
        Number of points in bottom circle.
    extra_twist : bool, optional
        If True, add an extra sector twist.

    Returns
    -------
    offset : float
        Offset in multiples of rotations.

    See Also
    --------
    rot3, s2_offset

    Notes
    -----
    Offset includes a half-sector twist, max-min separation, and optional twist.

    Examples
    --------
    >>> def disp(x):
    ...     print(f'{x:.5g}')
    >>> disp(circle_offset(3, 4))
    6.9389e-18
    >>> disp(circle_offset(3, 4, extra_twist=True))
    1.5
    >>> disp(circle_offset(7, 11))
    -0.019481
    """
    if n_top == 0 or n_bot == 0:
        raise ValueError("n_top and n_bot must be positive integers")
    offset = (1.0 / n_bot - 1.0 / n_top) / 2.0 + gcd(int(n_top), int(n_bot)) / (
        2.0 * n_top * n_bot
    )
    if extra_twist:
        twist = 6
        offset = offset + twist / float(n_bot)
    return offset


def ideal_region_list(dim, N, c_polar, n_collars):
    """
    Ideal real number of regions in each zone.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int
        Number of regions.
    c_polar : float
        North polar colatitude.
    n_collars : int
        Number of collars.

    Returns
    -------
    r_regions : ndarray
        Array of ideal real region counts, length n_collars+2.

    See Also
    --------
    ideal_collar_angle, num_collars, polar_colat

    Notes
    -----
    r_regions[0] and r_regions[-1] are 1. Sum equals N.

    Examples
    --------
    >>> dim = 2
    >>> N = 4
    >>> c_polar = polar_colat(dim, N)
    >>> n_collars = 1
    >>> ideal_region_list(dim, N, c_polar, n_collars)
    array([1., 2., 1.])
    """
    r_regions = np.zeros(2 + int(n_collars), dtype=float)
    r_regions[0] = 1.0
    if n_collars > 0:
        a_fitting = (pi - 2.0 * c_polar) / float(n_collars)
        ideal_region_area = area_of_ideal_region(dim, N)
        for collar_n in range(1, n_collars + 1):
            a1 = c_polar + (collar_n - 1) * a_fitting
            a2 = c_polar + collar_n * a_fitting
            ideal_collar_area = area_of_collar(dim, a1, a2)
            r_regions[collar_n] = ideal_collar_area / ideal_region_area
    r_regions[-1] = 1.0
    return r_regions


def num_collars(N, c_polar, a_ideal):
    """
    Number of collars between polar caps.

    Parameters
    ----------
    N : int or array_like
        Number of regions.
    c_polar : float or array_like
        North polar colatitude.
    a_ideal : float or array_like
        Ideal collar angle.

    Returns
    -------
    n_collars : int or ndarray
        Number of collars.

    See Also
    --------
    ideal_collar_angle, ideal_region_list, polar_colat

    Notes
    -----
    Zero for N <= 2 or a_ideal <= 0.

    Examples
    --------
    >>> dim = 2
    >>> N = 4
    >>> c_polar = polar_colat(dim, N)
    >>> a_ideal = 1.0471975512
    >>> num_collars(N, c_polar, a_ideal)
    1
    """
    N_arr = np.asarray(N)
    c_polar_arr = np.asarray(c_polar)
    a_ideal_arr = np.asarray(a_ideal)
    n_collars = np.zeros_like(N_arr, dtype=int)
    enough = (N_arr > 2) & (a_ideal_arr > 0)
    if np.any(enough):
        val = np.round((pi - 2.0 * c_polar_arr[enough]) / a_ideal_arr[enough])
        val = np.maximum(1, val).astype(int)
        n_collars[enough] = val
    if np.isscalar(N):
        return int(n_collars)
    return n_collars


def polar_colat(dim, N):
    """
    Colatitude of North polar (top) spherical cap.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array_like
        Number of regions.

    Returns
    -------
    c_polar : float or ndarray
        North polar colatitude.

    See Also
    --------
    ideal_collar_angle, ideal_region_list, num_collars

    Notes
    -----
    c_polar = pi for N==1, pi/2 for N==2, else computed from ideal region.

    Examples
    --------
    >>> def disp(x):
    ...     print(f'{x:.5g}')
    >>> disp(polar_colat(2, 4))
    1.0472
    >>> disp(polar_colat(2, 10))
    0.6435
    >>> disp(polar_colat(3, 6))
    0.98448
    """
    if np.isscalar(N):
        if N == 1:
            return pi
        if N == 2:
            return pi / 2.0
        area = area_of_ideal_region(dim, N)
        return sradius_of_cap(dim, area)
    N_arr = np.asarray(N)
    c_polar = np.zeros_like(N_arr, dtype=float)
    c_polar[N_arr == 1] = pi
    c_polar[N_arr == 2] = pi / 2.0
    mask = N_arr > 2
    if np.any(mask):
        areas = area_of_ideal_region(dim, N_arr[mask])
        c_polar[mask] = sradius_of_cap(dim, areas)
    return c_polar


def rot3(axis, angle):
    """
    R^3 rotation about a coordinate axis.

    Parameters
    ----------
    axis : {1,2,3}
        Axis index (1=x, 2=y, 3=z).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.

    Examples
    --------
    >>> rot3(1, pi/6)
    array([[ 1.   ,  0.   ,  0.   ],
           [ 0.   ,  0.866, -0.5  ],
           [ 0.   ,  0.5  ,  0.866]])
    >>> rot3(2, pi/6)
    array([[ 0.866,  0.   , -0.5  ],
           [ 0.   ,  1.   ,  0.   ],
           [ 0.5  ,  0.   ,  0.866]])
    >>> rot3(3, pi/6)
    array([[ 0.866, -0.5  ,  0.   ],
           [ 0.5  ,  0.866,  0.   ],
           [ 0.   ,  0.   ,  1.   ]])
    """
    c = cos(angle)
    s = sin(angle)
    if axis == 1:
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 2:
        R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    elif axis == 3:
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("axis must be 1, 2, or 3")
    return R


def round_to_naturals(N, r_regions):
    """
    Round off a given list of numbers of regions.

    Parameters
    ----------
    N : int
        Total number of regions.
    r_regions : array_like
        Ideal real region counts (length n_collars+2).

    Returns
    -------
    n_regions : ndarray
        Integer region counts, length as r_regions, sum equals N.

    Raises
    ------
    AssertionError
        If sum does not equal N.

    Notes
    -----
    Uses discrepancy tracking for rounding.

    Examples
    --------
    >>> round_to_naturals(4, [1.0, 2.0, 1.0])
    array([1, 2, 1])
    """
    r_regions = np.asarray(r_regions, dtype=float)
    n_regions = np.zeros_like(r_regions, dtype=int)
    discrepancy = 0.0
    for i in range(r_regions.size):
        n_regions[i] = int(round(r_regions[i] + discrepancy))
        discrepancy = discrepancy + r_regions[i] - float(n_regions[i])
    assert n_regions.sum() == int(N), "Sum of result n_regions does not equal N==%g" % N
    return n_regions


def s2_offset(points_1):
    """
    Experimental offset rotation of S^2.

    Parameters
    ----------
    points_1 : ndarray
        2 x M array representing M points in spherical polar coordinates.

    Returns
    -------
    rotation : ndarray
        3x3 rotation matrix moving north pole to specified location.

    See Also
    --------
    rot3, circle_offset

    Notes
    -----
    For equal colats, average azimuth used; else, azimuth+pi.

    Examples
    --------
    >>> s = np.array([[0., 0.785398, 2.356194, 3.9270, 5.49778, 0.],
    ...               [0., 1.570796, 1.570796, 1.570796, 1.570796, 3.1416]])
    >>> s2_offset(s)
    array([[ 2.3108e-07,  7.0711e-01,  7.0711e-01],
           [-1.0000e+00,  3.2679e-07,  0.0000e+00],
           [-2.3108e-07, -7.0711e-01,  7.0711e-01]])
    """
    points_1 = np.asarray(points_1, dtype=float)
    n_in_collar = points_1.shape[1]
    if n_in_collar > 2:
        if (n_in_collar > 3) and (points_1[1, 1] == points_1[1, 2]):
            a_3 = (points_1[0, 1] + points_1[0, 2]) / 2.0
        else:
            a_3 = points_1[0, 1] + pi
        a_2 = points_1[1, 1] / 2.0
    else:
        a_3 = 0.0
        a_2 = pi / 2.0
    return np.dot(rot3(2, -a_2), rot3(3, -a_3))


def sphere_region(dim):
    """
    Sphere represented as single region of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere S^dim.

    Returns
    -------
    region : ndarray
        (dim, 2) array representing the whole sphere region.

    See Also
    --------
    bot_cap_region, top_cap_region

    Examples
    --------
    >>> sphere_region(1)
    array([[0.    , 6.2832]])
    >>> sphere_region(2)
    array([[0.    , 6.2832],
           [0.    , 3.1416]])
    >>> sphere_region(3)
    array([[0.    , 6.2832],
           [0.    , 3.1416],
           [0.    , 3.1416]])
    """
    if dim == 1:
        return np.array([[0.0, 2.0 * pi]])
    sphere_region_1 = sphere_region(dim - 1)
    first_col = np.append(sphere_region_1[:, 0], 0.0)
    second_col = np.append(sphere_region_1[:, 1], pi)
    return np.column_stack([first_col, second_col])


def top_cap_region(dim, a_cap):
    """
    North polar (top) cap region of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere S^dim.
    a_cap : float
        Spherical radius of the top cap.

    Returns
    -------
    region : ndarray
        (dim, 2) array of two points in spherical polar coordinates
        representing the North polar cap region.

    See Also
    --------
    bot_cap_region, sphere_region

    Notes
    -----
    The returned array shape is (dim, 2).

    Examples
    --------
    >>> import math
    >>> top_cap_region(1, pi/6)
    array([[0.    , 0.5236]])
    >>> top_cap_region(2, pi/6)
    array([[0.    , 6.2832],
           [0.    , 0.5236]])

    >>> top_cap_region(3, pi/6)
    array([[0.    , 6.2832],
           [0.    , 3.1416],
           [0.    , 0.5236]])
    """
    if dim == 1:
        return np.array([[0.0, a_cap]])
    sphere_region_1 = sphere_region(dim - 1)
    first_col = np.append(sphere_region_1[:, 0], 0.0)
    second_col = np.append(sphere_region_1[:, 1], a_cap)
    return np.column_stack([first_col, second_col])
