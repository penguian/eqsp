import numpy as np

from math import pi

from utilities import (
    area_of_collar,
    area_of_ideal_region,
    asfloat,
    sradius_of_cap)


def bot_cap_region(dim, a_cap):
    """
    South polar (bottom) cap region of EQ partition.

    Syntax:
        region = bot_cap_region(dim, a_cap)

    Description:
        Sets `region` to be an array of two points in spherical polar
        coordinates representing the South polar (bottom) cap of spherical
        radius `a_cap` as a region of the sphere S^dim.

    Parameters:
        dim (int): Dimension of the sphere.
        a_cap (float): Radius of the spherical cap.

    Returns:
        numpy.ndarray: Array of shape (dim, 2) representing the region.

    Examples:
        >>> np.round(bot_cap_region(3, pi / 6), 5)
        array([[0.     , 6.28319],
               [0.     , 3.14159],
               [2.61799, 3.14159]])

        >>> np.round(bot_cap_region(1, pi / 6), 5)
        array([[5.75959, 6.28319]])

        >>> np.round(bot_cap_region(2, pi / 6), 5)
        array([[0.     , 6.28319],
               [2.61799, 3.14159]])



    See Also:
        top_cap_region, sphere_region
    """
    if dim == 1:
        region = np.array([[2*pi - a_cap, 2*pi]])
    else:
        sphere_region_1 = sphere_region(dim - 1)
        region = np.hstack(
            (np.append(sphere_region_1[:, 0], [pi - a_cap]).reshape(dim,1),
             np.append(sphere_region_1[:, 1], [pi]).reshape(dim,1)))
    return region


def cap_colats(dim, N, c_polar, n_regions):
    """
    Colatitudes of spherical caps enclosing cumulative sum of regions.

    Syntax:
        c_caps = cap_colats(dim, N, c_polar, n_regions)

    Description:
        Determines `c_caps`, an increasing array of colatitudes of spherical
        caps enclosing the same area as the cumulative sum of regions.

    Parameters:
        dim (int): Dimension of the sphere.
        N (int): Number of regions in the equal area partition.
        c_polar (float): Colatitude of the North polar cap.
        n_regions (array-like): Number of regions in each collar and the caps.

    Returns:
        numpy.ndarray: Array of colatitudes (float).

    Examples:
        >>> dim = 2; N = 4
        >>> c_polar = polar_colat(dim, N); n_regions = np.array([1, 2, 1])
        >>> np.round(cap_colats(dim, N, c_polar, n_regions), 5)
        array([1.0472 , 2.0944 , 3.14159])

        >>> dim = 2; N = 10
        >>> c_polar = polar_colat(dim, N); n_regions = np.array([1, 4, 4, 1])
        >>> np.round(cap_colats(dim, N, c_polar, n_regions), 5)
        array([0.6435,    1.5708,    2.4981,    3.1416])

        >>> dim = 3; N = 6
        >>> c_polar = polar_colat(dim, N); n_regions = np.array([1, 4, 1])
        >>> np.round(cap_colats(dim, N, c_polar, n_regions), 5)
        array([0.98451, 2.1571, 3.14159])

    See Also:
        polar_colat
    """
    c_caps = np.zeros(len(n_regions), dtype=float)
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(dim, N)
    subtotal_n_regions = 1
    for collar_n in range(len(n_regions) - 2):
        subtotal_n_regions += n_regions[collar_n + 1]
        c_caps[collar_n + 1] = sradius_of_cap(
            dim,
            subtotal_n_regions * ideal_region_area)
    c_caps[-1] = pi
    return c_caps


def centres_of_regions(regions):
    """
    Centre points of given regions.

    Syntax:
        points = centres_of_regions(regions)

    Description:
        Sets `points` to be the centres of regions in `regions`, given as
        pairs of points in spherical polar coordinates. If `regions` is a
        dim x 2 x N array, the result `points` is a dim x N array.

    Parameters:
        regions (numpy.ndarray): Array of shape (dim, 2, N).

    Returns:
        numpy.ndarray: Array of shape (dim, N) of centre points.

    Examples:
        >>> regions = np.array([
        ...     [[0, 1.5708, 4.7124, 0], [0, 1.5708, 1.5708, 3.1416]]
        ... ])
        >>> np.round(centres_of_regions(regions), 5)
        array([[0.    , 1.5708, 4.7124, 0.    ],
               [0.    , 1.5708, 1.5708, 3.1416]])

    See Also:
        eq_regions
    """
    tol = np.finfo(float).eps * 32
    dim, _, N = regions.shape
    points = np.zeros((dim, N))
    top = regions[:, 0, :]
    bot = regions[:, 1, :]
    zero_bot = np.abs(bot[0, :]) < tol
    bot[0, zero_bot] = 2*pi
    equal_bot = np.abs(bot[0, :] - top[0, :]) < tol
    bot[0, equal_bot] = top[0, equal_bot] + 2*pi
    twopi_bot = np.abs(bot[0, :] - top[0, :] - 2*pi) < tol
    points[0, twopi_bot] = 0
    points[0, ~twopi_bot] = np.mod(
        (bot[0, ~twopi_bot] + top[0, ~twopi_bot]) / 2, 2*pi
    )
    for k in range(1, dim):
        pi_bot = np.abs(bot[k, :] - pi) < tol
        points[k, pi_bot] = pi
        zero_top = np.abs(top[k, :]) < tol
        points[k, zero_top] = 0
        all_else = ~(zero_top | pi_bot)
        points[k, all_else] = np.mod(
            (top[k, all_else] + bot[k, all_else]) / 2, pi
        )
    return points


def circle_offset(n_top, n_bot, extra_twist=False):
    """
    Try to maximize minimum distance of center points for S^2 collars.

    Syntax:
        offset = circle_offset(n_top, n_bot, extra_twist)

    Description:
        Calculates the offset `offset` that maximizes the minimum distance
        between two sets of points on a circle, determined by the numbers
        `n_top` and `n_bot`.

    Parameters:
        n_top (int): Number of points on the top circle.
        n_bot (int): Number of points on the bottom circle.
        extra_twist (bool or int, optional): Whether to add an extra twist.

    Returns:
        float: The calculated offset.

    Notes:
        The offset consists of three parts:
        1) Half the difference between a twist of one sector on each of
           bottom and top.
        2) A rotation that maximizes the minimum angle between points.
        3) An optional extra twist by a whole number of sectors on the second
           circle.

    Examples:
        >>> circle_offset(3, 4)
        0.0

        >>> circle_offset(3, 4, 1)
        1.5

        >>> np.round(circle_offset(7, 11), 5)
        -0.01948
    """
    offset = (1 / n_bot - 1 / n_top) / 2 + np.gcd(n_top, n_bot) / (2 * n_top * n_bot)
    if extra_twist:
        twist = 6
        offset += twist / n_bot
    return offset


def ideal_region_list(dim, N, c_polar, n_collars):
    """
    The ideal real number of regions in each zone.

    Syntax:
        r_regions = ideal_region_list(dim, N, c_polar, n_collars)

    Description:
        Determines `r_regions`, an array of the ideal real number of regions
        in each collar plus the polar caps. The input includes `dim`, `N`,
        `c_polar`, and `n_collars`.

    Parameters:
        dim (int): Dimension of the sphere.
        N (int): Number of regions in the partition.
        c_polar (float): Colatitude of the North polar cap.
        n_collars (int): Number of collars in the partition.

    Returns:
        numpy.ndarray: Array of ideal real number of regions.

    Notes:
        The length of `r_regions` is `n_collars` + 2.
        r_regions[0] = 1 and r_regions[-1] = 1.
        The sum of `r_regions` is `N`.

    Examples:
        >>> from utilities import ideal_collar_angle

        >>> dim = 2; N = 4;
        >>> c_polar = polar_colat(dim, N);
        >>> n_collars = num_collars(N,c_polar,ideal_collar_angle(dim,N));
        >>> r_regions = ideal_region_list(dim,N,c_polar,n_collars)
        array([1. , 2. , 1. ])

        >>> dim = 2; N = 10;
        >>> c_polar = polar_colat(dim, N);
        >>> n_collars = num_collars(N,c_polar,ideal_collar_angle(dim,N));
        >>> r_regions = ideal_region_list(dim,N,c_polar,n_collars)
        array([1. , 4. , 4. , 1. ])

        >>> dim = 3; N = 6;
        >>> c_polar = polar_colat(dim, N);
        >>> n_collars = num_collars(N,c_polar,ideal_collar_angle(dim,N));
        >>> r_regions = ideal_region_list(dim,N,c_polar,n_collars)
        array([1. , 4. , 1. ])

    See Also:
        num_collars, polar_colat
    """
    r_regions = np.zeros(n_collars + 2)
    r_regions[0] = 1
    if n_collars > 0:
        a_fitting = (pi - 2 * c_polar) / n_collars
        ideal_area = area_of_ideal_region(dim, N)
        for collar_n in range(n_collars):
            collar_area = area_of_collar(
                dim,
                c_polar + collar_n * a_fitting,
                c_polar + (collar_n + 1) * a_fitting,
            )
            r_regions[1 + collar_n] = collar_area / ideal_area
    r_regions[-1] = 1
    return r_regions


def num_collars(N, c_polar, a_ideal):
    """
    The number of collars between the polar caps.

    Syntax:
        n_collars = num_collars(N, c_polar, a_ideal)

    Description:
        Determines the number of collars `n_collars` between the polar caps
        of an equal area partition of the sphere into `N` regions.

    Parameters:
        N (int): Number of regions in the partition.
        c_polar (float): Colatitude of the North polar cap.
        a_ideal (float): Ideal collar angle.

    Returns:
        int: Number of collars between the polar caps.

    Examples:
        >>> num_collars(4, 1.0472, 2.0944)
        1

        >>> num_collars(10, 0.6435, 0.92699)
        2

    See Also:
        ideal_region_list, polar_colat
    """
    if N <= 2 or a_ideal <= 0:
        return 0
    return max(1, int(round((pi - 2 * c_polar) / a_ideal)))


def polar_colat(dim, N):
    """
    Colatitude of the North polar (top) spherical cap.

    Syntax:
        c_polar = polar_colat(dim, N)

    Description:
        Determines the colatitude of the North polar spherical cap of the
        partition of the sphere S^dim into `N` equal regions.

    Parameters:
        dim (int): Dimension of the sphere.
        N (int): Number of regions.

    Returns:
        float: Colatitude of the North polar cap.

    Notes:
        The colatitude is calculated differently depending on the value of `N`.

    Examples:
        >>> polar_colat(2, 4)
        1.0472

        >>> polar_colat(2, 10)
        0.6435

    See Also:
        num_collars, ideal_region_list
    """
    if N == 1:
        return pi
    elif N == 2:
        return pi / 2
    else:
        return sradius_of_cap(dim, area_of_ideal_region(dim, N))


def rot3(axis, angle):
    """
    R^3 rotation about a coordinate axis.

    Syntax:
        R = rot3(axis, angle)

    Description:
        Constructs a 3x3 rotation matrix for a rotation by `angle` (in radians)
        about the specified `axis`.

    Parameters:
        axis (int): Coordinate axis (1, 2, or 3).
        angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.

    Examples:
        >>> np.round(rot3(1, pi/6), 5)
        array([[ 1.    ,  0.    ,  0.    ],
               [ 0.    ,  0.86603, -0.5   ],
               [ 0.    ,  0.5   ,  0.86603]])

        >>> np.round(rot3(2, pi/6), 5)
        array([[ 0.86603,  0.    , -0.5   ],
               [ 0.    ,  1.    ,  0.    ],
               [ 0.5   ,  0.    ,  0.86603]])

        >>> np.round(rot3(3, pi/6), 5)
        array([[ 0.86603, -0.5   ,  0.    ],
               [ 0.5   ,  0.86603,  0.    ],
               [ 0.    ,  0.    ,  1.    ]])

    See Also:
        None
    """
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 1:
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ])
    elif axis == 2:
        R = np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c],
        ])
    elif axis == 3:
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ])
    else:
        raise ValueError("Axis must be 1, 2, or 3.")
    return R


def round_to_naturals(N, r_regions):
    """
    Round off a given list of numbers of regions.

    Syntax:
        n_regions = round_to_naturals(N, r_regions)

    Description:
        Rounds the ideal region counts `r_regions` to natural numbers while
        preserving the sum to `N`.

    Parameters:
        N (int): Total number of regions.
        r_regions (array-like): Ideal region counts.

    Returns:
        numpy.ndarray: Rounded natural numbers for region counts.

    Notes:
        The sum of `n_regions` matches `N`.

    Examples:
        >>> round_to_naturals(4, np.array([1.0, 2.0, 1.0]))
        array([1, 2, 1])

        >>> round_to_naturals(10, np.array([1.0, 4.0, 4.0, 1.0]))
        array([1, 4, 4, 1])

    See Also:
        None
    """
    n_regions = np.zeros_like(r_regions, dtype=int)
    discrepancy = 0
    for i, r in enumerate(r_regions):
        n_regions[i] = int(round(r + discrepancy))
        discrepancy += r - n_regions[i]
    assert sum(n_regions) == N, f"Sum of n_regions does not match N={N}"
    return n_regions



def s2_offset(points_1):
    """
    Experimental offset rotation of S^2.

    Syntax:
        rotation = s2_offset(points_1)

    Description:
        Sets `rotation` to be an R^3 rotation matrix which rotates the
        north pole of S^2 to a point specified by the points in `points_1`.

        `points_1` must be a 2xM matrix, representing M points of S^2 in
        spherical polar coordinates, with M a positive integer.

    Parameters:
        points_1 (numpy.ndarray): 2xM array of points in spherical polar
          coordinates.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.

    Examples:
        >>> points_1 = np.array([
        ...     [0, 0.78539819, 2.35619449, 3.92699082, 5.49778714, 0],
        ...     [0, 1.57079633, 1.57079633, 1.57079633, 1.57079633, 3.14159265]
        ... ])
        >>> np.round(s2_offset(points_1), 5)
        array([[ 0.    ,  0.7071,  0.7071],
               [-1.    ,  0.    ,  0.    ],
               [-0.    , -0.7071,  0.7071]])

    See Also:
        rot3, circle_offset
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
        a_2 = pi / 2
    return rot3(2, -a_2) @ rot3(3, -a_3)


def sphere_region(dim):
    """
    The sphere represented as a single region of an EQ partition.

    Syntax:
        region = sphere_region(dim)

    Description:
        Sets `region` to be an array of two points in spherical polar
        coordinates representing the sphere S^dim as a single region.

    Parameters:
        dim (int): Dimension of the sphere.

    Returns:
        numpy.ndarray: Array of shape (dim, 2).

    Examples:
        >>> sphere_region(1)
        array([[0.    , 6.28319]])

        >>> sphere_region(2)
        array([[0.    , 6.28319],
               [0.    , 3.14159]])

        >>> sphere_region(3)
        array([[0.    , 6.28319],
               [0.    , 3.14159],
               [0.    , 3.14159]])

    See Also:
        bot_cap_region, top_cap_region
    """
    if dim == 1:
        return np.array([[0, 2*pi]])
    else:
        sphere_region_1 = sphere_region(dim - 1)
        return np.hstack([
            np.vstack([sphere_region_1[:, 0], 0]),
            np.vstack([sphere_region_1[:, 1], pi])
        ])


def top_cap_region(dim, a_cap):
    """
    North polar (top) cap region of EQ partition.

    Syntax:
        region = top_cap_region(dim, a_cap)

    Description:
        Sets `region` to be an array of two points in spherical polar
        coordinates representing the North polar (top) cap of spherical
        radius `a_cap` as a region of the sphere S^dim.

    Parameters:
        dim (int): Dimension of the sphere.
        a_cap (float): Radius of the spherical cap.

    Returns:
        numpy.ndarray: Array of shape (dim, 2).

    Examples:
        >>> np.round(top_cap_region(1, pi/6), 5)
        array([[0.    , 0.5236]])

        >>> np.round(top_cap_region(2, pi/6), 5)
        array([[0.    , 6.28319],
               [0.    , 0.5236]])

        >>> np.round(top_cap_region(3, pi/6), 5)
        array([[0.    , 6.28319],
               [0.    , 3.14159],
               [0.    , 0.5236]])

    See Also:
        bot_cap_region, sphere_region
    """
    if dim == 1:
        return np.array([[0, a_cap]])
    else:
        sphere_region_1 = sphere_region(dim - 1)
        return np.hstack([
            np.vstack([sphere_region_1[:, 0], 0]),
            np.vstack([sphere_region_1[:, 1], a_cap])
        ]).T

