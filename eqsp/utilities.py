"""
    EQSP: Recursive Zonal Equal Area Sphere Partitioning.
    Copyright 2025 Paul Leopardi.
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.
"""


import numpy as np
from math import pi
from scipy.optimize import root_scalar
from scipy.special import betainc, gamma


# Tolerance for comparisons close to zero.
tolerance = float(np.finfo(np.float32).eps)


def asfloat(x):
    """
    a = asfloat(x)

    Convert from a Numpy array to a float when this makes sense.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    a : ndarray or float
        If x is a (), or (1,) or (1,1) array, then a is returned as a float.
        Otherwise a is just x as a Numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> x0 = 12.789
    >>> a0 = asfloat(x0)
    >>> print(a0)
    12.789
    >>> x1 = [[22.546]]
    >>> a1 = asfloat(x1)
    >>> print(a1)
    22.546
    >>> x2 = [12.789, 22.546]
    >>> a2 = asfloat(x2)
    >>> print(a2)
    [12.789 22.546]
    >>> x3 = np.array([12.789, 22.546])
    >>> a3 = asfloat(x3)
    >>> print(a3)
    [12.789 22.546]
    """
    a = np.asarray(x, dtype=float)
    match a.shape:
        case ():
            return float(a)
        case (1,):
            return float(a[0])
        case (1,1):
            return float(a[0, 0])
        case _:
            return a


def cart2polar2(x):
    """
    s = cart2polar2(x)

    Convert from Cartesian to spherical coordinates on sphere S^2.

    Parameters
    ----------
    x : ndarray
        An array of real numbers of shape (3, N), where N is any positive integer.
        Each column represents a point in 3D Cartesian coordinates.

    Returns
    -------
    s : ndarray
        An array of shape (2, N), where for each point:
        - s[0, :] is the longitude phi in [0, 2*pi),
        - s[1, :] is the colatitude theta in [0, pi].

    See Also
    --------
    polar2cart

    Notes
    -----
    This function projects any X in R^3 onto the unit sphere S^2 via a line
    through the origin [0, 0, 0]'.
    If X includes the origin, this results in a ValueError exception.

    Examples
    --------
    >>> import numpy as np
    >>> x0 = np.array([[ 0., 0.,  0.,  0.],
    ...                [ 0., 1., -1.,  0.],
    ...                [ 1., 0.,  0., -1.]])
    >>> s0 = cart2polar2(x0)
    >>> print(s0)
    [[0.     1.5708 4.7124 0.    ]
     [0.     1.5708 1.5708 3.1416]]
    >>> x1 = np.array([[ 0., 0.,  0.,  0.],
    ...                [ 0., 1., -1.,  0.],
    ...                [ 0., 0.,  0.,  0.],
    ...                [ 1., 0.,  0., -1.]])
    >>> s1 = cart2polar2(x1)
    Traceback (most recent call last):
        ...
    ValueError: Input x must have shape (3, N)
    >>> x2 = np.array([[ 0., 0.,  0.,  0.],
    ...                [ 0., 0., -1.,  0.],
    ...                [ 1., 0.,  0., -1.]])
    >>> s2 = cart2polar2(x2)
    Traceback (most recent call last):
        ...
    ValueError: Input x must not contain the origin
    """
    x = np.asarray(x)
    if x.shape[0] != 3:
        raise ValueError("Input x must have shape (3, N)")

    # Project any x onto the unit sphere S^2 by normalizing (except for origin)
    norms = np.linalg.norm(x, axis=0)
    # If one or more points is the origin, raise ValueError
    if np.any(norms < tolerance):
        raise ValueError("Input x must not contain the origin")
    x_proj = x / norms

    # Spherical coordinates: phi = atan2(y, x), theta = arccos(z)
    phi = np.arctan2(x_proj[1, :], x_proj[0, :]) % (2 * np.pi)
    theta = np.arccos(x_proj[2, :])

    s = np.vstack((phi, theta))
    return s


def polar2cart(s):
    """
    x = polar2cart(s)

    Convert spherical polar to Cartesian coordinates.

    Parameters
    ----------
    s : numpy.ndarray
        Array of real numbers of shape (dim, N) representing N points of S^dim
        in spherical polar coordinates, where dim and N are positive integers.

    Returns
    -------
    x : numpy.ndarray
        Array of shape (dim+1, N) containing the Cartesian coordinates of the
        points represented by the spherical polar coordinates s.

    See Also
    --------
    cart2polar2

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> s = np.array([
    ...     [0, pi/2, 3*pi/2, 0],
    ...     [0, pi/2, pi/2,   pi]])
    >>> x = polar2cart(s)
    >>> print(x)
    [[ 0.  0. -0.  0.]
     [ 0.  1. -1.  0.]
     [ 1.  0.  0. -1.]]
    """
    s = np.asarray(s)
    dim, n = s.shape
    x = np.zeros((dim + 1, n))
    sinprod = np.ones(n)
    for k in range(dim, 1, -1):
        x[k, :] = sinprod * np.cos(s[k - 1, :])
        sinprod = sinprod * np.sin(s[k - 1, :])
    x[1, :] = sinprod * np.sin(s[0, :])
    x[0, :] = sinprod * np.cos(s[0, :])
    r = np.linalg.norm(x, axis=0)
    mask = np.abs(r - 1) > tolerance
    if np.any(mask):
        x[:, mask] = x[:, mask] / r[mask]
    return x


def euc2sph_dist(e):
    """
    s = euc2sph_dist(e)

    Convert Euclidean to spherical distance.

    Parameters
    ----------
    e : float or array-like
        A real number or array of real numbers, with |e| <= 2.

    Returns
    -------
    s : float or ndarray
        The spherical distance(s), same shape as e.

    Notes
    -----
    The argument e is assumed to satisfy abs(e) <= 2.
    The formula is valid for the unit sphere in all dimensions.

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> print(f"{euc2sph_dist(2):.4f}")
    3.1416
    >>> euc2sph_dist(np.array([0, np.sqrt(2), 2.0]))
    array([0.    , 1.5708, 3.1416])
    >>> print(f"{euc2sph_dist(-2):.4f}")
    -3.1416
    """
    e = np.asarray(e)
    s = 2.0 * np.arcsin(e / 2.0)
    return asfloat(s)


def sph2euc_dist(s):
    """
    e = sph2euc_dist(s)

    Convert spherical distance to Euclidean distance on the unit sphere.

    Parameters
    ----------
    s : float or array_like
        Spherical distance(s), in radians.

    Returns
    -------
    e : float or ndarray
        Euclidean (chord) distance(s), same shape as input.

    Examples
    --------
    >>> from math import pi
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> sph2euc_dist(0)
    0.0
    >>> sph2euc_dist(pi)
    2.0
    >>> print(sph2euc_dist(np.array([0, pi/4, pi/2, 3*pi/4, pi])))
    [0.     0.7654 1.4142 1.8478 2.    ]
    >>> print(f"{sph2euc_dist(-pi/2):.4f}")
    -1.4142
    """
    s = np.asarray(s)
    e = 2.0 * np.sin(s / 2.0)
    return asfloat(e)


def euclidean_dist(x, y):
    """
    d = euclidean_dist(x, y)

    Euclidean distance between two points in Cartesian coordinates.

    Parameters
    ----------
    x : array_like, shape (M, N)
        Array of shape (M, N), where each column is a Cartesian vector.
    y : array_like, shape (M, N)
        Array of shape (M, N), where each column is a Cartesian vector.
        The shapes of x and y must be identical.

    Returns
    -------
    d : ndarray, shape (N,)
        The Euclidean distance between corresponding pairs of points in x and y.

    See Also
    --------
    spherical_dist
    euc2sph_dist
    sph2euc_dist


    Examples
    --------
    >>> x = np.array([[0,     0,      0,     0],
    ...               [0,     1,     -1,     0],
    ...               [1,     0,      0,    -1]])
    >>> y = np.array([[ 0,    0,      0,     0],
    ...               [-0.5,  0.866, -0.866, 0.5],
    ...               [ 0.866,0.5,   -0.5,  -0.866]])
    >>> euclidean_dist(x, y)
    array([0.5176, 0.5176, 0.5176, 0.5176])


    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")

    # Compute the Euclidean distance
    return asfloat(np.sqrt(np.sum((x - y) ** 2, axis=0)))


def spherical_dist(x, y):
    """
    d = spherical_dist(x, y)

    Returns the spherical distance between two arrays of points x and y.

    Parameters
    ----------
    x : array_like
        Array of shape (M, N), where each column is a Cartesian vector.
    y : array_like
        Array of shape (M, N), where each column is a Cartesian vector.
        The shapes of x and y must be identical.

    Returns
    -------
    d : ndarray
        Array of shape (N,), containing spherical distances (in radians)
        between corresponding pairs.

    See Also
    --------
    euclidean_dist
    euc2sph_dist
    sph2euc_dist

    Examples
    --------
    >>> x0 = np.array([[0,     0,      0,     0],
    ...                [0,     1,     -1,     0],
    ...                [1,     0,      0,    -1]])
    >>> y0 = np.array([[ 0,    0,      0,     0],
    ...                [-0.5,  0.866, -0.866, 0.5],
    ...                [ 0.866,0.5,   -0.5,  -0.866]])
    >>> print(spherical_dist(x0, y0))
    [0.5236 0.5236 0.5236 0.5236]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must both have shape (M, N)")

    # Compute dot product for each point (across columns)
    dots = np.sum(x * y, axis=0)
    # Clip dot product to [-1, 1] to avoid numerical errors outside acos domain
    dots = np.clip(dots, -1.0, 1.0)
    return asfloat(np.arccos(dots))


def area_of_sphere(dim):
    """
    a = area_of_sphere(dim)

    Returns the area of the unit sphere S^dim.

    Parameters
    ----------
    dim : int or array-like of int
        Dimension(s) of the sphere(s). Must be positive integer(s).

    Returns
    -------
    a : float or np.ndarray
        Area(s) of the unit sphere(s).

    Notes
    -----
    The area of S^dim is defined via the Lebesgue measure on S^dim
    inherited from its embedding in R^(dim+1).

    The area is calculated as:
        a = 2 * pi^((dim+1)/2) / gamma((dim+1)/2)

    References
    ----------
    [Mue98] p39.

    See Also
    --------
    volume_of_ball


    Examples
    --------
    >>> area_of_sphere(range(1, 8))
    array([ 6.2832, 12.5664, 19.7392, 26.3189, 31.0063, 33.0734, 32.4697])
    """
    dim = np.asarray(dim)
    power = (dim + 1) / 2
    area = np.asarray(2.0 * pi ** power / gamma(power))
    return asfloat(area)


def volume_of_ball(dim):
    """
    v = volume_of_ball(dim)

    Volume of the unit ball B^dim in R^dim.

    Parameters
    ----------
    dim : int or array-like
        Dimension(s) of the ball(s). Must be positive integer(s).

    Returns
    -------
    v : float or ndarray
        Volume(s) of the unit ball(s).

    Notes
    -----
    The volume of B^dim is defined via the Lebesgue measure on R^dim.

    References
    ----------
    [WeiMW].

    See Also
    --------
    area_of_sphere

    Examples
    --------
    >>> import numpy as np
    >>> volume_of_ball(range(1, 8))
    array([2.    , 3.1416, 4.1888, 4.9348, 5.2638, 5.1677, 4.7248])
    """
    dim = np.asarray(dim)
    return asfloat(area_of_sphere(dim - 1) / dim)


def area_of_ideal_region(dim, N):
    """
    a = area_of_ideal_region(dim, N)

    Area of one region of an EQ partition.

    This function returns the area of one of N equal-area regions of
    a unit sphere S^dim, i.e., 1/N times area_of_sphere(dim).

    Parameters
    ----------
    dim : int
        Dimension of the sphere (must be positive).
    N : int or array-like of int
        Number(s) of regions (must be positive).

    Returns
    -------
    a : float or numpy.ndarray with the same shape as N
        Area(s) of the ideal region(s).

    See Also
    --------
    area_of_sphere

    Examples
    --------
    >>> area_of_ideal_region(3, range(1, 7))
    array([19.7392,  9.8696,  6.5797,  4.9348,  3.9478,  3.2899])
    """
    area = area_of_sphere(dim) / np.array(N)
    return asfloat(area)


def area_of_cap(dim, s_cap):
    """
    a = area_of_cap(dim, s_cap)

    Area of spherical cap on S^dim of spherical radius s_cap.

    Parameters
    ----------
    dim : int
        Positive integer, the dimension of the sphere.
    s_cap : float or array_like
        Spherical radius/radii of the cap(s), in [0, pi].

    Returns
    -------
    area : float or ndarray
        Area(s) of the spherical cap(s).

    Notes
    -----
    The area is defined via the Lebesgue measure on S^dim inherited from
    its embedding in R^(dim+1).

    For dim <= 2, and for dim==3 (when pi/6 <= s_cap <= 5*pi/6),
    AREA is calculated in closed form, using the analytic solution of
    the definite integral.

    Otherwise, AREA is calculated using the incomplete Beta function ratio.

    References
    ----------
    [LeGS01 Lemma 4.1 p255]

    See Also
    --------
    sradius_of_cap

    Examples
    --------
    >>> print(f"{area_of_cap(2, pi/2):.4f}")
    6.2832
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> print(area_of_cap(3, np.linspace(0, pi, 5)))
    [ 0.      1.7932  9.8696 17.946  19.7392]
    """
    s_cap = np.asarray(s_cap)
    match dim:
        case 1:
            area = 2.0 * s_cap
        case 2:
            area = 4.0 * pi * np.sin(s_cap / 2.0) ** 2
        case 3:
            shape = s_cap.shape
            s_cap_flat = s_cap.ravel()
            area = np.zeros_like(s_cap_flat, dtype=np.float64)
            MIN_TROPICAL = pi / 6
            MAX_TROPICAL = 5 * pi / 6
            near_pole = (
                (s_cap_flat < MIN_TROPICAL) | (s_cap_flat > MAX_TROPICAL))
            # Use incomplete beta function ratio near poles
            area[near_pole] = area_of_sphere(dim) * betainc(
                dim/2,
                dim/2,
                np.sin(s_cap_flat[near_pole]/2)**2)
            # Use closed form in the tropics
            s_cap_trop = s_cap_flat[~near_pole]
            area[~near_pole] = (2 * s_cap_trop - np.sin(2 * s_cap_trop)) * pi
            area = area.reshape(shape)
        case _:
            area = area_of_sphere(dim) * betainc(
                dim/2,
                dim/2,
                np.sin(s_cap/2)**2)
    return asfloat(area)


def sradius_of_cap(dim, area):
    """
    s_cap = sradius_of_cap(dim, area)

    Spherical radius of a spherical cap of given area on S^dim.

    Parameters
    ----------
    dim : int
        Dimension of the sphere (must be >= 1).
    area : float or array-like
        Area(s) of the cap(s).

    Returns
    -------
    s_cap : float or ndarray
        Spherical radius/radii in [0, pi], same shape as area.

    Notes
    -----
    For dim <= 2, the result is exact (closed form).
    For dim > 2, the result is found numerically.

    See Also
    --------
    area_of_cap

    Examples
    --------

    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> area = area_of_sphere(2) / 2
    >>> print(f"{sradius_of_cap(2, area):.4f}")
    1.5708

    >>> areas = np.linspace(0, 4, 5) * area_of_sphere(3) / 4
    >>> sradius_of_cap(3, areas)
    array([0.    , 1.1549, 1.5708, 1.9867, 3.1416])
    """
    area = np.asarray(area)

    if dim == 1:
        s_cap = area / 2
    elif dim == 2:
        s_cap = 2 * np.arcsin(np.sqrt(area / (4*pi)))
    else:
        orig_shape = area.shape
        flat_area = area.flatten()
        s_cap = np.zeros(flat_area.shape)
        sphere_area = area_of_sphere(dim)
        for k, ak in enumerate(flat_area):
            if ak >= sphere_area:
                s_cap[k] = pi
            else:
                flipped = False
                if 2 * ak > sphere_area:
                    ak = sphere_area - ak
                    flipped = True

                def area_diff(s):
                    # Define the difference function for root finding.
                    return area_of_cap(dim, s) - ak
                # Find root in [0, pi]
                result = root_scalar(
                    area_diff,
                    bracket=[0, pi],
                    method='bisect')
                sk = result.root
                s_cap[k] = pi - sk if flipped else sk
        s_cap = s_cap.reshape(orig_shape)
    return asfloat(s_cap)


def area_of_collar(dim, a_top, a_bot):
    """
    a = area_of_collar(dim, a_top, a_bot)

    Area of a spherical collar.

    Parameters
    ----------
    dim : int
        Positive integer, the dimension of the sphere.
    a_top : float or array-like
        Top (smaller) spherical radius/radii, in [0, pi].
    a_bot : float or array-like
        Bottom (larger) spherical radius/radii, in [0, pi].

    Returns
    -------
    a : float or ndarray
        Area(s) of the spherical collar(s).

    Notes
    -----
    a_top and a_bot must have the same shape.
    The area is defined via the Lebesgue measure on S^dim
    inherited from its embedding in R^(dim+1).

    References
    ----------
    [LeGS01 Lemma 4.1 p255]

    See Also
    --------
    area_of_cap

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> area_of_collar(2, np.array([0, 1, 2]), np.array([1, 2, 3]))
    array([2.8884, 6.0095, 3.6056])
    """
    a_top = np.asarray(a_top)
    a_bot = np.asarray(a_bot)
    return asfloat(area_of_cap(dim, a_bot) - area_of_cap(dim, a_top))
