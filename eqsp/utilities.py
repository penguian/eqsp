"""
PyEQSP utilities module.

Copyright 2026 Paul Leopardi
"""

from math import pi

import numpy as np
from scipy.optimize import newton
from scipy.special import betainc, gamma  # pylint: disable=no-name-in-module

TAU = 2.0 * pi

# Tolerance for comparisons close to zero.
_TOLERANCE = float(np.finfo(np.float32).eps)


def asfloat(x):
    """
    Convert from a numpy array to a float when this makes sense.

    It checks if the input is a 0-dimensional array (scalar) or a 1-element array,
    and if so, converts it to a standard Python `float`. Otherwise, it returns
    the input as a NumPy array. This ensures that functions return native Python
    scalars when appropriate (e.g., area of a single region) while still supporting
    vectorized operations.

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
        case (1, 1):
            return float(a[0, 0])
        case _:
            return a


def cart2polar2(x):
    """
    Convert from Cartesian to spherical coordinates on the manifold S^2 in R^3.

    Parameters
    ----------
    x : ndarray
        An array of real numbers of shape (3, N).
        Each column represents a point in 3D Cartesian coordinates.

    Returns
    -------
    s : ndarray
        An array of shape (2, N), where for each point:
        - s[0, :] is the longitude phi in [0, TAU),
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
    if np.min(norms) < _TOLERANCE:
        raise ValueError("Input x must not contain the origin")
    x_proj = x / norms

    # Spherical coordinates: phi = atan2(y, x), theta = arccos(z)
    phi = np.arctan2(x_proj[1, :], x_proj[0, :]) % TAU
    theta = np.arccos(x_proj[2, :])

    s = np.vstack((phi, theta))
    return s


def polar2cart(s):
    """
    Convert spherical polar to Cartesian coordinates.

    Parameters
    ----------
    s : array_like
        Array of real numbers of shape (dim, N) representing N points of the
        sphere as a manifold: S^dim in R^(dim+1).

    Returns
    -------
    x : ndarray
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
    if s.ndim == 1:
        s = s[:, np.newaxis]
        was_1d = True
    else:
        was_1d = False
    dim, n = s.shape
    x = np.zeros((dim + 1, n))
    sinprod = np.ones(n)
    for k in range(dim, 1, -1):
        x[k, :] = sinprod * np.cos(s[k - 1, :])
        sinprod = sinprod * np.sin(s[k - 1, :])
    x[1, :] = sinprod * np.sin(s[0, :])
    x[0, :] = sinprod * np.cos(s[0, :])
    if was_1d:
        return x.flatten()
    return x


def euc2sph_dist(e):
    """
    Convert Euclidean to spherical distance.

    Parameters
    ----------
    e : float or array-like
        A real number or array of real numbers, with ``abs(e) <= 2``.

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
    Euclidean distance between two points in Cartesian coordinates.

    Parameters
    ----------
    x : array_like, shape (M, N)
        Array of shape (M, N), where M = dim+1, and dim is the dimension of the
        sphere as a manifold: S^dim in R^(dim+1).
    y : array_like, shape (M, N)
        Array of shape (M, N). The shapes of x and y must be identical.

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
    Spherical distance between two points in Cartesian coordinates.

    Parameters
    ----------
    x : array_like, shape (M, N)
        Array of shape (M, N), where M = dim+1, and dim is the dimension of the
        sphere as a manifold: S^dim in R^(dim+1).
    y : array_like, shape (M, N)
        Array of shape (M, N). The shapes of x and y must be identical.

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
    Returns the area of the unit sphere as a manifold: S^dim in R^(dim+1).

    Parameters
    ----------
    dim : int or array-like of int
        The dimension of the sphere as a manifold: S^dim in R^(dim+1).

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
    area = np.asarray(2.0 * pi**power / gamma(power))
    return asfloat(area)


def volume_of_ball(dim):
    """
    Volume of the unit ball B^dim in R^dim.

    Parameters
    ----------
    dim : int or array-like
        The dimension of the ball B^dim in R^dim.

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
    Area of one region of an EQ partition.

    Parameters
    ----------
    dim : int
        The dimension of the sphere as a manifold: S^dim in R^(dim+1).
    N : int or array-like
        The number of regions in the partition.

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


def ideal_collar_angle(dim, N):
    """
    The ideal angle for spherical collars of an EQ partition.

    Parameters
    ----------
    dim : int
        The dimension of the sphere as a manifold: S^dim in R^(dim+1).
    N : int or array-like
        The number of regions in the partition.

    Returns
    -------
    angle : float or np.ndarray
        The ideal angle(s).

    Notes
    -----
    The ideal collar angle is determined by the side of a dim-dimensional
    hypercube of the same volume as the area of single region of S^dim.

    See Also
    --------
    area_of_ideal_region

    Examples
    --------
    >>> print(f"{ideal_collar_angle(2, 10):.4g}")
    1.121
    >>> np.round(ideal_collar_angle(3, np.arange(1,7)), 4)
    array([2.7026, 2.145 , 1.8739, 1.7025, 1.5805, 1.4873])
    """
    return asfloat(area_of_ideal_region(dim, N) ** (1 / dim))


def area_of_cap(dim, s_cap):
    """
    Area of spherical cap on the manifold S^dim in R^(dim+1).

    Parameters
    ----------
    dim : int
        The dimension of the sphere as a manifold: S^dim in R^(dim+1).
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
            near_pole = (s_cap_flat < MIN_TROPICAL) | (s_cap_flat > MAX_TROPICAL)
            # Use incomplete beta function ratio near poles
            area[near_pole] = area_of_sphere(dim) * betainc(
                dim / 2, dim / 2, np.sin(s_cap_flat[near_pole] / 2) ** 2
            )
            # Use closed form in the tropics
            s_cap_trop = s_cap_flat[~near_pole]
            area[~near_pole] = (2 * s_cap_trop - np.sin(2 * s_cap_trop)) * pi
            area = area.reshape(shape)
        case _:
            area = area_of_sphere(dim) * betainc(
                dim / 2, dim / 2, np.sin(s_cap / 2) ** 2
            )
    return asfloat(area)


def sradius_of_cap(dim, area):
    """
    Spherical radius of a spherical cap of given area.

    Parameters
    ----------
    dim : int
        The dimension of the sphere as a manifold: S^dim in R^(dim+1).
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
    sphere_area = area_of_sphere(dim)

    if np.any(area < 0):
        raise ValueError("Area must be non-negative.")
    if np.any(area > sphere_area * (1 + 1e-10)):
        raise ValueError(f"Area {np.max(area)} exceeds area of sphere {sphere_area}.")

    if dim == 1:
        s_cap = np.clip(area / 2, 0, pi)
    elif dim == 2:
        # Avoid nan for area slightly > sphere_area by clipping to [0, 1]
        arg = np.clip(area / sphere_area, 0.0, 1.0)
        s_cap = 2 * np.arcsin(np.sqrt(arg))
    else:
        orig_shape = area.shape
        flat_area = np.ravel(area)
        s_cap = np.zeros_like(flat_area, dtype=float)

        # Handle cases matching or exceeding full sphere area
        full_idx = flat_area >= sphere_area
        s_cap[full_idx] = pi

        # Handle zero or negative area (ValueError caught negative areas)
        zero_idx = flat_area <= 0
        s_cap[zero_idx] = 0.0

        # Process remaining cases
        calc_idx = ~(full_idx | zero_idx)
        if np.any(calc_idx):
            ak_calc = flat_area[calc_idx]

            # Mirror areas greater than half the sphere for better convergence
            flipped = 2 * ak_calc > sphere_area
            ak_target = np.where(flipped, sphere_area - ak_calc, ak_calc)

            # Pre-allocate inputs for vectorized optimization
            # Start with a good initial guess (e.g. midpoint of remaining space)
            x0 = np.full_like(ak_target, pi / 2)

            def area_diff(s):
                # Vectorized difference function for Newton solver.
                return area_of_cap(dim, s) - ak_target

            def area_diff_prime(s):
                # Derivative of area of cap with respect to s.
                # Strictly: d/ds Area(dim, s) = Area(dim-1, s) based on S^{dim-1}
                # radius sin(s).
                return area_of_sphere(dim - 1) * (np.sin(s) ** (dim - 1))

            # Find roots using vectorized newton
            sk = newton(area_diff, x0, fprime=area_diff_prime)

            # Ensure roots are clamped within [0, pi]
            sk = np.clip(sk, 0.0, pi)

            # Map back flipped variants
            s_cap[calc_idx] = np.where(flipped, pi - sk, sk)

        s_cap = s_cap.reshape(orig_shape)
    return asfloat(s_cap)


def area_of_collar(dim, a_top, a_bot):
    """
    Area of a spherical collar.

    Parameters
    ----------
    dim : int
        The dimension of the sphere as a manifold: S^dim in R^(dim+1).
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


def x2stereo(x):
    """
    Stereographic projection of Euclidean points.

    Parameters
    ----------
    x : ndarray
        Points in R^(dim+1), shape (dim+1, N).

    Returns
    -------
    result : ndarray
        Projected points in R^dim, shape (dim, N).
    """
    x = np.asarray(x)
    dim = x.shape[0] - 1

    last = x[dim, :]
    mask = np.isclose(last, 1.0)

    scale = np.ones(x.shape[1])
    scale[~mask] = 1.0 - last[~mask]

    with np.errstate(divide="ignore"):
        result = x[:dim, :] / scale

    result[:, mask] = np.nan
    return result


def x2eqarea(x):
    """
    Equal area projection of Euclidean points.

    Parameters
    ----------
    x : ndarray
        Points in R^(dim+1), shape (dim+1, N).

    Returns
    -------
    result : ndarray
        Projected points in R^dim, shape (dim, N).
    """
    x = np.asarray(x)
    dim = x.shape[0] - 1
    last = x[dim, :]

    theta = np.arccos(np.clip(-last, -1.0, 1.0))
    a_cap = area_of_cap(dim, theta)
    v_ball = volume_of_ball(dim)
    r = (a_cap / v_ball) ** (1.0 / dim)

    sin_theta = np.sin(theta)
    mask = np.isclose(sin_theta, 0.0)

    scale = np.zeros_like(theta)
    scale[~mask] = r[~mask] / sin_theta[~mask]

    result = np.zeros((dim, x.shape[1]))
    result[:, ~mask] = x[:dim, ~mask] * scale[~mask]
    return result


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
