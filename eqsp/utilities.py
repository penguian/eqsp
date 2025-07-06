import numpy as np
from math import gamma, pi
from scipy.optimize import root_scalar
from scipy.special import betainc


def as_float(x):
    return float(x) if len(np.shape(x)) == 0 else x


def cart2polar2(x):
    """
    Convert from Cartesian to spherical coordinates on sphere S^2.

    Parameters
    ----------
    x : np.ndarray
        Array of real numbers of size (3, N), where N is any positive integer.

    Returns
    -------
    s : np.ndarray
        Array of size (2, N):
        rows are [phi; theta], phi in [0, 2*pi), theta in [0, pi].

    Notes
    -----
    cart2polar2(x) projects any x in R^3 onto the sphere S^2 via a line through
    the origin. The origin [0 0 0]' is itself projected onto a point on the
    equator such that

        polar2cart(cart2polar2([0, 0, 0])) == [1, 0, 0]

    Examples
    --------
    >>> x = np.array([[0, 0, 0, 0],
    ...               [0, 1, -1, 0],
    ...               [1, 0, 0, -1]])
    >>> s = cart2polar2(x)
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> s
    array([[0.    , 1.5708, 4.7124, 0.    ],
           [0.    , 1.5708, 1.5708, 3.1416]])
    """
    x = np.asarray(x, dtype=float)
    if x.shape[0] != 3:
        raise ValueError("Input x must have shape (3, N)")

    # Project zeros to [1, 0, 0]
    x_proj = x.copy()
    zero_mask = np.all(x == 0, axis=0)
    x_proj[:, zero_mask] = np.array([[1], [0], [0]])

    # Normalize to unit sphere
    norms = np.linalg.norm(x_proj, axis=0)
    x_unit = x_proj / norms

    # cart2sph: phi is azimuth, theta is elevation
    phi = np.arctan2(x_unit[1, :], x_unit[0, :]) % (2 * pi)
    theta = np.arccos(x_unit[2, :])  # angle from z+ axis, in [0, pi]

    s = np.vstack((phi, theta))
    return s


def polar2cart(s):
    """
    Convert spherical polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    s : np.ndarray
        Array of shape (dim, N) representing N points of S^dim
        in spherical polar coordinates.

    Returns
    -------
    x : np.ndarray
        Array of shape (dim+1, N) representing the Cartesian coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([[0, 1.5708, 4.7124, 0],
    ...               [0, 1.5708, 1.5708, 3.1416]])
    >>> x = polar2cart(s)
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> x
    array([[ 0., -0.,  0., -0.],
           [ 0.,  1., -1., -0.],
           [ 1., -0., -0., -1.]])
    """
    s = np.asarray(s, dtype=float)
    dim, n = s.shape
    x = np.zeros((dim+1, n))
    sinprod = np.ones(n)

    for k in range(dim, 1, -1):
        x[k, :] = sinprod * np.cos(s[k-1, :])
        sinprod = sinprod * np.sin(s[k-1, :])

    x[1, :] = sinprod * np.sin(s[0, :])
    x[0, :] = sinprod * np.cos(s[0, :])

    r = np.sqrt(np.sum(x**2, axis=0))
    mask = (r != 1)
    if np.any(mask):
        x[:, mask] = x[:, mask] / r[mask]

    return x


def euclidean_dist(x, y):
    """
    Compute the Euclidean distance between two N-vectors.

    Parameters
    ----------
    x : array_like
        First input vector. Must be one-dimensional.
    y : array_like
        Second input vector. Must be one-dimensional.

    Returns
    -------
    d : float
        The Euclidean distance between x and y.

    Examples
    --------
    >>> euclidean_dist([0, 0], [1, 0])
    1.0
    >>> euclidean_dist([1, 2, 3], [4, 5, 6])
    5.196152422706632
    >>> x = np.array([1, 0, 0])
    >>> y = np.array([0, 1, 0])
    >>> euclidean_dist(x, y)
    1.4142135623730951
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(
            "Both x and y must be one-dimensional arrays of the same length.")
    return as_float(np.linalg.norm(x - y))


def euc2sph_dist(e):
    """
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
    >>> euc2sph_dist(2)
    3.141592653589793
    >>> euc2sph_dist(np.arange(0, 2.5, 0.5))
    array([0.    , 0.5054, 1.0472, 1.6961, 3.1416])
    >>> euc2sph_dist(-2)
    -3.141592653589793
    """
    e = np.asarray(e)
    s = 2 * np.arcsin(e / 2)
    return as_float(s)


def spherical_dist(x, y):
    """
    Returns the spherical distance between two points x and y.

    Compute the spherical distances (angles in radians) between
    multiple pairs of points on S^M given in Cartesian coordinates,
    with arrays of shape (M, N).

    Parameters
    ----------
    x : array_like
        Array of shape (M, N), each column is a Cartesian vector.
    y : array_like
        Array of shape (M, N), each column is a Cartesian vector.

    Returns
    -------
    d : ndarray
        Array of shape (N,), containing spherical distances (in radians)
        between corresponding pairs.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> x = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]])
    >>> y = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    >>> spherical_dist(x, y)
    array([1.5708, 3.1416, 0.    ])

    >>> x = np.array([[1, 1], [0, 0], [0, 0]])
    >>> y = np.array([[0.5, 1], [np.sqrt(3)/2, 0], [0, 0]])
    >>> spherical_dist(x, y)
    array([1.0472, 0.    ])
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must both have shape (M, N)")

    # Normalize to unit vectors along columns
    x_norm = x / np.linalg.norm(x, axis=0, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=0, keepdims=True)
    # Dot product for each pair (across columns)
    dots = np.sum(x_norm * y_norm, axis=0)
    # Clamp for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    d = np.arccos(dots)
    return as_float(d)


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
    >>> sph2euc_dist(np.array([0, pi/2, pi]))
    array([0.    , 1.4142, 2.    ])
    >>> print(f"{sph2euc_dist(-pi/2):.4f}")
    -1.4142
    """
    s = np.asarray(s)
    e = 2.0 * np.sin(s / 2.0)
    return as_float(e)


def area_of_sphere(dim):
    """
    Returns the area of the unit sphere S^dim.

    The area is calculated as:
        area = 2 * pi^((dim+1)/2) / gamma((dim+1)/2)

    Parameters
    ----------
    dim : int or array-like of int
        Positive integer(s) indicating the sphere dimension(s).

    Returns
    -------
    area : float or np.ndarray
        Area(s) of the unit sphere(s).

    Examples
    --------
    >>> print(f"{area_of_sphere(1):.4f}")
    6.2832
    >>> print(f"{area_of_sphere(2):.4f}")
    12.5664
    >>> print(f'{area_of_sphere(3):.4f}')
    19.7392
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> area_of_sphere(np.arange(1, 8))
    array([ 6.2832, 12.5664, 19.7392, 26.3189, 31.0063, 33.0734, 32.4697])
    """
    dim = np.asarray(dim)
    power = (dim + 1) / 2
    area = np.asarray(2.0 * pi ** power / np.vectorize(gamma)(power))
    return as_float(area)


def volume_of_ball(dim):
    """
    Volume of the unit ball B^dim in R^dim.

    Parameters
    ----------
    dim : int or array-like
        Dimension(s) of the ball(s). Must be positive integer(s).

    Returns
    -------
    volume : float or ndarray
        Volume(s) of the unit ball(s).

    Notes
    -----
    The volume of B^dim is defined via the Lebesgue measure on R^dim.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> volume_of_ball(np.arange(1, 8))
    array([2.    , 3.1416, 4.1888, 4.9348, 5.2638, 5.1677, 4.7248])
    """
    dim = np.asarray(dim)
    return as_float(area_of_sphere(dim - 1) / dim)


def area_of_ideal_region(dim, N):
    """
    Area of one region of an EQ partition.

    AREA = area_of_ideal_region(dim, N) returns the area
    of one of N equal-area regions on the surface of a unit sphere S^dim,
    i.e., 1/N times area_of_sphere(dim).

    Parameters
    ----------
    dim : int
        Dimension of the sphere (must be positive).
    N : int or array-like of int
        Number(s) of regions (must be positive).

    Returns
    -------
    area : float or numpy.ndarray
        Area(s) of the ideal region(s).

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> area_of_ideal_region(3, 1)
    19.739208802178716
    >>> area_of_ideal_region(3, np.arange(1, 7))
    array([19.7392,  9.8696,  6.5797,  4.9348,  3.9478,  3.2899])
    """
    area = area_of_sphere(dim) / np.array(N)
    return as_float(area)


def area_of_cap(dim, s_cap):
    """
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

    For dim <= 2, and for dim==3 (when pi/6 <= s_cap <= pi*5/6),
    AREA is calculated in closed form, using the analytic solution of
    the definite integral.

    Otherwise, AREA is calculated using the incomplete Beta function ratio.

    References
    ----------
    [LeGS01 Lemma 4.1 p255]

    Examples
    --------
    >>> from math import pi
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> print(f"{area_of_cap(2, pi/2.0):.4f}")
    6.2832

    >>> area_of_cap(3, np.arange(0, pi+0.01, pi/4))
    array([ 0.    ,  1.7932,  9.8696, 17.946 , 19.7392])
    """
    s_cap = np.asarray(s_cap)
    if dim == 1:
        area = 2.0 * s_cap
    elif dim == 2:
        area = 4.0 * pi * np.sin(s_cap / 2.0) ** 2
    elif dim == 3:
        shape = s_cap.shape
        s_cap_flat = s_cap.ravel()
        area = np.zeros_like(s_cap_flat, dtype=np.float64)
        pole = (s_cap_flat < pi/6) | (s_cap_flat > pi*5/6)
        # Use incomplete beta function ratio near poles
        area[pole] = area_of_sphere(dim) * betainc(
            dim/2,
            dim/2,
            np.sin(s_cap_flat[pole]/2)**2)
        # Use closed form in the tropics
        trop_idx = ~pole
        trop = s_cap_flat[trop_idx]
        area[trop_idx] = (2 * trop - np.sin(2 * trop)) * pi
        area = area.reshape(shape)
    else:
        area = area_of_sphere(dim) * betainc(
            dim/2,
            dim/2,
            np.sin(s_cap/2)**2)
    return as_float(area)


def sradius_of_cap(dim, area):
    """
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

    Examples
    --------

    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> area = area_of_sphere(2) / 2
    >>> print(f"{sradius_of_cap(2, area):.4f}")
    1.5708

    >>> areas = np.arange(0, 5) * area_of_sphere(3) / 4
    >>> sradius_of_cap(3, areas)
    array([0.    , 1.1549, 1.5708, 1.9867, 3.1416])
    """
    area = np.asarray(area)

    if dim == 1:
        s_cap = area / 2
    elif dim == 2:
        s_cap = 2 * np.arcsin(np.sqrt(area / pi) / 2)
    else:
        orig_shape = area.shape
        flat_area = area.flatten()
        s_cap = np.zeros(flat_area.shape)
        for k, ak in enumerate(flat_area):
            asph = area_of_sphere(dim)
            if ak >= asph:
                s_cap[k] = pi
            else:
                flipped = False
                if 2 * ak > asph:
                    ak = asph - ak
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
    return as_float(s_cap)


def area_of_collar(dim, a_top, a_bot):
    """
    Area of spherical collar.

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
    area : float or ndarray
        Area(s) of the spherical collar(s).

    Notes
    -----
    a_top and a_bot must have the same shape.
    The area is defined via the Lebesgue measure on S^dim
    inherited from its embedding in R^(dim+1).

    References
    ----------
    [LeGS01 Lemma 4.1 p255]

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> area_of_collar(2, np.array([0, 1, 2]), np.array([1, 2, 3]))
    array([2.8884, 6.0095, 3.6056])
    """
    a_top = np.asarray(a_top)
    a_bot = np.asarray(a_bot)
    return as_float(area_of_cap(dim, a_bot) - area_of_cap(dim, a_top))
