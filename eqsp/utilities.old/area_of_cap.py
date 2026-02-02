import numpy as np
from scipy.special import betainc
from math import pi

def area_of_sphere(dim):
    """
    Area of a unit sphere in dimension dim.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.

    Returns
    -------
    float
        Area of the unit sphere in the given dimension.
    """
    from math import gamma
    return 2 * pi ** ((dim + 1) / 2) / gamma((dim + 1) / 2)

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
    >>> area_of_cap(2, np.pi/2)
    6.283185307179586

    >>> np.round(area_of_cap(3, np.arange(0, np.pi+0.01, np.pi/4)), 4)
    array([ 0.    ,  1.7932,  9.8696, 17.946 , 19.7392])
    """
    s_cap = np.asarray(s_cap)
    if dim == 1:
        area = 2 * s_cap
    elif dim == 2:
        area = 4 * pi * np.sin(s_cap / 2) ** 2
    elif dim == 3:
        shape = s_cap.shape
        s_cap_flat = s_cap.ravel()
        area = np.zeros_like(s_cap_flat, dtype=float)
        pole = (s_cap_flat < pi/6) | (s_cap_flat > pi*5/6)
        # Use incomplete beta function ratio near poles
        area[pole] = area_of_sphere(dim) * betainc(dim/2, dim/2, np.sin(s_cap_flat[pole]/2)**2)
        # Use closed form in the tropics
        trop_idx = ~pole
        trop = s_cap_flat[trop_idx]
        area[trop_idx] = (2 * trop - np.sin(2 * trop)) * pi
        area = area.reshape(shape)
    else:
        area = area_of_sphere(dim) * betainc(dim/2, dim/2, np.sin(s_cap/2)**2)
    return area
