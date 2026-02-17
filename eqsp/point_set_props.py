"""
EQSP Point Set Properties module.
"""

import math
import numpy as np
from .partitions import eq_point_set
from .utilities import (
    area_of_cap,
    area_of_sphere,
    euc2sph_dist,
)


def calc_dist_coeff(dim, N, min_euclidean_dist):
    """
    Coefficient of minimum distance.

    Parameters
    ----------
    dim : int
        Number of dimensions, must be positive integer.
    N : int or array-like
        Number of regions, must be positive integer(s).
    min_euclidean_dist : array-like
        Minimum Euclidean distance(s), same shape as N.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.

    Notes
    -----
    The expression for the lower bound on minimum distance of a minimum
    r^(-s) energy point set on S^dim was given by [RakSZ95] for s == 0,
    dim = 2, [Dahl78] for s == dim-1, [KuiSS04 Theorem 8] for dim-1 <= s < dim,
    and [KuiS98 (1.12) p. 525] for s > dim.

    See Also
    --------
    eq_min_dist, eq_dist_coeff

    Examples
    --------
    >>> N = np.arange(2, 7)
    >>> dist = eq_min_dist(2, N)
    >>> calc_dist_coeff(2, N, dist)
    array([2.82842712, 2.44948974, 2.82842712, 3.16227766, 3.46410162])
    """
    return min_euclidean_dist * np.power(N, 1 / dim)


def calc_energy_coeff(dim, N, s, energy):
    """
    Coefficient of second term in expansion of energy.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    s : float
        Exponent parameter.
    energy : array-like
        Energy values, same shape as N.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.

    Notes
    -----
    The energy expansion is not valid for N == 1,
    and in particular, EQ_ENERGY_COEFF(dim, N, 0, energy) := 0.

    For s > 0, [KuiS98 (1.6) p524] has
    E(dim, N, s) == (SPHERE_INT_ENERGY(dim, s)/2) N^2 + COEFF N^(1+s/dim) + ...

    For s == 0 (logarithmic potential), see [SafK97 (4) p7].

    See Also
    --------
    eq_energy_dist, eq_energy_coeff

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> N = np.arange(2, 7)
    >>> energy, dist = eq_energy_dist(2, N, 0)
    >>> calc_energy_coeff(2, N, 0, energy)
    array([-0.2213, -0.1569, -0.2213, -0.2493, -0.2569])
    """

    def sphere_int_energy_inner(dim, s_val):
        if s_val != 0:
            return (
                math.gamma((dim + 1) / 2)
                * math.gamma(dim - s_val)
                / (math.gamma((dim - s_val + 1) / 2) * math.gamma(dim - s_val / 2))
            )
        if dim != 1:
            from scipy.special import psi

            return (psi(dim) - psi(dim / 2) - math.log(4)) / 2
        return 0

    N = np.asarray(N)
    energy = np.asarray(energy)
    if s > 0:
        first_term = (sphere_int_energy_inner(dim, s) / 2) * np.power(N, 2)
        coeff = (energy - first_term) / np.power(N, 1 + s / dim)
    else:
        shape = N.shape
        n_partitions = int(np.prod(shape))
        N_flat = N.reshape(1, n_partitions)
        first_term = (sphere_int_energy_inner(dim, s) / 2) * np.power(N_flat, 2)
        coeff = np.zeros_like(N_flat, dtype=float)
        neq1 = N_flat != 1
        coeff[neq1] = (
            energy.reshape(1, n_partitions)[0][neq1.ravel()]
            - first_term[0][neq1.ravel()]
        ) / (N_flat[0][neq1.ravel()] * np.log(N_flat[0][neq1.ravel()]))
        coeff = coeff.reshape(shape)
    return coeff


def sphere_int_energy(dim, s):
    """
    Energy integral of r^(-s) potential.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    s : float
        Exponent parameter.

    Returns
    -------
    energy : float
        Energy integral on S^dim of the r^(-s) potential.

    Notes
    -----
    Ref for s > 0: [KuiS98 (1.6) p524]
    Ref for s == 0 and dim == 2: [SafK97 (4) p. 7]

    See Also
    --------
    eq_energy_dist, calc_energy_coeff

    Examples
    --------
    >>> sphere_int_energy(2, 0)
    np.float64(-0.1931471805599453)
    """
    if s != 0:
        return (
            math.gamma((dim + 1) / 2)
            * math.gamma(dim - s)
            / (math.gamma((dim - s + 1) / 2) * math.gamma(dim - s / 2))
        )
    if dim != 1:
        from scipy.special import psi

        return (psi(dim) - psi(dim / 2) - math.log(4)) / 2
    return 0


def calc_packing_density(dim, N, min_euclidean_dist):
    """
    Density of packing given by minimum distance.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    min_euclidean_dist : array-like
        Minimum Euclidean distance(s), same shape as N.

    Returns
    -------
    density : array-like
        Density values, same shape as N.

    Notes
    -----
    The packing density is defined as the sum of the areas of the spherical
    caps divided by the area of the unit sphere S^dim.

    The spherical radius of the caps is half the minimum spherical distance.
    For N == 1, the spherical radius is pi.

    See Also
    --------
    eq_min_dist, area_of_cap, area_of_sphere, eq_packing_density

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> N = np.arange(2, 7)
    >>> dist = eq_min_dist(2, N)
    >>> calc_packing_density(2, N, dist)
    array([1.    , 0.4393, 0.5858, 0.7322, 0.8787])
    """
    s_cap = euc2sph_dist(min_euclidean_dist) / 2
    s_cap = np.array(s_cap)
    N = np.asarray(N)
    s_cap[N == 1] = np.pi
    density = N * area_of_cap(dim, s_cap) / area_of_sphere(dim)
    return density


def eq_dist_coeff(dim, N, extra_offset=False):
    """
    Coefficient of minimum distance of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.
    """
    dist = eq_min_dist(dim, N, extra_offset=extra_offset)
    coeff = dist * np.power(N, 1 / dim)
    return coeff


def eq_energy_coeff(dim, N, s=None, extra_offset=False):
    """
    Coefficient in expansion of energy of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    s : float, optional
        Exponent parameter. Defaults to dim-1.
    extra_offset : bool, optional
        Use extra offsets. Default False.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.
    """
    if s is None:
        s = dim - 1
    dist_result = eq_energy_dist(dim, N, s=s, extra_offset=extra_offset)
    if isinstance(dist_result, tuple):
        energy = dist_result[0]
    else:
        energy = dist_result
    coeff = calc_energy_coeff(dim, N, s, energy)
    return coeff


def eq_energy_dist(dim, N, s=None, extra_offset=False):
    """
    Energy and minimum distance of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    s : float, optional
        Exponent parameter. Defaults to dim-1.
    extra_offset : bool, optional
        Use extra offsets. Default False.

    Returns
    -------
    energy : array-like
        Energy values, same shape as N.
    dist : array-like, optional
        Minimum Euclidean distance(s), same shape as N.
    """
    if s is None:
        s = dim - 1

    shape = np.shape(N)
    N_flat = np.reshape(N, (1, int(np.prod(shape))))
    energy = np.zeros_like(N_flat, dtype=float)
    dist = np.zeros_like(N_flat, dtype=float)
    for i, n_val in enumerate(N_flat[0]):
        points = eq_point_set(dim, n_val, extra_offset)
        if len(energy.shape) > 1 or len(dist.shape) > 1:
            energy[0, i], dist[0, i] = point_set_energy_dist(points, s)
        else:
            energy[0, i] = point_set_energy_dist(points, s)
    energy = energy.reshape(shape)
    dist = dist.reshape(shape)
    if len(dist.shape) > 0:
        return energy, dist
    return energy


def eq_min_dist(dim, N, extra_offset=False):
    """
    Minimum distance between center points of an EQ partition.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.

    Returns
    -------
    dist : array-like
        Minimum Euclidean distance(s), same shape as N.
    """
    return eq_point_set_property(point_set_min_dist, dim, N, extra_offset=extra_offset)


def eq_packing_density(dim, N, extra_offset=False):
    """
    Density of packing given by minimum distance of EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.

    Returns
    -------
    density : array-like
        Density values, same shape as N.
    """
    min_euclidean_dist = eq_min_dist(dim, N, extra_offset=extra_offset)
    density = calc_packing_density(dim, N, min_euclidean_dist)
    return density


def eq_point_set_property(fhandle, dim, N, extra_offset=False):
    """
    Property of an EQ point set.

    Parameters
    ----------
    fhandle : callable
        Function expecting an array (dim+1 x N), returns property value.
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.

    Returns
    -------
    property : array-like
        Property value(s), same shape as N.
    """
    shape = np.shape(N)
    N_flat = np.reshape(N, (1, int(np.prod(shape))))
    property_vals = np.zeros_like(N_flat, dtype=float)
    for i, n_val in enumerate(N_flat[0]):
        points = eq_point_set(dim, n_val, extra_offset)
        property_vals[0, i] = fhandle(points)
    property_vals = property_vals.reshape(shape)
    return property_vals


def point_set_dist_coeff(points):
    """
    Coefficient of minimum distance of a point set.

    Parameters
    ----------
    points : array-like
        Array of shape (dim+1, N), columns are points in R^{dim+1}.

    Returns
    -------
    coeff : float
        Coefficient value.

    Notes
    -----
    For details, see calc_dist_coeff.

    See Also
    --------
    point_set_min_dist, calc_dist_coeff, eq_dist_coeff, eq_min_dist

    Examples
    --------
    >>> x = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [1, 0, 0, -1]])
    >>> point_set_dist_coeff(x)
    np.float64(2.8284271247461903)
    """
    dim = points.shape[0] - 1
    N = points.shape[1]
    min_euclidean_dist = point_set_min_dist(points)
    coeff = calc_dist_coeff(dim, N, min_euclidean_dist)
    return coeff


def point_set_energy_coeff(points, s=None):
    """
    Coefficient in expansion of energy of a point set.

    Parameters
    ----------
    points : array-like
        Array of shape (dim+1, N), columns are points in R^{dim+1}.
    s : float, optional
        Exponent parameter. Defaults to dim-1.

    Returns
    -------
    coeff : float
        Coefficient value.

    Notes
    -----
    For details, see calc_energy_coeff.

    See Also
    --------
    point_set_energy_dist, calc_energy_coeff, eq_energy_coeff, eq_energy_dist

    Examples
    --------
    >>> x = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [1, 0, 0, -1]])
    >>> point_set_energy_coeff(x)
    array([-0.5214, -0.8232])
    """
    dim = points.shape[0] - 1
    N = points.shape[1]
    if s is None:
        s = dim - 1
    energy = point_set_energy_dist(points, s)
    coeff = calc_energy_coeff(dim, N, s, energy)
    return coeff


def point_set_energy_dist(points, s=None):
    """
    Energy and minimum distance of a point set.

    Parameters
    ----------
    points : array-like
        Array of shape (M, N), columns are points in R^M.
    s : float, optional
        Exponent parameter. Defaults to dim-1.

    Returns
    -------
    energy : float
        Energy value.
    min_dist : float, optional
        Minimum Euclidean distance.

    Notes
    -----
    ENERGY for single point is 0. MIN_DIST for single point is 2.

    See Also
    --------
    eq_energy_dist, point_set_min_dist

    Examples
    --------
    >>> x = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [1, 0, 0, -1]])
    >>> point_set_energy_dist(x, 2)
    (np.float64(2.5), np.float64(1.4142135623730951))
    >>> point_set_energy_dist(x, 0)
    (np.float64(-2.7725887222397816), np.float64(1.4142135623730951))
    """
    M, N = points.shape
    dim = M - 1
    if s is None:
        s = dim - 1
    # Compute pairwise distances, exclude diagonal
    # Optimized using broadcasting

    # Handle N=1 case
    if N <= 1:
        return 0.0, 2.0

    # Expand dims to (M, N, 1) and (M, 1, N) for broadcasting
    # diffs[i, j, k] = points[i, j] - points[i, k]
    # We want dists[j, k] = norm(points[:, j] - points[:, k])

    # points: (M, N)
    # Use standard numpy trick for pairwise distance matrix
    # But for large N this might be memory intensive.
    # However, N is usually small in this context (thousands?)

    # Efficient pairwise distance
    # dist terms: x^2 + y^2 - 2xy. On sphere x^2=1. So 2 - 2xy = 2(1 - x.y).
    # But points might not be exactly on sphere if modified or numerical error.
    # Safe Euclidean:
    dists = np.linalg.norm(points[:, :, None] - points[:, None, :], axis=0)

    # Mask diagonal
    np.fill_diagonal(dists, np.inf)

    min_dist = np.min(dists)

    # Energy: sum r_ij^-s for i != j
    # Flatten and remove Infs
    valid_dists = dists[~np.isinf(dists)]
    # This calculates sum_{i!=j} r_ij^-s.
    # Matlab code calculates sum_{i<j} r_ij^-s.
    # So we divide by 2.
    if s == 0:
        energy = np.sum(-np.log(valid_dists)) / 2.0
    else:
        energy = np.sum(np.power(valid_dists, -s)) / 2.0

    return energy, min_dist


def point_set_min_dist(points):
    """
    Minimum distance between points of a point set.

    Parameters
    ----------
    points : array-like
        Array of shape (dim+1, N).

    Returns
    -------
    min_dist : float
        Minimum Euclidean distance.
    """
    _, min_dist = point_set_energy_dist(points, s=0)  # s doesn't matter for min_dist
    return min_dist
