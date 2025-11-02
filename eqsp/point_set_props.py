import numpy as np
import math
from eq_min_dist import eq_min_dist
from eq_energy_dist import eq_energy_dist
from eq_point_set import eq_point_set
from partition_options import partition_options
from point_set_min_dist import point_set_min_dist
from area_of_cap import area_of_cap
from area_of_sphere import area_of_sphere
from euc2sph_dist import euc2sph_dist

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
    >>> N = np.arange(2, 7)
    >>> energy = eq_energy_dist(2, N, 0)
    >>> calc_energy_coeff(2, N, 0, energy)
    array([-0.22130839, -0.15693414, -0.22130839, -0.24933177, -0.25694627])
    """
    def sphere_int_energy(dim, s):
        if s != 0:
            return (math.gamma((dim + 1) / 2) *
                    math.gamma(dim - s) /
                    (math.gamma((dim - s + 1) / 2) *
                     math.gamma(dim - s / 2)))
        elif dim != 1:
            from scipy.special import psi
            return (psi(dim) - psi(dim / 2) - math.log(4)) / 2
        else:
            return 0

    N = np.asarray(N)
    energy = np.asarray(energy)
    if s > 0:
        first_term = (sphere_int_energy(dim, s) / 2) * np.power(N, 2)
        coeff = (energy - first_term) / np.power(N, 1 + s / dim)
    else:
        shape = N.shape
        n_partitions = np.prod(shape)
        N_flat = N.reshape(1, n_partitions)
        first_term = (sphere_int_energy(dim, s) / 2) * np.power(N_flat, 2)
        coeff = np.zeros_like(N_flat, dtype=float)
        neq1 = (N_flat != 1)
        coeff[neq1] = ((energy.reshape(1, n_partitions)[0][neq1] -
                        first_term[0][neq1]) /
                       (N_flat[0][neq1] * np.log(N_flat[0][neq1])))
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
    -0.22130839205385095
    """
    if s != 0:
        return (math.gamma((dim + 1) / 2) *
                math.gamma(dim - s) /
                (math.gamma((dim - s + 1) / 2) *
                 math.gamma(dim - s / 2)))
    elif dim != 1:
        from scipy.special import psi
        return (psi(dim) - psi(dim / 2) - math.log(4)) / 2
    else:
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
    >>> N = np.arange(2, 7)
    >>> dist = eq_min_dist(2, N)
    >>> calc_packing_density(2, N, dist)
    array([1.        , 0.43931456, 0.58578644, 0.73222988, 0.87867331])
    """
    s_cap = euc2sph_dist(min_euclidean_dist) / 2
    s_cap = np.array(s_cap)
    N = np.asarray(N)
    s_cap[N == 1] = np.pi
    density = N * area_of_cap(dim, s_cap) / area_of_sphere(dim)
    return density

def eq_dist_coeff(dim, N, *args):
    """
    Coefficient of minimum distance of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    *args : optional
        Options, e.g. 'offset', 'extra'.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.

    Notes
    -----
    The expression for the lower bound on minimum distance of a minimum r^(-s)
    energy point set on S^dim was given in various references.

    See Also
    --------
    partition_options, eq_min_dist

    Examples
    --------
    >>> eq_dist_coeff(2, 10)
    3.325048771007371
    >>> eq_dist_coeff(3, np.arange(1, 7))
    array([2.        , 2.51981592, 2.03960781, 2.24492495, 2.41832821,
           2.56978336])
    """
    dist = eq_min_dist(dim, N, *args)
    coeff = dist * np.power(N, 1 / dim)
    return coeff

def eq_energy_coeff(dim, N, *args):
    """
    Coefficient in expansion of energy of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    *args : optional
        s parameter and options.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.

    Notes
    -----
    The default value of s is dim-1.
    The energy expansion is not valid for N == 1.

    See Also
    --------
    partition_options, eq_energy_dist, calc_energy_coeff

    Examples
    --------
    >>> eq_energy_coeff(2, 10)
    -0.5461028849820652
    >>> eq_energy_coeff(3, np.arange(1, 7))
    array([-0.5       , -0.5512227 , -0.5208487 , -0.54565828, -0.5471944 ,
           -0.56791135])
    >>> eq_energy_coeff(2, np.arange(1, 7), 0)
    array([ 0.        , -0.22130839, -0.15693414, -0.22130839, -0.24933177,
           -0.25694627])
    """
    if len(args) == 0 or isinstance(args[0], str):
        s = dim - 1
        options = args
    else:
        s = args[0]
        options = args[1:]
    energy = eq_energy_dist(dim, N, s, *options)
    coeff = calc_energy_coeff(dim, N, s, energy)
    return coeff

def eq_energy_dist(dim, N, *args):
    """
    Energy and minimum distance of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    *args : optional
        s parameter, options, e.g. 'offset', 'extra'.

    Returns
    -------
    energy : array-like
        Energy values, same shape as N.
    dist : array-like, optional
        Minimum Euclidean distance(s), same shape as N.

    Notes
    -----
    The default value of s is dim-1.

    See Also
    --------
    eq_point_set, partition_options, point_set_energy_dist, eq_min_dist

    Examples
    --------
    >>> eq_energy_dist(2, 10)
    32.731191479
    >>> eq_energy_dist(3, np.arange(1, 7), 0)
    (array([ 0.        , -0.69314718, -1.38629436, -2.77258872, -4.15888308,
           -6.23832463]), array([2.        , 2.        , 1.41421356, 1.41421356,
           1.41421356, 1.41421356]))
    """
    if len(args) == 0 or isinstance(args[0], str):
        s = dim - 1
        options = args
    else:
        s = args[0]
        options = args[1:]
    pdefault = {'extra_offset': False}
    if len(options) == 0:
        extra_offset = pdefault['extra_offset']
    else:
        popt = partition_options(pdefault, *options)
        extra_offset = popt['extra_offset']

    shape = np.shape(N)
    N_flat = np.reshape(N, (1, np.prod(shape)))
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
    else:
        return energy

def eq_min_dist(dim, N, *args):
    """
    Minimum distance between center points of an EQ partition.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    *args : optional
        Options, e.g. 'offset', 'extra'.

    Returns
    -------
    dist : array-like
        Minimum Euclidean distance(s), same shape as N.

    Notes
    -----
    For dim == 2 or dim == 3, uses experimental extra rotation offsets
    with 'offset', 'extra' arguments.

    See Also
    --------
    partition_options, eq_energy_dist

    Examples
    --------
    >>> eq_min_dist(2, 10)
    1.0514622242382672
    >>> eq_min_dist(3, np.arange(1, 7))
    array([2.        , 2.        , 1.41421356, 1.41421356, 1.41421356,
           1.41421356])
    """
    return eq_point_set_property(point_set_min_dist, dim, N, *args)

def eq_packing_density(dim, N, *args):
    """
    Density of packing given by minimum distance of EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    *args : optional
        Options, e.g. 'offset', 'extra'.

    Returns
    -------
    density : array-like
        Density values, same shape as N.

    Notes
    -----
    The packing density is the sum of the areas of spherical caps
    divided by the area of the unit sphere S^dim.

    See Also
    --------
    eq_min_dist, area_of_cap, area_of_sphere, partition_options

    Examples
    --------
    >>> eq_packing_density(2, 10)
    0.7466774364375444
    >>> eq_packing_density(3, np.arange(1, 7))
    array([1.        , 1.        , 0.27252796, 0.36337062, 0.45421328,
           0.54505594])
    """
    min_euclidean_dist = eq_min_dist(dim, N, *args)
    density = calc_packing_density(dim, N, min_euclidean_dist)
    return density

def eq_point_set_property(fhandle, dim, N, *args):
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
    *args : optional
        Options, e.g. 'offset', 'extra'.

    Returns
    -------
    property : array-like
        Property value(s), same shape as N.

    Notes
    -----
    For dim == 2 or 3, uses experimental extra rotation offsets
    with 'offset', 'extra' arguments.

    See Also
    --------
    eq_point_set, partition_options

    Examples
    --------
    >>> eq_point_set_property(point_set_min_dist, 2, 10)
    1.0514622242382672
    """
    pdefault = {'extra_offset': False}
    popt = partition_options(pdefault, *args)
    extra_offset = popt['extra_offset']
    shape = np.shape(N)
    N_flat = np.reshape(N, (1, np.prod(shape)))
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
    2.8284271247461903
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
    -0.5214054331644342
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
    >>> point_set_energy_dist(x)
    2.5
    >>> point_set_energy_dist(x, 0)
    -0.6931471805599453
    """
    M, N = points.shape
    dim = M - 1
    if s is None:
        s = dim
    # Compute pairwise distances, exclude diagonal
    dists = np.linalg.norm(points[:, None, :] - points[:, :, None], axis=0)
    mask = ~np.eye(N, dtype=bool)
    energy = np.sum(np.power(dists[mask], -s)) if N > 1 else 0
    min_dist = np.min(dists[mask]) if N > 1 else 2
    if N > 1:
        return energy / (N * (N - 1)), min_dist
    else:
        return energy, min_dist
