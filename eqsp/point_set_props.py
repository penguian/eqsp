"""
PyEQSP point set properties module.

Copyright 2026 Paul Leopardi
"""

import math

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.special import psi

from .partitions import eq_caps, eq_point_set
from .utilities import (
    area_of_cap,
    area_of_sphere,
    euc2sph_dist,
)

__all__ = [
    "calc_dist_coeff",
    "calc_energy_coeff",
    "calc_packing_density",
    "eq_dist_coeff",
    "eq_energy_coeff",
    "eq_energy_dist",
    "eq_min_dist",
    "point_set_dist_coeff",
    "point_set_energy_coeff",
    "point_set_energy_dist",
    "point_set_min_dist",
    "sphere_int_energy",
]


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
    $r^{-s}$ energy point set on $S^{dim}$ was given by
    :ref:`[Rak94] <rak94>` for s == 0, dim = 2,
    :ref:`[Dah78] <dah78>` for s == dim-1,
    :ref:`[Kui04] <kui04>` for dim-1 <= s < dim,
    and :ref:`[Kui98] <kui98>` for s > dim.

    See Also
    --------
    eq_min_dist, eq_dist_coeff

    Examples
    --------
    >>> N = np.arange(2, 7)
    >>> dist = eq_min_dist(2, N)
    >>> np.round(calc_dist_coeff(2, N, dist), 4)
    array([2.8284, 2.4495, 2.8284, 3.1623, 3.4641])
    """
    return min_euclidean_dist * np.power(N, 1 / dim)


def calc_energy_coeff(dim, N, s, energy):
    r"""
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
    The returned coefficient `coeff` (denoted $C$) corresponds to the second term
    in the energy expansion. In the PhD thesis :ref:`[Leo07] <leo07>`,
    the "energy coefficient" $ec_d(\mathcal{N})$ is defined as:
    $ec_d(\mathcal{N}) := (1 - E/I(d,s)) \mathcal{N}^{s/d}$.
    For $s = dim-1$ (where $I(d,s)=1$ on $S^2$ and higher), this relates to $C$ as:
    $ec_d(\mathcal{N}) = -2 \times C$.

    The energy expansion is not valid for N == 1,
    and in particular, eq_energy_coeff(dim, N, 0, energy) := 0.

    For s > 0, :ref:`[Kui98] <kui98>` has
    E(dim, N, s) == (sphere_int_energy(dim, s)/2) N^2 + COEFF N^(1+s/dim) + ...

    For s == 0 (logarithmic potential), see :ref:`[Saf97] <saf97>`.

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
    Ref for s > 0: :ref:`[Kui98] <kui98>`
    Ref for s == 0 and dim == 2: :ref:`[Saf97] <saf97>`

    See Also
    --------
    eq_energy_dist, calc_energy_coeff

    Examples
    --------
    >>> float(sphere_int_energy(2, 0))
    -0.1931471805599453
    """
    if s != 0:
        return (
            math.gamma((dim + 1) / 2)
            * math.gamma(dim - s)
            / (math.gamma((dim - s + 1) / 2) * math.gamma(dim - s / 2))
        )
    if dim != 1:
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
    caps divided by the area of the unit sphere $S^{dim}$.

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


def eq_dist_coeff(dim, N, extra_offset=False, show_progress=False, even_collars=False):
    """
    Coefficient of minimum distance of an EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets (experimental feature from the original MATLAB
        toolbox for dim 2-3).
        Default False.
    show_progress : bool, optional
        Show progress messages. Default False.
    even_collars : bool, optional
        Use even number of collars for symmetric partitions. Default False.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.

    Examples
    --------
    >>> np.round(eq_dist_coeff(2, np.arange(2, 5)), 4)
    array([2.8284, 2.4495, 2.8284])
    >>> float(np.round(eq_dist_coeff(2, 6, show_progress=True), 4))
    3.4641
    """
    dist = eq_min_dist(
        dim, N, extra_offset=extra_offset, show_progress=show_progress,
        even_collars=even_collars
    )
    coeff = dist * np.power(N, 1 / dim)
    return coeff


def eq_energy_coeff(
    dim, N, s=None, extra_offset=False, show_progress=False, even_collars=False
):
    r"""
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
        Use extra offsets (experimental feature from the original MATLAB
        toolbox for dim 2-3).
        Default False.
    show_progress : bool, optional
        Show progress messages. Default False.
    even_collars : bool, optional
        Use even number of collars for symmetric partitions. Default False.

    Returns
    -------
    coeff : array-like
        Coefficient(s), same shape as N.

    Notes
    -----
    The returned `coeff` ($C$) relates to the thesis metric $ec_d(\mathcal{N})$ as:
    $ec_d(\mathcal{N}) = -2 \times C$ (for $s=dim-1$).
    See Remark on page 198 of :ref:`[Leo07] <leo07>`.

    Examples
    --------
    >>> float(np.round(eq_energy_coeff(2, 4), 4))
    -0.5214
    >>> float(np.round(eq_energy_coeff(2, 6, show_progress=True), 4))
    -0.5453
    """
    if s is None:
        s = dim - 1
    dist_result = eq_energy_dist(
        dim,
        N,
        s=s,
        extra_offset=extra_offset,
        show_progress=show_progress,
        even_collars=even_collars,
    )
    if isinstance(dist_result, tuple):
        energy = dist_result[0]
    else:
        energy = dist_result
    coeff = calc_energy_coeff(dim, N, s, energy)
    return coeff


def eq_energy_dist(
    dim, N, s=None, extra_offset=False, show_progress=False, even_collars=False
):
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
        Use extra offsets (experimental feature from the original MATLAB
        toolbox for dim 2-3).
        Default False.
    show_progress : bool, optional
        Show progress messages. Default False.
    even_collars : bool, optional
        Use even number of collars for symmetric partitions. Default False.

    Returns
    -------
    energy : array-like
        Energy values, same shape as N.
    dist : array-like, optional
        Minimum Euclidean distance(s), same shape as N.

    Examples
    --------
    >>> energy = eq_energy_dist(2, 6, show_progress=True)
    >>> float(np.round(energy, 4))
    9.9853
    """
    if s is None:
        s = dim - 1

    shape = np.shape(N)
    n_partitions = int(np.prod(shape))
    N_flat = np.reshape(N, (1, n_partitions))
    energy = np.zeros_like(N_flat, dtype=float)
    dist = np.zeros_like(N_flat, dtype=float)
    for i, n_val in enumerate(N_flat[0]):
        if show_progress and n_partitions > 1:
            print(f"    N={n_val:6} ({i + 1}/{n_partitions})", end="\r", flush=True)
        points = eq_point_set(dim, n_val, extra_offset, even_collars=even_collars)
        if len(dist.shape) > 1:
            energy[0, i], dist[0, i] = point_set_energy_dist(points, s)
        else:
            energy[0, i] = point_set_energy_dist(points, s)  # pragma: no cover
    if show_progress and n_partitions > 1:
        print()  # Clear the line
    energy = energy.reshape(shape)
    dist = dist.reshape(shape)
    if len(dist.shape) > 0:
        return energy, dist
    return energy


def eq_min_dist(dim, N, extra_offset=False, show_progress=False, even_collars=False):
    """
    Minimum distance between centre points of an EQ partition.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets (experimental feature from the original MATLAB
        toolbox for dim 2-3).
        Default False.
    show_progress : bool, optional
        Show progress messages. Default False.
    even_collars : bool, optional
        Use even number of collars for symmetric partitions. Default False.

    Returns
    -------
    dist : array-like
        Minimum Euclidean distance(s), same shape as N.

    Notes
    -----
    Exploits the collar structure for efficient calculation.

    Examples
    --------
    >>> float(np.round(eq_min_dist(2, 6, show_progress=True), 4))
    1.4142
    """
    shape = np.shape(N)
    n_partitions = int(np.prod(shape))
    N_flat = np.reshape(N, (1, n_partitions))
    dist = np.zeros_like(N_flat, dtype=float)
    for i, n_val in enumerate(N_flat[0]):
        if show_progress and n_partitions > 1:
            print(f"    N={n_val:6} ({i + 1}/{n_partitions})", end="\r", flush=True)
        dist[0, i] = _eq_min_dist_scalar(
            dim, int(n_val), extra_offset, even_collars=even_collars
        )
    if show_progress and n_partitions > 1:
        print()  # Clear the line
    return dist.reshape(shape)


def _eq_min_dist_scalar(dim, N, extra_offset=False, even_collars=False):
    """
    Scalar version of eq_min_dist.
    """
    if N <= 1:
        return 2.0
    if dim == 1:
        # Distance on a circle with N points: 2 * sin(pi/N)
        return 2 * np.sin(np.pi / N)

    _, n_regions = eq_caps(dim, N, even_collars=even_collars)

    # Exploiting the collar structure:
    # 1. Intra-collar distances
    # 2. Inter-collar distances (adjacent only)
    #
    # We localized the KDTree search to adjacent collars to achieve
    # near-linear scaling and low memory usage.

    point_sets = []
    points = eq_point_set(dim, N, extra_offset, even_collars=even_collars)
    idx = 0
    for n_k in n_regions:
        nk = int(n_k)
        point_sets.append(points[:, idx : idx + nk])
        idx += nk

    d_min = 2.0
    for k, n_k in enumerate(n_regions):
        # Intra-collar
        if n_k > 1:
            d_intra = point_set_min_dist(point_sets[k])
            d_min = min(d_min, d_intra)
        # Inter-collar with NEXT
        if k < len(n_regions) - 1:
            # Query the KDTree of collar k with the points of collar k+1
            tree = KDTree(point_sets[k].T)
            dists, _ = tree.query(point_sets[k + 1].T, k=1)
            d_inter = np.min(dists)
            d_min = min(d_min, d_inter)

    return d_min


def eq_packing_density(
    dim, N, extra_offset=False, show_progress=False, even_collars=False
):
    """
    Density of packing given by minimum distance of EQ point set.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    N : int or array-like
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets (experimental feature from the original MATLAB
        toolbox for dim 2-3).
        Default False.
    show_progress : bool, optional
        Show progress messages. Default False.
    even_collars : bool, optional
        Use even number of collars for symmetric partitions. Default False.

    Returns
    -------
    density : array-like
        Density values, same shape as N.

    Examples
    --------
    >>> float(np.round(eq_packing_density(2, 4), 4))
    0.5858
    >>> float(np.round(eq_packing_density(2, 6, show_progress=True), 4))
    0.8787
    """
    min_euclidean_dist = eq_min_dist(
        dim,
        N,
        extra_offset=extra_offset,
        show_progress=show_progress,
        even_collars=even_collars,
    )
    density = calc_packing_density(dim, N, min_euclidean_dist)
    return density


def eq_point_set_property(fhandle, dim, N, extra_offset=False, show_progress=False):
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
        Use extra offsets (experimental feature from the original MATLAB
        toolbox for dim 2-3).
        Default False.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    property : array-like
        Property value(s), same shape as N.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.mean(x)
    >>> float(np.round(eq_point_set_property(f, 2, 6, show_progress=True), 10))
    0.0
    """
    shape = np.shape(N)
    n_partitions = int(np.prod(shape))
    N_flat = np.reshape(N, (1, n_partitions))
    property_vals = np.zeros_like(N_flat, dtype=float)
    for i, n_val in enumerate(N_flat[0]):
        if show_progress and n_partitions > 1:
            print(f"    N={n_val:6} ({i + 1}/{n_partitions})", end="\r", flush=True)
        points = eq_point_set(dim, n_val, extra_offset)
        property_vals[0, i] = fhandle(points)
    if show_progress and n_partitions > 1:
        print()  # Clear the line
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
    >>> float(point_set_dist_coeff(x))
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
    array([-0.5214, -0.8232])
    """
    dim = points.shape[0] - 1
    N = points.shape[1]
    if s is None:
        s = dim - 1
    energy = point_set_energy_dist(points, s)
    coeff = calc_energy_coeff(dim, N, s, energy)
    return coeff


def point_set_energy_dist(points, s=None, block_size=2000):
    """
    Energy and minimum distance of a point set.

    Parameters
    ----------
    points : array-like
        Array of shape (M, N), columns are points in R^M.
    s : float, optional
        Exponent parameter. Defaults to dim-1.
    block_size : int, optional
        Maximum number of points to process in a single block.
        Defaults to 2000, balancing memory usage and vectorization overhead.

    Returns
    -------
    energy : float
        Energy value.
    min_dist : float
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
    >>> energy, dist = point_set_energy_dist(x, 2)
    >>> (float(energy), float(dist))
    (2.5, 1.4142135623730951)
    >>> energy, dist = point_set_energy_dist(x, 0)
    >>> (float(energy), float(dist))
    (-2.7725887222397816, 1.4142135623730951)
    """
    M, N = points.shape
    dim = M - 1
    if s is None:
        s = dim - 1

    # Handle N=1 case
    if N <= 1:
        return 0.0, 2.0

    energy = 0.0
    min_dist = np.inf

    # Process in blocks to limit peak memory usage to O(block_size^2)
    # and exploit symmetry (only compute upper triangle of block matrix)
    for i in range(0, N, block_size):
        end_i = min(i + block_size, N)
        pts_i = points[:, i:end_i].T

        for j in range(i, N, block_size):
            end_j = min(j + block_size, N)
            pts_j = points[:, j:end_j].T

            # Efficient pairwise distance between block i and block j
            dists = cdist(pts_i, pts_j, metric="euclidean")

            if i == j:
                # Diagonal block: mask self-distances
                np.fill_diagonal(dists, np.inf)

                # Only take upper triangle to avoid double counting within the block
                valid_mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
                valid_dists = dists[valid_mask]

                if valid_dists.size > 0:
                    min_dist = min(min_dist, np.min(valid_dists))
            else:
                # Off-diagonal block: all distances are valid and distinct
                # We calculate i<j, representing half the symmetric interaction
                valid_dists = dists.flatten()
                min_dist = min(min_dist, np.min(valid_dists))

            # Accumulate energy contribution
            if valid_dists.size > 0:
                if s == 0:
                    energy += np.sum(-np.log(valid_dists))
                else:
                    energy += np.sum(np.power(valid_dists, -s))

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

    Notes
    -----
    Uses scipy.spatial.KDTree for efficient O(N log N) calculation.
    """
    _, N = points.shape
    if N <= 1:
        return 2.0
    tree = KDTree(points.T)
    # query(k=2) returns the distance to itself (0) and the nearest neighbor
    dists, _ = tree.query(points.T, k=2)
    min_dist = np.min(dists[:, 1])
    return min_dist


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
