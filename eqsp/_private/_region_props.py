import numpy as np

from ..partitions import eq_regions
from ..utilities import (
    euclidean_dist,
    polar2cart,
    sph2euc_dist,
)

def expand_region_for_diam(region):
    """
    The set of 2^d vertices of a region.

    Expands a region from the 2 vertex definition to the set of 2^dim vertices
    of the pseudo-region of a region, so that the Euclidean diameter of a region
    is approximated by the diameter of this set.

    Parameters
    ----------
    region : ndarray
        The input region, shape (dim, 2).

    Returns
    -------
    expanded_region : ndarray
        The expanded region, shape (dim, 2**dim).

    See Also
    --------
    pseudo_region_for_diam, max_vertex_diam_of_regions, eq_vertex_diam

    Examples
    --------
    >>> regions = eq_regions(2, 10)
    >>> region = regions[:, :, 2]
    >>> expanded_region = expand_region_for_diam(region)
    >>> np.round(expanded_region, 4)
    array([[0.    , 1.5708, 0.    , 1.5708],
           [0.6435, 0.6435, 1.5708, 1.5708]])
    """
    dim = region.shape[0]
    if dim > 1:
        s_top = region[dim - 1, 0]
        s_bot = region[dim - 1, 1]
        region_1 = expand_region_for_diam(region[: dim - 1, :])
        expanded_region = np.concatenate(
            [append(region_1, s_top), append(region_1, s_bot)], axis=1
        )
    else:
        expanded_region = pseudo_region_for_diam(region)
    return expanded_region


def append(matrix, value):
    """
    Append a coordinate value to each column of a matrix.

    Parameters
    ----------
    matrix : ndarray
        Input matrix, shape (n, m).
    value : float
        Value to append as a new row.

    Returns
    -------
    result : ndarray
        Matrix with appended row, shape (n+1, m).

    Examples
    --------
    >>> m = np.array([[1, 2], [3, 4]])
    >>> append(m, 5)
    array([[1., 2.],
           [3., 4.],
           [5., 5.]])
    """
    return np.vstack([matrix, np.ones(matrix.shape[1]) * value])


def max_diam_bound_of_regions(regions):
    """
    The maximum diameter bound in an array of regions.

    Parameters
    ----------
    regions : ndarray
        Array of regions, shape (dim, 2, N).

    Returns
    -------
    diam_bound : float
        The maximum diameter bound.

    See Also
    --------
    max_vertex_diam_of_regions, eq_diam_bound

    Notes
    -----
    For licensing, see COPYING.
    For references, see AUTHORS.
    For revision history, see CHANGELOG.

    Examples
    --------
    >>> regions = eq_regions(2, 10)
    >>> round(max_diam_bound_of_regions(regions), 4)
    1.6733
    """
    dim = regions.shape[0]
    if dim == 1:
        diam_bound = diam_bound_region(regions[:, :, 0])
    else:
        colatitude = -np.inf * np.ones(dim - 1)
        diam_bound = 0.0
        N = regions.shape[2]
        for region_n in range(N):
            top = regions[:, 0, region_n]
            if np.linalg.norm(top[1:dim] - colatitude) != 0:
                colatitude = top[1:dim]
                diam_bound = max(
                    diam_bound, diam_bound_region(regions[:, :, region_n])
                )
    return diam_bound


def diam_bound_region(region):
    """
    Calculate the per-region bound on the Euclidean diameter of a region.

    Parameters
    ----------
    region : ndarray
        Input region, shape (dim, 2).

    Returns
    -------
    diam_bound : float
        Per-region diameter bound.

    See Also
    --------
    pseudo_region_for_diam

    Examples
    --------
    >>> regions = eq_regions(2, 10)
    >>> round(diam_bound_region(regions[:, :, 0]), 4)
    1.6733
    """
    tol = np.finfo(float).eps * 2 ** 5
    pseudo_region = pseudo_region_for_diam(region)
    dim = pseudo_region.shape[0]
    top = pseudo_region[:, 0]
    bot = pseudo_region[:, 1]
    s = bot[dim - 1] - top[dim - 1]
    e = sph2euc_dist(s)
    if dim == 1:
        diam_bound = e
    else:
        max_sin = max(np.sin(top[dim - 1]), np.sin(bot[dim - 1]))
        if (top[dim - 1] <= np.pi / 2) and (bot[dim - 1] >= np.pi / 2):
            max_sin = 1
        if (abs(top[dim - 1]) < tol) or (abs(np.pi - bot[dim - 1]) < tol):
            diam_bound = 2 * max_sin
        else:
            region_1 = np.column_stack([top[: dim - 1], bot[: dim - 1]])
            diam_bound_1 = max_sin * diam_bound_region(region_1)
            diam_bound = min(2, np.sqrt(e ** 2 + diam_bound_1 ** 2))
    return diam_bound


def max_vertex_diam_of_regions(regions):
    """
    The max vertex diameter in a cell array of regions.

    Parameters
    ----------
    regions : ndarray
        Array of regions, shape (dim, 2, N).

    Returns
    -------
    vertex_diam : float
        The maximum vertex diameter.

    See Also
    --------
    max_diam_bound_of_regions, eq_vertex_diam

    Examples
    --------
    >>> regions = eq_regions(2, 10)
    >>> round(max_vertex_diam_of_regions(regions), 4)
    1.4142
    """
    dim = regions.shape[0]
    if dim == 1:
        vertex_diam = vertex_diam_region(regions[:, :, 0])
    else:
        colatitude = -np.inf * np.ones(dim - 1)
        vertex_diam = 0.0
        N = regions.shape[2]
        for region_n in range(N):
            top = regions[:, 0, region_n]
            if np.linalg.norm(top[1:dim] - colatitude) != 0:
                colatitude = top[1:dim]
                vertex_diam = max(
                    vertex_diam, vertex_diam_region(regions[:, :, region_n])
                )
    return vertex_diam


def vertex_diam_region(region):
    """
    Calculate the Euclidean diameter of the set of 2^dim vertices
    of the pseudo-region of a region.

    Parameters
    ----------
    region : ndarray
        Input region, shape (dim, 2).

    Returns
    -------
    diam : float
        The diameter.

    See Also
    --------
    expand_region_for_diam

    Examples
    --------
    >>> regions = eq_regions(2, 10)
    >>> round(vertex_diam_region(regions[:, :, 0]), 4)
    1.4142
    """
    expanded_region = expand_region_for_diam(region)
    dim = expanded_region.shape[0]
    full = expanded_region.shape[1]
    half = full // 2
    top = expanded_region[:, 0]
    bot = expanded_region[:, full - 1]
    diam = 0.0
    if np.sin(top[dim - 1]) > np.sin(bot[dim - 1]):
        for point_n_1 in range(0, half, 2):
            for point_n_2 in range(point_n_1 + 1, full, 2):
                x1 = polar2cart(expanded_region[:, point_n_1])
                x2 = polar2cart(expanded_region[:, point_n_2])
                diam = max(diam, euclidean_dist(x1, x2))
    else:
        for point_n_1 in range(full - 1, half, -2):
            for point_n_2 in range(point_n_1 - 1, -1, -2):
                x1 = polar2cart(expanded_region[:, point_n_1])
                x2 = polar2cart(expanded_region[:, point_n_2])
                diam = max(diam, euclidean_dist(x1, x2))
    return diam


def pseudo_region_for_diam(region):
    """
    Two points which maximize the vertex diameter of a region.

    Parameters
    ----------
    region : ndarray
        Input region, shape (dim, 2).

    Returns
    -------
    pseudo_region : ndarray
        Pseudo-region, shape (dim, 2).

    See Also
    --------
    expand_region_for_diam, max_vertex_diam_of_regions, eq_vertex_diam

    Examples
    --------
    >>> regions = eq_regions(2, 10)
    >>> region = regions[:, :, 2]
    >>> pseudo_region = pseudo_region_for_diam(region)
    >>> np.round(pseudo_region, 4)
    array([[0.    , 1.5708],
           [0.6435, 1.5708]])
    """
    tol = np.finfo(float).eps * 2 ** 5
    phi_top = region[0, 0]
    phi_bot = region[0, 1]
    if phi_bot == 0:
        phi_bot = 2 * np.pi
    if (
        np.mod(phi_bot - phi_top, 2 * np.pi) < tol
        or np.mod(phi_bot - phi_top, 2 * np.pi) > np.pi
    ):
        phi_bot = phi_top + np.pi
    pseudo_region = region.copy()
    pseudo_region[0, 1] = phi_bot
    return pseudo_region
