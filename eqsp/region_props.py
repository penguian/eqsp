"""
Recursive Zonal Equal Area Sphere Partitioning

Copyright 2026 Paul Leopardi
"""

import numpy as np

from ._private._region_props import (
    max_diam_bound_of_regions,
    max_vertex_diam_of_regions,
)
from .partitions import eq_regions
from .utilities import (
    area_of_collar,
    area_of_ideal_region,
    area_of_sphere,
)


def eq_area_error(dim, N, show_progress=False):
    """
    Total area error and max area error per region of an EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int or array-like of int
        Number of regions to partition.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    total_error : ndarray
        Absolute difference between total area of all regions and the area
        of $S^{dim}$. Total error should be near 0.
    max_error : ndarray
        Maximum absolute difference between the area of any region and the
        ideal area.

    Raises
    ------
    ValueError
        If arguments are not provided.

    See Also
    -------
    eq_regions, area_of_sphere, area_of_ideal_region

    Notes
    -----
    The results will be arrays of the same size as N.
    Note that both total_error and max_error are returned as native floats
    if N is a scalar, otherwise as NumPy arrays.

    Implementation
    --------------
    To accurately measure the accumulated rounding error of the partitioning
    algorithm itself, this function computes the area for every single region
    independently based on the exact geometric boundaries (`s_top` and `s_bot`)
    produced by the algorithm. It does not assume regions within a given collar
    are strictly identical, nor does it substitute theoretical ideal areas.
    The calculations are vectorized using NumPy arrays for performance while
    maintaining strict geometric fidelity to the algorithm's actual output.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> total_error, max_error = eq_area_error(2, 10)
    >>> np.allclose(total_error, 0)
    True
    >>> total_error, max_error = eq_area_error(2, 6, show_progress=True)
    >>> float(total_error)
    0.0
    >>> np.allclose(max_error, 0, atol=1e-12)
    True
    >>> np.allclose(total_error * 1e12, [0, 0, 0, 0, 0, 0], atol=1)
    True
    >>> np.allclose(max_error * 1e12, [0, 0, 0, 0, 0, 0], atol=1)
    True
    >>> total_error, max_error = eq_area_error(2, 6, show_progress=True)
    >>> np.allclose(total_error, 0)
    True
    """
    if dim is None or N is None:
        raise ValueError("Both dim and N must be provided.")

    shape = np.shape(N)
    n_partitions = int(np.prod(shape))
    N_flat = np.reshape(N, (1, n_partitions))

    total_error = np.zeros_like(N_flat, dtype=float)
    max_error = np.zeros_like(N_flat, dtype=float)
    sphere_area = area_of_sphere(dim)

    for i in range(n_partitions):
        n = int(N_flat[0, i])
        if show_progress and n_partitions > 1:
            print(f"    N={n:6} ({i + 1}/{n_partitions})", end="\r", flush=True)
        regions = eq_regions(dim, n)
        ideal_area = area_of_ideal_region(dim, n)

        areas = area_of_region(regions)
        total_area = np.sum(areas)

        # max_error logic
        region_errors = np.abs(areas - ideal_area)
        if np.size(region_errors) > 0:
            max_error[0, i] = np.max(region_errors)

        total_error[0, i] = abs(sphere_area - total_area)

    if show_progress and n_partitions > 1:
        print()  # Clear the line
    total_error = np.reshape(total_error, shape)
    max_error = np.reshape(max_error, shape)
    return total_error, max_error


def area_of_region(region):
    """
    Area of given region(s).

    Parameters
    ----------
    region : ndarray
        Region(s), typically of shape (dim, 2) or (dim, 2, N).

    Returns
    -------
    area : float or ndarray
        Area of the region(s).

    See Also
    --------
    area_of_sphere, area_of_collar

    Examples
    --------
    >>> import numpy as np
    >>> region = np.array([[0, 2*np.pi], [0, np.pi]])
    >>> float(area_of_region(region))
    12.566370614359172
    """
    dim = region.shape[0]
    s_top = region[dim - 1, 0, ...]
    s_bot = region[dim - 1, 1, ...]
    if dim > 1:
        area = (
            area_of_collar(dim, s_top, s_bot)
            * area_of_region(region[: dim - 1, ...])
            / area_of_sphere(dim - 1)
        )
    else:
        if np.ndim(s_bot) == 0:
            if s_bot == 0:
                s_bot = 2 * np.pi
            if s_top == s_bot:
                s_bot = s_top + 2 * np.pi
            area = s_bot - s_top
        else:
            s_bot = np.copy(s_bot)
            s_bot[s_bot == 0] = 2 * np.pi
            mask = s_top == s_bot
            s_bot[mask] = s_top[mask] + 2 * np.pi
            area = s_bot - s_top
    return area


def eq_diam_bound(dim, N, show_progress=False):
    """
    Maximum per-region diameter bound of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of partitions.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    diam_bound : ndarray
        Maximum of per-region diameter bound.

    See Also
    --------
    eq_vertex_diam, eq_diam_coeff, eq_regions_property

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> print(f"{float(eq_diam_bound(2, 10)):.4f}")
    1.6733
    >>> eq_diam_bound(3, np.arange(1, 7))
    array([2., 2., 2., 2., 2., 2.])
    >>> print(f"{float(eq_diam_bound(2, 6, show_progress=True)):.4f}")
    1.9437
    """
    return eq_regions_property(
        max_diam_bound_of_regions, dim, N, show_progress=show_progress
    )


def eq_diam_coeff(dim, N, show_progress=False):
    """
    Coefficients of diameter bound and vertex diameter of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of regions.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    bound_coeff : ndarray
        Diameter bound coefficient.
    vertex_coeff : ndarray, optional
        Vertex diameter coefficient.

    Raises
    ------
    ValueError
        If arguments are not provided.

    See Also
    --------
    eq_diam_bound, eq_vertex_diam, eq_regions, eq_vertex_diam_coeff

    Notes
    -----
    If called with one output, returns only bound_coeff.

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> bound_coeff, vertex_coeff = eq_diam_coeff(2, 10)
    >>> print(f"{float(bound_coeff):.4f}, {float(vertex_coeff):.4f}")
    5.2915, 4.4721
    >>> b, v = eq_diam_coeff(2, 6, show_progress=True)
    >>> print(f"{float(b):.4f}")
    4.7610
    """
    if dim is None or N is None:
        raise ValueError("Both dim and N must be provided.")

    shape = np.shape(N)
    n_partitions = int(np.prod(shape))
    N_flat = np.reshape(N, (1, n_partitions))
    bound_coeff = np.zeros_like(N_flat, dtype=float)
    vertex_coeff = np.zeros_like(N_flat, dtype=float)
    for i in range(n_partitions):
        n = int(N_flat[0, i])
        if show_progress and n_partitions > 1:
            print(f"    N={n:6} ({i + 1}/{n_partitions})", end="\r", flush=True)
        regions = eq_regions(dim, n)
        scale = np.power(n, 1 / dim)
        bound_coeff[0, i] = max_diam_bound_of_regions(regions) * scale
        vertex_coeff[0, i] = max_vertex_diam_of_regions(regions) * scale
    if show_progress and n_partitions > 1:
        print()  # Clear the line
    bound_coeff = np.reshape(bound_coeff, shape)
    vertex_coeff = np.reshape(vertex_coeff, shape)
    return bound_coeff, vertex_coeff


def eq_regions_property(fhandle, dim, N, show_progress=False):
    """
    Property of regions of an EQ partition.

    Parameters
    ----------
    fhandle : callable
        Function handle to apply to regions.
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of regions.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    property : ndarray
        Calculated property for each partition.

    See Also
    --------
    eq_regions, eq_diam_bound, eq_vertex_diam

    Notes
    -----
    The function specified by fhandle must accept an array of shape
    (dim, 2, N_regions) and return a single value.

    Examples
    --------
    >>> def dummy_property(regions): return regions.shape[2]
    >>> eq_regions_property(dummy_property, 2, [3, 4]).astype(int)
    array([3, 4])
    >>> float(eq_regions_property(dummy_property, 2, 6, show_progress=True))
    6.0
    """
    shape = np.shape(N)
    n_partitions = int(np.prod(shape))
    N_flat = np.reshape(N, (1, n_partitions))
    property_ = np.zeros_like(N_flat, dtype=float)
    for i in range(n_partitions):
        n = int(N_flat[0, i])
        if show_progress and n_partitions > 1:
            print(f"    N={n:6} ({i + 1}/{n_partitions})", end="\r", flush=True)
        regions = eq_regions(dim, n)
        property_[0, i] = fhandle(regions)
    if show_progress and n_partitions > 1:
        print()  # Clear the line
    property_ = np.reshape(property_, shape)
    return property_


def eq_vertex_diam_coeff(dim, N, show_progress=False):
    """
    Coefficient of maximum vertex diameter of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of partitions.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    coeff : ndarray
        Vertex diameter coefficient.

    See Also
    --------
    eq_vertex_diam, eq_diam_coeff

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> print(f"{float(eq_vertex_diam_coeff(2, 10)):.4f}")
    4.4721
    >>> eq_vertex_diam_coeff(3, np.arange(1, 7))
    array([2.    , 2.5198, 2.8845, 3.1748, 3.42  , 3.6342])
    >>> print(f"{float(eq_vertex_diam_coeff(2, 6, show_progress=True)):.4f}")
    4.1633
    """
    return eq_vertex_diam(dim, N, show_progress=show_progress) * np.power(N, 1 / dim)


def eq_vertex_diam(dim, N, show_progress=False):
    """
    Maximum vertex diameter of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of regions.
    show_progress : bool, optional
        Show progress messages. Default False.

    Returns
    -------
    vertex_diam : ndarray
        Maximum vertex diameter over all regions.

    See Also
    --------
    eq_diam_bound, eq_vertex_diam_coeff, eq_diam_coeff, eq_regions_property

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> print(f"{float(eq_vertex_diam(2, 10)):.4f}")
    1.4142
    >>> eq_vertex_diam(3, np.arange(1, 7))
    array([2., 2., 2., 2., 2., 2.])
    >>> print(f"{float(eq_vertex_diam(2, 6, show_progress=True)):.4f}")
    1.6997
    """
    return eq_regions_property(
        max_vertex_diam_of_regions, dim, N, show_progress=show_progress
    )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
