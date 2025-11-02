import numpy as np
from eq_regions import eq_regions
from area_of_sphere import area_of_sphere
from area_of_ideal_region import area_of_ideal_region
from area_of_collar import area_of_collar
from eq_regions_property import eq_regions_property
from max_diam_bound_of_regions import max_diam_bound_of_regions
from max_vertex_diam_of_regions import max_vertex_diam_of_regions

def eq_area_error(dim, N):
    """
    Total area error and max area error per region of an EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int or array-like of int
        Number of regions to partition.

    Returns
    -------
    total_error : ndarray
        Absolute difference between total area of all regions and the area
        of S^dim.
    max_error : ndarray
        Maximum absolute difference between the area of any region and the
        ideal area.

    Raises
    ------
    ValueError
        If arguments are not provided.

    See Also
    --------
    eq_regions, area_of_sphere, area_of_ideal_region

    Notes
    -----
    The results will be arrays of the same size as N.

    Examples
    --------
    >>> total_error, max_error = eq_area_error(2, 10)
    >>> np.round(total_error, 4)
    0.0
    >>> np.round(max_error, 4)
    0.0
    >>> total_error, max_error = eq_area_error(3, np.arange(1, 7))
    >>> np.round(total_error * 1e12, 4)
    array([0.0036, 0.0036, 0.1847, 0.0142, 0.0142, 0.2132])
    >>> np.round(max_error * 1e12, 4)
    array([0.0036, 0.0018, 0.1954, 0.0284, 0.044 , 0.0777])
    """
    if dim is None or N is None:
        raise ValueError("Both dim and N must be provided.")

    shape = np.shape(N)
    n_partitions = np.prod(shape)
    N = np.reshape(N, (1, n_partitions))

    total_error = np.zeros_like(N, dtype=float)
    max_error = np.zeros_like(N, dtype=float)
    sphere_area = area_of_sphere(dim)

    for partition_n in range(n_partitions):
        n = N[0, partition_n]
        regions = eq_regions(dim, n)
        ideal_area = area_of_ideal_region(dim, n)
        total_area = 0.0
        for region_n in range(regions.shape[2]):
            area = area_of_region(regions[:, :, region_n])
            total_area += area
            region_error = abs(area - ideal_area)
            if region_error > max_error[0, partition_n]:
                max_error[0, partition_n] = region_error
        total_error[0, partition_n] = abs(sphere_area - total_area)

    total_error = np.reshape(total_error, shape)
    max_error = np.reshape(max_error, shape)
    return total_error, max_error

def area_of_region(region):
    """
    Area of given region.

    Parameters
    ----------
    region : ndarray
        Region, typically of shape (dim, 2).

    Returns
    -------
    area : float
        Area of the region.

    See Also
    --------
    area_of_sphere, area_of_collar

    Examples
    --------
    >>> import numpy as np
    >>> region = np.array([[0, np.pi], [0, 2*np.pi]])
    >>> round(area_of_region(region), 4)
    6.2832
    """
    dim = region.shape[0]
    s_top = region[dim - 1, 0]
    s_bot = region[dim - 1, 1]
    if dim > 1:
        area = (
            area_of_collar(dim, s_top, s_bot)
            * area_of_region(region[: dim - 1, :])
            / area_of_sphere(dim - 1)
        )
    else:
        if s_bot == 0:
            s_bot = 2 * np.pi
        if s_top == s_bot:
            s_bot = s_top + 2 * np.pi
        area = s_bot - s_top
    return area

def eq_diam_bound(dim, N):
    """
    Maximum per-region diameter bound of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of partitions.

    Returns
    -------
    diam_bound : ndarray
        Maximum of per-region diameter bound.

    See Also
    --------
    eq_vertex_diam, eq_diam_coeff, eq_regions_property

    Examples
    --------
    >>> eq_diam_bound(2, 10)
    1.6733200530681511
    >>> eq_diam_bound(3, np.arange(1, 7))
    array([2, 2, 2, 2, 2, 2])
    """
    return eq_regions_property(max_diam_bound_of_regions, dim, N)

def eq_diam_coeff(dim, N):
    """
    Coefficients of diameter bound and vertex diameter of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of regions.

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
    >>> eq_diam_coeff(2, 10)
    5.291502530159141
    >>> eq_diam_coeff(3, np.arange(1, 7))
    (array([2.    , 2.5198, 2.8845, 3.1748, 3.42  , 3.6342]),
     array([2.    , 2.5198, 2.8845, 3.1748, 3.42  , 3.6342]))
    """
    if dim is None or N is None:
        raise ValueError("Both dim and N must be provided.")

    import inspect

    caller_outputs = len(inspect.stack()[1][0].f_locals.get("output", []))
    if caller_outputs < 2:
        bound_coeff = eq_diam_bound(dim, N) * np.power(N, 1 / dim)
        return bound_coeff
    else:
        shape = np.shape(N)
        n_partitions = np.prod(shape)
        N = np.reshape(N, (1, n_partitions))
        bound_coeff = np.zeros_like(N, dtype=float)
        vertex_coeff = np.zeros_like(N, dtype=float)
        for partition_n in range(n_partitions):
            n = N[0, partition_n]
            regions = eq_regions(dim, n)
            scale = np.power(n, 1 / dim)
            bound_coeff[0, partition_n] = (
                max_diam_bound_of_regions(regions) * scale
            )
            vertex_coeff[0, partition_n] = (
                max_vertex_diam_of_regions(regions) * scale
            )
        bound_coeff = np.reshape(bound_coeff, shape)
        vertex_coeff = np.reshape(vertex_coeff, shape)
        return bound_coeff, vertex_coeff

def eq_regions_property(fhandle, dim, N):
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
    >>> eq_regions_property(dummy_property, 2, [3, 4])
    array([3, 4])
    """
    shape = np.shape(N)
    n_partitions = np.prod(shape)
    N = np.reshape(N, (1, n_partitions))
    property_ = np.zeros_like(N, dtype=float)
    for partition_n in range(n_partitions):
        regions = eq_regions(dim, N[0, partition_n])
        property_[0, partition_n] = fhandle(regions)
    property_ = np.reshape(property_, shape)
    return property_

def eq_vertex_diam_coeff(dim, N):
    """
    Coefficient of maximum vertex diameter of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of partitions.

    Returns
    -------
    coeff : ndarray
        Vertex diameter coefficient.

    See Also
    --------
    eq_vertex_diam, eq_diam_coeff

    Examples
    --------
    >>> eq_vertex_diam_coeff(2, 10)
    4.47213595499958
    >>> eq_vertex_diam_coeff(3, np.arange(1, 7))
    array([2.    , 2.5198, 2.8845, 3.1748, 3.42  , 3.6342])
    """
    return eq_vertex_diam(dim, N) * np.power(N, 1 / dim)

def eq_vertex_diam(dim, N):
    """
    Maximum vertex diameter of EQ partition.

    Parameters
    ----------
    dim : int
        Dimension of sphere.
    N : int or array-like of int
        Number of regions.

    Returns
    -------
    vertex_diam : ndarray
        Maximum vertex diameter over all regions.

    See Also
    --------
    eq_diam_bound, eq_vertex_diam_coeff, eq_diam_coeff, eq_regions_property

    Examples
    --------
    >>> eq_vertex_diam(2, 10)
    1.4142135623730951
    >>> eq_vertex_diam(3, np.arange(1, 7))
    array([2, 2, 2, 2, 2, 2])
    """
    return eq_regions_property(max_vertex_diam_of_regions, dim, N)
