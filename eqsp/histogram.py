import numpy as np

from histogram_private import lookup_s2_region
from partitions import eq_regions, eq_caps


def eq_count_points_by_s2_region(s_point, N):
    """
    Count points in each of N regions of S^2.

    Parameters
    ----------
    s_point : ndarray
        Sequence of points on S^2, as a 2 x n_points array in spherical
        polar coordinates, with longitude 0 <= s[0, p_idx] <= 2*pi,
        colatitude 0 <= s[1, p_idx] <= pi.
    N : int
        Required number of regions, a positive integer.

    Returns
    -------
    count_v : ndarray
        Array of length N containing the number of points of s_point
        contained in each region.

    See Also
    --------
    eq_find_s2_region

    Examples
    --------
    >>> import numpy as np
    >>> from partitions import eq_point_set_polar
    >>> points_s = eq_point_set_polar(2, 8)
    >>> eq_count_points_by_s2_region(points_s, 8)
    array([1, 1, 1, 1, 1, 1, 1, 1])
    >>> eq_count_points_by_s2_region(points_s, 5)
    array([1, 2, 2, 2, 1])
    >>> points_s = eq_point_set_polar(2, 128)
    >>> eq_count_points_by_s2_region(points_s, 8)
    array([19, 15, 14, 17, 15, 14, 15, 19])
    >>> eq_count_points_by_s2_region(points_s, 5)
    array([19, 29, 32, 29, 19])
    """
    r_idx = eq_find_s2_region(s_point, N)
    count_v = np.histogram(r_idx, bins=np.arange(1, N + 2))[0]
    return count_v

def eq_find_s2_region(s_point, N):
    """
    Partition S^2 into N regions and find the index for each point.

    Parameters
    ----------
    s_point : ndarray
        Sequence of points on S^2, as a 2 x n_points array in spherical
        polar coordinates, with longitude 0 <= s[0, p_idx] <= 2*pi,
        colatitude 0 <= s[1, p_idx] <= pi.
    N : int
        Required number of regions, a positive integer.

    Returns
    -------
    r_idx : ndarray
        Array of length s_point.shape[1] containing the index of the region
        corresponding to each point.

    See Also
    --------
    eq_count_points_by_s2_region
    lookup_s2_region

    Examples
    --------
    >>> from partitions import eq_point_set_polar
    >>> points_s = eq_point_set_polar(2, 8)
    >>> eq_find_s2_region(points_s, 8)
    array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> eq_find_s2_region(points_s, 5)
    array([1, 2, 2, 3, 3, 4, 4, 5])
    """
    s_regions = eq_regions(2, N)
    s_cap, n_regions = eq_caps(2, N)
    c_regions = np.cumsum(n_regions)
    r_idx = lookup_s2_region(s_point, s_regions, s_cap, c_regions)
    return r_idx

def in_s2_region(s_point, region):
    """
    Test if points on S^2 are within a given region.

    Parameters
    ----------
    s_point : ndarray
        Sequence of points on S^2, as a 2 x n_points array in spherical
        polar coordinates, with longitude 0 <= s[0, p_idx] <= 2*pi,
        colatitude 0 <= s[1, p_idx] <= pi.
    region : ndarray
        One region of S^2 as returned by eq_regions(2, N).

    Returns
    -------
    result : ndarray
        Boolean array of length s_point.shape[1] indicating whether each
        point is in the region.

    See Also
    --------
    eq_regions
    eq_find_s2_region

    Examples
    --------
    >>> from partitions import eq_point_set_polar, eq_regions
    >>> points_s = eq_point_set_polar(2, 8)
    >>> s_regions = eq_regions(2, 5)
    >>> region = s_regions[:, :, 2]
    >>> in_s2_region(points_s, region)
    array([False, False, False,  True,  True, False, False, False])
    """
    n_points = s_point.shape[1]
    result = np.zeros(n_points, dtype=bool)
    for p_idx in range(n_points):
        longitude = s_point[0, p_idx]
        min_long = region[0, 0]
        max_long = region[0, 1]
        in_long = (min_long < longitude <= max_long)
        if not in_long:
            longitude = longitude + 2 * np.pi
            in_long = (min_long < longitude <= max_long)
        colatitude = s_point[1, p_idx]
        min_colat = region[1, 0]
        max_colat = region[1, 1]
        in_colat = True
        if (min_colat == 0.0 and colatitude < min_colat) or \
           (min_colat > 0.0 and colatitude <= min_colat) or \
           (max_colat < colatitude):
            in_colat = False
        result[p_idx] = in_long and in_colat
    return result
