import math
import numpy as np

from math import pi

from ._private._partitions import (
    bot_cap_region,
    cap_colats,
    centres_of_regions,
    circle_offset,
    ideal_region_list,
    num_collars,
    polar_colat,
    round_to_naturals,
    s2_offset,
    sphere_region,
    top_cap_region,
)
from .utilities import (
    asfloat,
    cart2polar2,
    ideal_collar_angle,
    polar2cart,
)


TAU = 2.0 * pi


def eq_caps(dim, N):
    """
    Partition a sphere into to nested spherical caps.

    Parameters
    ----------
    dim : int
        The number of dimensions.
    N : int
        The number of regions.

    Returns
    -------
    s_cap : ndarray
        1D array containing increasing colatitudes of caps. Size is
        (n_collars+2,).
    n_regions : ndarray
        1D array containing the integer number of regions in each zone.
        Size is (n_collars+2,).

    Raises
    ------
    ValueError
        If dim or N are not positive integers.

    See Also
    --------
    eq_regions, eq_point_set_polar

    Notes
    -----
    - s_cap[0] is the colatitude of the North polar cap.
    - s_cap[-2] is pi - c_polar.
    - s_cap[-1] is pi.
    - n_regions[0] is 1 and n_regions[-1] is 1.
    - The sum of n_regions equals N.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> s_cap, n_regions = eq_caps(2,10)
    >>> s_cap
    array([0.6435, 1.5708, 2.4981, 3.1416])
    >>> n_regions
    array([1., 4., 4., 1.])

    >>> s_cap, n_regions = eq_caps(3,6)
    >>> s_cap
    array([0.9845, 2.1571, 3.1416])
    >>> n_regions
    array([1., 4., 1.])
    """
    if not (isinstance(dim, (int, np.integer)) and dim >= 1):
        raise ValueError("dim must be a positive integer")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("N must be a positive integer")

    if dim == 1:
        # Circle: return the angles of N equal sectors.
        sector = np.arange(1, N + 1)
        s_cap = sector * TAU / N
        n_regions = np.ones_like(sector, dtype=int)
    elif N == 1:
        # Only one region: whole sphere.
        s_cap = np.array([pi])
        n_regions = np.array([1], dtype=int)
    else:
        # Determine polar colatitude
        c_polar = polar_colat(dim, N)
        # Determine number of collars
        n_collars = num_collars(N, c_polar, ideal_collar_angle(dim, N))
        # Ideal real number of regions per collar
        r_regions = ideal_region_list(dim, N, c_polar, n_collars)
        # Round to natural numbers preserving sum N
        n_regions = round_to_naturals(N, r_regions)
        # Colatitudes of cap tops
        s_cap = cap_colats(dim, N, c_polar, n_regions)

    return asfloat(s_cap), asfloat(n_regions)


def eq_point_set(dim, N, extra_offset=False):
    """
    Center points of regions of EQ partition, in Cartesian coordinates.

    Parameters
    ----------
    dim : int
        The number of dimensions.
    N : int
        The number of regions.
    extra_offset : bool
        If True, enables experimental extra offsets for dim 2 and 3.

    Returns
    -------
    points_x : ndarray
        Array of shape (dim+1, N) containing center points of each region in
        Cartesian coordinates. Each column is a point on S^dim.

    Raises
    ------
    ValueError
        If dim or N are not positive integers.

    See Also
    --------
    eq_point_set_polar, polar2cart, eq_regions

    Notes
    -----
    Uses the recursive zonal equal area algorithm. The optional argument
    "offset", "extra" may be provided and is forwarded to
    eq_point_set_polar.

    Examples
    --------
    >>> points = eq_point_set(2, 4)  # doctest: +ELLIPSIS
    >>> points.shape
    (3, 4)
    """
    points_polar = eq_point_set_polar(dim, N, extra_offset=extra_offset)
    points_x = polar2cart(points_polar)
    return points_x


def eq_point_set_polar(dim, N, extra_offset=False):
    """
    Center points of regions of an EQ partition in polar coordinates.

    Parameters
    ----------
    dim : int
        The spatial dimension of the sphere (S^dim in R^{dim+1}).
    N : int
        The number of regions.
    extra_offset : bool
        If True, enables experimental extra offsets for dim 2 and 3.

    Returns
    -------
    points_s : ndarray
        Array of shape (dim, N) containing center points in spherical polar
        coordinates. Each column is a point on S^dim.

    Raises
    ------
    ValueError
        If dim or N are not positive integers.

    See Also
    --------
    eq_point_set, polar2cart, cart2polar2

    Notes
    -----
    - If dim > 3, extra offsets are ignored.
    - Points are arranged such that each region center is the centre of the
      corresponding product interval in spherical coordinates, except for
      polar caps where the spherical cap centre is used.

    Examples
    --------
    >>> pts = eq_point_set_polar(2, 4)  # doctest: +ELLIPSIS
    >>> pts.shape
    (2, 4)
    """
    # Extra offsets not used for dim > 3
    if dim > 3:
        extra_offset = False

    if not (isinstance(dim, (int, np.integer)) and dim >= 1):
        raise ValueError("dim must be a positive integer")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("N must be a positive integer")

    if N == 1:
        points_s = np.zeros((dim, 1))
        return points_s

    a_cap, n_regions = eq_caps(dim, N)
    # a_cap is increasing list of angles of caps

    if dim == 1:
        # Circle: points placed half way along each sector
        points_s = a_cap - pi / N
        points_s = np.asarray(points_s).reshape((1, N))
        return points_s

    n_collars = int(np.size(n_regions) - 2)
    use_cache = dim >= 2
    cache = None
    if use_cache:
        cache_size = n_collars // 2
        cache = [None] * max(1, cache_size)

    points_s = np.zeros((dim, N))
    point_n = 1  # MATLAB indexing: points_s column 1 is north pole
    # North polar cap 'centre' is North pole (all zeros except last angle 0)
    # Represented by zeros column; will fill first column later
    # For extra_offset rotation bookkeeping
    if extra_offset and (dim == 3):
        R = np.eye(3)
    if dim == 2:
        offset = 0.0

    # Start with north pole
    points_s[:, 0] = 0.0
    point_n = 1

    for collar_n in range(1, n_collars + 1):
        a_top = a_cap[collar_n - 1]
        a_bot = a_cap[collar_n]
        n_in_collar = int(n_regions[collar_n])

        # Partition the (dim-1)-sphere into n_in_collar regions
        if use_cache:
            twin_collar_n = n_collars - collar_n + 1
            if (
                twin_collar_n <= (len(cache))
                and cache[twin_collar_n - 1] is not None
                and cache[twin_collar_n - 1].shape[1] == n_in_collar
            ):
                points_1 = cache[twin_collar_n - 1]
            else:
                points_1 = eq_point_set_polar(dim - 1, n_in_collar, extra_offset)
                if collar_n <= len(cache):
                    cache[collar_n - 1] = points_1
        else:
            points_1 = eq_point_set_polar(dim - 1, n_in_collar, extra_offset)

        if extra_offset and (dim == 3) and (collar_n > 1):
            # Rotate 2-spheres to prevent alignment of north poles.
            R = s2_offset(points_1) @ R
            points_1 = cart2polar2(R @ polar2cart(points_1))

        a_point = (a_top + a_bot) / 2.0
        n_points_1 = points_1.shape[1]
        point_idx = np.arange(point_n, point_n + n_points_1)

        if dim == 2:
            # 1D angles on circle
            pts = points_1[:, :].flatten()
            pts = np.mod(pts + TAU * offset, TAU)
            points_s[0, point_idx] = pts
            # Update offset
            next_n = (
                int(n_regions[collar_n + 1]) if (collar_n + 1) < len(n_regions) else 0
            )
            offset = offset + circle_offset(n_in_collar, next_n, extra_offset)
            offset = offset - math.floor(offset)
        else:
            points_s[0 : dim - 1, point_idx] = points_1[:, :]

        points_s[dim - 1, point_idx] = a_point
        point_n = point_n + n_points_1

    # Bottom polar cap centre
    points_s[:, point_n] = 0.0
    points_s[dim - 1, point_n] = pi
    return points_s


def eq_regions(dim, N, extra_offset=False):
    """
    Recursive zonal equal area (EQ) partition of sphere.

    Parameters
    ----------
    dim : int
        The spatial dimension of the sphere (S^dim).
    N : int
        The number of regions.
    extra_offset : bool
        If True, enables experimental extra offsets for dim 2 and 3.

    Returns
    -------
    regions : ndarray
        Array of shape (dim, 2, N) representing the regions. Each region is
        a pair of vertex points in spherical polar coordinates. regions[:,0,k]
        and regions[:,1,k] give the lower and upper limits of the k-th region.
    dim_1_rot : list (optional)
        If requested (by caller), a list of N rotation matrices, each of size
        (dim, dim), describing R^dim rotations needed to place regions when
        extra offsets are used (only meaningful for dim == 3).

    Raises
    ------
    ValueError
        If dim or N are not positive integers.

    See Also
    --------
    eq_point_set, centres_of_regions

    Notes
    -----
    - For N == 1, the single region is the whole sphere.
    - If extra_offset is used and dim == 3, the returned dim_1_rot describes
      the rotation applied to sub-spheres.

    Examples
    --------
    >>> regs = eq_regions(2, 4)  # doctest: +ELLIPSIS
    >>> regs.shape
    (2, 2, 4)
    """

    if dim > 3:
        extra_offset = False

    if not (isinstance(dim, (int, np.integer)) and dim >= 1):
        raise ValueError("dim must be a positive integer")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("N must be a positive integer")

    dim_1_rot = None
    # Prepare output rotation containers if caller wants them.
    dim_1_rot = [None] * N

    if N == 1:
        regions = np.zeros((dim, 2, 1))
        regions[:, :, 0] = sphere_region(dim)
        if extra_offset and dim == 3:
            dim_1_rot[0] = np.eye(dim)
            return regions, dim_1_rot
        return regions

    s_cap, n_regions = eq_caps(dim, N)

    if dim == 1:
        # Circle: return pairs of sector angles
        regions = np.zeros((dim, 2, N))
        if N > 1:
            regions[:, 0, 1:N] = s_cap[0 : N - 1]
        regions[:, 1, :] = s_cap
        # rotations are identity
        if extra_offset:  # Should dim=1 return rot? Doc says only dim=3 meaningful.
            # But let's assume if extra_offset was requested.
            # Actually existing code returned it.
            # But doc says "only meaningful for dim == 3".
            # Let's simple check extra_offset.
            for idx in range(N):
                dim_1_rot[idx] = np.eye(dim)
            return regions, dim_1_rot
        return regions

    n_collars = int(np.size(n_regions) - 2)
    use_cache = dim > 2
    cache = None
    if use_cache:
        cache_size = n_collars // 2
        cache = [None] * max(1, cache_size)

    regions = np.zeros((dim, 2, N))
    # Top cap
    regions[:, :, 0] = top_cap_region(dim, s_cap[0])
    region_n = 0

    if (True) or (extra_offset and dim == 3):
        R = np.eye(dim)
    dim_1_rot[0] = R

    if dim == 2:
        offset = 0.0

    for collar_n in range(1, n_collars + 1):
        c_top = s_cap[collar_n - 1]
        c_bot = s_cap[collar_n]
        n_in_collar = int(n_regions[collar_n])

        # Partition the (dim-1)-sphere
        if use_cache:
            twin_collar_n = n_collars - collar_n + 1
            if (
                twin_collar_n <= len(cache)
                and cache[twin_collar_n - 1] is not None
                and cache[twin_collar_n - 1].shape[2] == n_in_collar
            ):
                regions_1 = cache[twin_collar_n - 1]
            else:
                if extra_offset:
                    regions_1, _ = eq_regions(dim - 1, n_in_collar, extra_offset)
                else:
                    regions_1 = eq_regions(dim - 1, n_in_collar, extra_offset)
                if collar_n <= len(cache):
                    cache[collar_n - 1] = regions_1
        else:
            if extra_offset:
                regions_1, _ = eq_regions(dim - 1, n_in_collar, extra_offset)
            else:
                regions_1 = eq_regions(dim - 1, n_in_collar, extra_offset)

        if extra_offset and (dim == 3) and (collar_n > 1):
            R = s2_offset(centres_of_regions(regions_1)) @ R

        # Append regions for this collar
        for region_1_n in range(regions_1.shape[2]):
            region_n += 1
            if dim == 2:
                r_top = np.array(
                    [np.mod(regions_1[0, 0, region_1_n] + TAU * offset, TAU), c_top]
                )
                r_bot = np.array(
                    [np.mod(regions_1[0, 1, region_1_n] + TAU * offset, TAU), c_bot]
                )
                if r_bot[0] < r_top[0]:
                    r_bot[0] = r_bot[0] + TAU
                regions[:, :, region_n] = np.vstack((r_top, r_bot)).T
            else:
                regions[0 : dim - 1, :, region_n] = regions_1[:, :, region_1_n]
                regions[dim - 1, :, region_n] = np.array([c_top, c_bot])

            if dim_1_rot is not None:
                dim_1_rot[region_n] = R

        if dim == 2:
            next_n = (
                int(n_regions[collar_n + 1]) if (collar_n + 1) < len(n_regions) else 0
            )
            offset = offset + circle_offset(n_in_collar, next_n, extra_offset)
            offset = offset - math.floor(offset)

    # Bottom cap
    regions[:, :, N - 1] = bot_cap_region(dim, s_cap[0])
    if (dim == 3) and extra_offset:
        dim_1_rot[N - 1] = np.eye(dim)
        return regions, dim_1_rot
    else:
        return regions
