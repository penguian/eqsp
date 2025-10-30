# Translated from partitions.m (partial)
# NOTE: This module imports a number of helper functions which must be
# available as Python modules in the same package or on sys.path.  The
# original MATLAB code used separately defined functions; the user asked
# that every undefined function be imported rather than stubbed.

import math

import matplotlib.pyplot as plt
import numpy as np

from polar_colat import polar_colat
from ideal_collar_angle import ideal_collar_angle
from num_collars import num_collars
from ideal_region_list import ideal_region_list
from round_to_naturals import round_to_naturals
from cap_colats import cap_colats
from polar2cart import polar2cart
from partition_options import partition_options
from s2_offset import s2_offset
from cart2polar2 import cart2polar2
from circle_offset import circle_offset
from centres_of_regions import centres_of_regions
from top_cap_region import top_cap_region
from bot_cap_region import bot_cap_region
from sphere_region import sphere_region
from project_s2_partition import project_s2_partition
from illustration_options import illustration_options
from option_arguments import option_arguments


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
    >>> eq_caps(2, 10)  # doctest: +ELLIPSIS
    (array([0.6435..., 1.5708..., 2.4981..., 3.1416...]), array([1, 4, 4, 1]))
    >>> eq_caps(3, 6)  # doctest: +ELLIPSIS
    (array([0.9845..., 2.1571..., 3.1416...]), array([1, 4, 1]))
    """
    if not (isinstance(dim, (int, np.integer)) and dim >= 1):
        raise ValueError("dim must be a positive integer")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("N must be a positive integer")

    if dim == 1:
        # Circle: return the angles of N equal sectors.
        sector = np.arange(1, N + 1)
        s_cap = sector * 2.0 * math.pi / N
        n_regions = np.ones_like(sector, dtype=int)
    elif N == 1:
        # Only one region: whole sphere.
        s_cap = np.array([math.pi])
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

    return np.asarray(s_cap), np.asarray(n_regions)


def eq_point_set(dim, N, *args):
    """
    Center points of regions of EQ partition, in Cartesian coordinates.

    Parameters
    ----------
    dim : int
        The number of dimensions.
    N : int
        The number of regions.
    *args :
        Optional partition options passed to eq_point_set_polar.

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
    points_polar = eq_point_set_polar(dim, N, *args)
    points_x = polar2cart(points_polar)
    return points_x


def eq_point_set_polar(dim, N, *args):
    """
    Center points of regions of an EQ partition in polar coordinates.

    Parameters
    ----------
    dim : int
        The spatial dimension of the sphere (S^dim in R^{dim+1}).
    N : int
        The number of regions.
    *args :
        Optional arguments forwarded to partition_options. The option
        'offset','extra' enables experimental extra offsets for dim 2 and 3.

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
    partition_options, eq_point_set, polar2cart, cart2polar2

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
    # Default options
    pdefault = {"extra_offset": False}
    if len(args) == 0:
        extra_offset = pdefault["extra_offset"]
    else:
        popt = partition_options(pdefault, *args)
        # Accept either attribute or dict style
        extra_offset = getattr(
            popt, "extra_offset", popt.get("extra_offset", pdefault["extra_offset"])
        )

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
        points_s = a_cap - math.pi / N
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
            pts = np.mod(pts + 2.0 * math.pi * offset, 2.0 * math.pi)
            points_s[0, point_idx] = pts
            # Update offset
            next_n = int(n_regions[collar_n + 1]) if (collar_n + 1) < len(n_regions) else 0
            offset = offset + circle_offset(n_in_collar, next_n, extra_offset)
            offset = offset - math.floor(offset)
        else:
            points_s[0 : dim - 1, point_idx] = points_1[:, :]

        points_s[dim - 1, point_idx] = a_point
        point_n = point_n + n_points_1

    # Bottom polar cap centre
    points_s[:, point_n] = 0.0
    points_s[dim - 1, point_n] = math.pi
    return points_s


def eq_regions(dim, N, *args):
    """
    Recursive zonal equal area (EQ) partition of sphere.

    Parameters
    ----------
    dim : int
        The spatial dimension of the sphere (S^dim).
    N : int
        The number of regions.
    *args :
        Optional arguments forwarded to partition_options. Recognizes the
        'offset','extra' option.

    Returns
    -------
    regions : ndarray
        Array of shape (dim, 2, N) representing the regions. Each region is
        a pair of vertex points in spherical polar coordinates. regions[:,0,k]
        and regions[:,1,k] give the lower and upper limits of the k-th region.
    dim_1_rot : list
        If requested (by caller), a list of N rotation matrices, each of size
        (dim, dim), describing R^dim rotations needed to place regions when
        extra offsets are used (only meaningful for dim == 3).

    Raises
    ------
    ValueError
        If dim or N are not positive integers.

    See Also
    --------
    eq_point_set, partition_options, centres_of_regions

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
    # Default options
    pdefault = {"extra_offset": False}
    if len(args) == 0:
        extra_offset = pdefault["extra_offset"]
    else:
        popt = partition_options(pdefault, *args)
        extra_offset = getattr(
            popt, "extra_offset", popt.get("extra_offset", pdefault["extra_offset"])
        )

    if dim > 3:
        extra_offset = False

    if not (isinstance(dim, (int, np.integer)) and dim >= 1):
        raise ValueError("dim must be a positive integer")
    if not (isinstance(N, (int, np.integer)) and N >= 1):
        raise ValueError("N must be a positive integer")

    dim_1_rot = None
    # Prepare output rotation containers if caller wants them; Python
    # alternative: always prepare and return.
    dim_1_rot = [None] * N

    if N == 1:
        regions = np.zeros((dim, 2, 1))
        regions[:, :, 0] = sphere_region(dim)
        dim_1_rot[0] = np.eye(dim)
        return regions, dim_1_rot

    s_cap, n_regions = eq_caps(dim, N)

    if dim == 1:
        # Circle: return pairs of sector angles
        regions = np.zeros((dim, 2, N))
        if N > 1:
            regions[:, 0, 1:N] = s_cap[0 : N - 1]
        regions[:, 1, :] = s_cap
        # rotations are identity
        for idx in range(N):
            dim_1_rot[idx] = np.eye(dim)
        return regions, dim_1_rot

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
                regions_1, _ = eq_regions(dim - 1, n_in_collar, extra_offset)
                cache[collar_n - 1] = regions_1
        else:
            regions_1, _ = eq_regions(dim - 1, n_in_collar, extra_offset)

        if extra_offset and (dim == 3) and (collar_n > 1):
            R = s2_offset(centres_of_regions(regions_1)) @ R

        # Append regions for this collar
        for region_1_n in range(regions_1.shape[2]):
            region_n += 1
            if dim == 2:
                r_top = np.array(
                    [np.mod(regions_1[0, 0, region_1_n] + 2.0 * math.pi * offset, 2.0 * math.pi),
                     c_top]
                )
                r_bot = np.array(
                    [np.mod(regions_1[0, 1, region_1_n] + 2.0 * math.pi * offset, 2.0 * math.pi),
                     c_bot]
                )
                if r_bot[0] < r_top[0]:
                    r_bot[0] = r_bot[0] + 2.0 * math.pi
                regions[:, :, region_n] = np.vstack((r_top, r_bot)).T
            else:
                regions[0:dim - 1, :, region_n] = regions_1[:, :, region_1_n]
                regions[dim - 1, :, region_n] = np.array([c_top, c_bot])

            if dim_1_rot is not None:
                dim_1_rot[region_n] = R

        if dim == 2:
            next_n = int(n_regions[collar_n + 1]) if (collar_n + 1) < len(n_regions) else 0
            offset = offset + circle_offset(n_in_collar, next_n, extra_offset)
            offset = offset - math.floor(offset)

    # Bottom cap
    regions[:, :, N - 1] = bot_cap_region(dim, s_cap[0])
    dim_1_rot[N - 1] = np.eye(dim)
    return regions, dim_1_rot


def illustrate_eq_algorithm(dim, N, *args):
    """
    Illustrate the EQ partition algorithm with matplotlib subplots.

    Parameters
    ----------
    dim : int
        The spatial dimension of the sphere.
    N : int
        The number of regions.
    *args :
        Options forwarded to partition_options and illustration_options.

    Notes
    -----
    Produces a 2x2 subplot illustration:
    1. Steps 1 and 2
    2. Steps 3 to 5
    3. Steps 6 and 7
    4. Lower dimensional partitions for dim 2 or 3

    Examples
    --------
    >>> illustrate_eq_algorithm(3, 99)  # doctest: +SKIP
    """
    pdefault = {"extra_offset": False}
    popt = partition_options(pdefault, *args)

    gdefault = {
        "fontsize": 16,
        "show_title": True,
        "long_title": False,
        "stereo": False,
        "show_points": True,
    }
    gopt = illustration_options(gdefault, *args)
    opt_args = option_arguments(popt, gopt)

    plt.subplot(2, 2, 1)
    plt.axis("off")
    illustrate_steps_1_2(dim, N, *opt_args)

    plt.subplot(2, 2, 2)
    plt.axis("off")
    illustrate_steps_3_5(dim, N, *opt_args)

    plt.subplot(2, 2, 3)
    plt.axis("off")
    illustrate_steps_6_7(dim, N, *opt_args)

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.cla()

    gopt2 = gopt.copy()
    gopt2["fontsize"] = 32

    if dim == 2:
        project_s2_partition(N, *option_arguments(popt, gopt2))
    elif dim == 3:
        # Extract caps and display a few collars as examples
        _, m = eq_caps(dim, N)
        max_collar = min(4, int(np.size(m) - 2))
        for k in range(1, max_collar + 1):
            subn = 9 + 2 * k - ((k - 1) % 2)
            plt.subplot(4, 4, subn)
            plt.axis("off")
            project_s2_partition(int(m[0 + k]), *option_arguments(popt, gopt2))


def illustrate_steps_1_2(dim, N, *args):
    """
    Illustrate steps 1 and 2 of the EQ partition on a circle representation.

    Parameters
    ----------
    dim : int
        The dimension of the sphere.
    N : int
        The number of regions.
    *args :
        Options forwarded to illustration_options.

    Notes
    -----
    Produces a 2D plot showing the polar cap colatitude and the ideal collar
    angle.

    Examples
    --------
    >>> illustrate_steps_1_2(3, 10)  # doctest: +SKIP
    """
    gdefault = {"fontsize": 14, "show_title": True, "long_title": False}
    gopt = illustration_options(gdefault, *args)

    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * math.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")
    plt.hold(False) if hasattr(plt, "hold") else None
    plt.hold(True) if hasattr(plt, "hold") else None

    c_polar = polar_colat(dim, N)

    k = np.linspace(-1.0, 1.0, 21)
    j = np.ones_like(k)

    # Bounding parallels of the polar caps
    plt.plot(np.sin(c_polar) * k, np.cos(c_polar) * j, "r", linewidth=2)
    plt.plot(np.sin(c_polar) * k, -np.cos(c_polar) * j, "r", linewidth=2)

    # North-South axis
    plt.plot(np.zeros_like(j), k, "b", linewidth=1)
    # Polar angle
    plt.plot(np.sin(c_polar) * h, np.cos(c_polar) * h, "b", linewidth=2)
    plt.text(0.05, 2.0 / 3.0, r"$\theta_c$", fontsize=gopt["fontsize"])

    # Ideal collar angle
    Delta_I = ideal_collar_angle(dim, N)
    theta = c_polar + Delta_I
    plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)

    mid = c_polar + Delta_I / 2.0
    plt.text(np.sin(mid) * 2.0 / 3.0, np.cos(mid) * 2.0 / 3.0, r"$\Delta_I$",
             fontsize=gopt["fontsize"])

    # Arc to indicate angles
    theta = h * (c_polar + Delta_I)
    plt.plot(np.sin(theta) / 5.0, np.cos(theta) / 5.0, "b", linewidth=1)

    plt.text(
        -0.9,
        -0.1,
        f"V(\\theta_c) = V_R \\n    = \\sigma(S^{dim})/{N}",
        fontsize=gopt["fontsize"],
    )

    caption_angle = min(mid + 2.0 * Delta_I, math.pi - c_polar)
    plt.text(
        np.sin(caption_angle) / 3.0,
        np.cos(caption_angle) / 3.0,
        rf"$\Delta_I = V_R^{{1/{dim}}}$",
        fontsize=gopt["fontsize"],
    )

    if gopt["show_title"]:
        title_str = f"EQ({dim},{N}) Steps 1 to 2\n"
        plt.title(title_str, fontsize=gopt["fontsize"])


def illustrate_steps_3_5(dim, N, *args):
    """
    Illustrate steps 3 to 5 of the EQ partition.

    Parameters
    ----------
    dim : int
        The spatial dimension.
    N : int
        Number of regions.
    *args :
        Options forwarded to illustration_options.

    Examples
    --------
    >>> illustrate_steps_3_5(3, 50)  # doctest: +SKIP
    """
    gdefault = {"fontsize": 14, "show_title": True, "long_title": False}
    gopt = illustration_options(gdefault, *args)

    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * math.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")

    c_polar = polar_colat(dim, N)
    n_collars = num_collars(N, c_polar, ideal_collar_angle(dim, N))
    r_regions = ideal_region_list(dim, N, c_polar, n_collars)
    s_cap = cap_colats(dim, N, c_polar, r_regions)

    k = np.linspace(-1.0, 1.0, 21)
    j = np.ones_like(k)
    plt.plot(np.sin(c_polar) * k, np.cos(c_polar) * j, "r", linewidth=2)
    plt.plot(np.zeros_like(j), k, "b", linewidth=1)

    for collar_n in range(0, n_collars + 1):
        zone_n = 1 + collar_n
        theta = s_cap[zone_n - 1]
        plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)
        theta_str = rf"\theta_{{F,{zone_n}}}"
        plt.text(np.sin(theta) * 1.1, np.cos(theta) * 1.1, theta_str,
                 fontsize=gopt["fontsize"])
        if collar_n != 0:
            plt.plot(np.sin(theta) * k, np.cos(theta) * j, "r", linewidth=2)
            theta_p = s_cap[collar_n - 1]
            arc = theta_p + (theta - theta_p) * h
            plt.plot(np.sin(arc) / 5.0, np.cos(arc) / 5.0, "b", linewidth=1)
            mid = (theta_p + theta) / 2.0
            plt.text(np.sin(mid) / 2.0, np.cos(mid) / 2.0, r"$\Delta_F$",
                     fontsize=gopt["fontsize"])
            y_str = f"y_{{{collar_n}}} = {r_regions[zone_n-1]:3.1f}..."
            plt.text(-np.sin(mid) + 1.0 / 20.0, np.cos(mid) + (mid - math.pi) / 30.0,
                     y_str, fontsize=gopt["fontsize"])

    if gopt["show_title"]:
        title_str = f"EQ({dim},{N}) Steps 3 to 5\n"
        plt.title(title_str, fontsize=gopt["fontsize"])


def illustrate_steps_6_7(dim, N, *args):
    """
    Illustrate steps 6 and 7 of the EQ partition.

    Parameters
    ----------
    dim : int
        The spatial dimension.
    N : int
        Number of regions.
    *args :
        Options forwarded to illustration_options.

    Examples
    --------
    >>> illustrate_steps_6_7(3, 50)  # doctest: +SKIP
    """
    gdefault = {"fontsize": 14, "show_title": True, "long_title": False}
    gopt = illustration_options(gdefault, *args)

    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * math.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")

    c_polar = polar_colat(dim, N)
    # The original MATLAB file continues here with further plotting of the
    # final steps of the algorithm. The full body was truncated in the
    # provided input. Implementers should complete plotting consistent
    # with illustrate_steps_1_2 and _3_5 based on the original MATLAB file.
    raise NotImplementedError(
        "illustrate_steps_6_7 translation incomplete; source was truncated."
    )
