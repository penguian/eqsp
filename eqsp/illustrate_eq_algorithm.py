"""
Python translation of illustrate_eq_algorithm.m

This module provides functions to illustrate the recursive zonal equal area
sphere partitioning algorithm. It reproduces the plotting and steps from the
original MATLAB implementation.

Note
----
This file imports helper functions from the package namespace. The package
must expose the following names at the package level for these functions to
work:

- partition_options
- illustration_options
- polar_colat
- ideal_collar_angle
- eq_caps
- project_s2_partition
- num_collars
- ideal_region_list
- cap_colats
- round_to_naturals

Matplotlib and numpy are used for plotting and numeric operations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import (
    cap_colats,
    eq_caps,
    ideal_collar_angle,
    ideal_region_list,
    num_collars,
    partition_options,
    polar_colat,
    project_s2_partition,
    round_to_naturals,
    illustration_options,
)


def illustrate_eq_algorithm(dim, N, *varargin):
    """
    Illustrate the EQ partition algorithm.

    Parameters
    ----------
    dim : int
        Dimension of the sphere S^dim (unit sphere in R^{dim+1}).
    N : int
        Number of equal-area regions to partition the sphere into.
    *varargin
        Additional option name/value pairs forwarded to partition and
        illustration option parsers.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If dim or N are not sensible for partitioning (delegated to helper
        functions).

    See Also
    --------
    partition_options, illustration_options, project_s2_partition

    Notes
    -----
    The illustration consists of four subplots showing intermediate steps of
    the recursive zonal equal area partitioning algorithm:

    1. Steps 1 and 2
    2. Steps 3 to 5
    3. Steps 6 and 7
    4. Lower dimensional partitions (if dim == 2 or dim == 3)

    The function supports an experimental "extra" offset mode via the option
    pair ('offset', 'extra') for dim == 2 or dim == 3. More partition and
    illustration options are accepted; see the helpers for details.

    Examples
    --------
    >>> # Basic call (this will create figures)
    >>> illustrate_eq_algorithm(3, 99)
    >>> # Use equal-area projection and extra offset
    >>> illustrate_eq_algorithm(3, 99, 'offset', 'extra', 'proj', 'eqarea')
    """
    pdefault = {"extra_offset": False}
    popt = partition_options(pdefault, *varargin)

    gdefault = {
        "fontsize": 16,
        "show_title": True,
        "long_title": False,
        "stereo": False,
        "show_points": True,
    }

    gopt = illustration_options(gdefault, *varargin)
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

    # Increase font size for lower-dim visualisations
    gopt["fontsize"] = 32
    if dim == 2:
        opt_args = option_arguments(popt, gopt)
        project_s2_partition(N, *opt_args)
    elif dim == 3:
        opt_args = option_arguments(popt, gopt)
        _, m = eq_caps(dim, N)
        max_collar = min(4, m.shape[1] - 2)
        for k in range(1, max_collar + 1):
            subn = 9 + 2 * k - ((k - 1) % 2)
            plt.subplot(4, 4, subn)
            plt.axis("off")
            project_s2_partition(int(m[0, k]), *opt_args)


def illustrate_steps_1_2(dim, N, *varargin):
    """
    Illustrate steps 1 and 2 of the EQ partition.

    Parameters
    ----------
    dim : int
        Sphere dimension S^dim.
    N : int
        Number of regions.
    *varargin
        Option name/value pairs for illustration and partition options.

    Raises
    ------
    ValueError
        If inputs are invalid (delegated to helper functions).

    See Also
    --------
    polar_colat, ideal_collar_angle

    Notes
    -----
    The function draws a unit circle to represent the dth coordinate of S^d,
    plots polar cap bounding parallels, the North-South axis, the polar angle
    and the ideal collar angle with annotations.

    Examples
    --------
    >>> illustrate_steps_1_2(3, 99)
    """
    gdefault = {"fontsize": 14, "show_title": True, "long_title": False}
    gopt = illustration_options(gdefault, *varargin)

    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * np.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")
    # hold on: not necessary in matplotlib; subsequent plot calls add to
    # current axes

    c_polar = polar_colat(dim, N)

    k = np.linspace(-1.0, 1.0, 41)
    j = np.ones_like(k)

    # Plot the bounding parallels of the polar caps
    plt.plot(np.sin(c_polar) * k, np.cos(c_polar) * j, "r", linewidth=2)
    plt.plot(np.sin(c_polar) * k, -np.cos(c_polar) * j, "r", linewidth=2)

    # Plot the North-South axis
    plt.plot(np.zeros_like(j), k, "b", linewidth=1)
    # Plot the polar angle
    plt.plot(np.sin(c_polar) * h, np.cos(c_polar) * h, "b", linewidth=2)

    plt.text(0.05, 2.0 / 3.0, r"$\theta_c$", fontsize=gopt["fontsize"])

    # Plot the ideal collar angle
    Delta_I = ideal_collar_angle(dim, N)
    theta = c_polar + Delta_I
    plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)

    mid = c_polar + Delta_I / 2.0
    plt.text(np.sin(mid) * 2.0 / 3.0, np.cos(mid) * 2.0 / 3.0, r"$\Delta_I$",
             fontsize=gopt["fontsize"])

    # Plot an arc to indicate angles
    theta = h * (c_polar + Delta_I)
    plt.plot(np.sin(theta) / 5.0, np.cos(theta) / 5.0, "b", linewidth=1)

    plt.text(-0.9, -0.1,
             "V(\\theta_c) = V_R \n    = \\sigma(S^{%d})/%d" % (dim, N),
             fontsize=gopt["fontsize"])

    caption_angle = min(mid + 2.0 * Delta_I, np.pi - c_polar)
    plt.text(np.sin(caption_angle) / 3.0,
             np.cos(caption_angle) / 3.0,
             r"$\Delta_I = V_R^{1/%d}$" % dim,
             fontsize=gopt["fontsize"])

    if gopt.get("show_title", True):
        title_str = f"EQ({dim},{N}) Steps 1 to 2\n"
        plt.title(title_str, fontsize=gopt["fontsize"])


def illustrate_steps_3_5(dim, N, *varargin):
    """
    Illustrate steps 3 to 5 of the EQ partition.

    Parameters
    ----------
    dim : int
        Sphere dimension.
    N : int
        Number of regions.
    *varargin
        Option name/value pairs forwarded to helpers.

    Raises
    ------
    ValueError
        If inputs are invalid (delegated to helper functions).

    See Also
    --------
    polar_colat, ideal_collar_angle, num_collars, ideal_region_list, cap_colats

    Notes
    -----
    The function draws parallels for polar caps and the computed zone
    boundaries. It annotates the ideal number of regions and shows
    representative arcs and captions for each collar.

    Examples
    --------
    >>> illustrate_steps_3_5(3, 99)
    """
    gdefault = {"fontsize": 14, "show_title": True, "long_title": False}
    gopt = illustration_options(gdefault, *varargin)

    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * np.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")

    c_polar = polar_colat(dim, N)
    n_collars = num_collars(N, c_polar, ideal_collar_angle(dim, N))
    r_regions = ideal_region_list(dim, N, c_polar, n_collars)
    s_cap = cap_colats(dim, N, c_polar, r_regions)

    k = np.linspace(-1.0, 1.0, 41)
    j = np.ones_like(k)
    plt.plot(np.sin(c_polar) * k, np.cos(c_polar) * j, "r", linewidth=2)

    plt.plot(np.zeros_like(j), k, "b", linewidth=1)

    for collar_n in range(0, n_collars + 1):
        zone_n = 1 + collar_n
        theta = s_cap[zone_n]
        plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)
        theta_str = r"\theta_{F,%d}" % zone_n
        plt.text(np.sin(theta) * 1.1, np.cos(theta) * 1.1,
                 theta_str, fontsize=gopt["fontsize"])
        if collar_n != 0:
            plt.plot(np.sin(theta) * k, np.cos(theta) * j, "r", linewidth=2)
            theta_p = s_cap[collar_n]
            arc = theta_p + (theta - theta_p) * h
            plt.plot(np.sin(arc) / 5.0, np.cos(arc) / 5.0, "b",
                     linewidth=1)
            mid = (theta_p + theta) / 2.0
            plt.text(np.sin(mid) / 2.0, np.cos(mid) / 2.0, r"$\Delta_F$",
                     fontsize=gopt["fontsize"])
            y_str = "y_{%d} = %3.1f..." % (collar_n, r_regions[zone_n])
            plt.text(-np.sin(mid) + 1.0 / 20.0,
                     np.cos(mid) + (mid - np.pi) / 30.0,
                     y_str, fontsize=gopt["fontsize"])
    if gopt.get("show_title", True):
        title_str = f"EQ({dim},{N}) Steps 3 to 5\n"
        plt.title(title_str, fontsize=gopt["fontsize"])


def illustrate_steps_6_7(dim, N, *varargin):
    """
    Illustrate steps 6 and 7 of the EQ partition.

    Parameters
    ----------
    dim : int
        Sphere dimension.
    N : int
        Number of regions.
    *varargin
        Option name/value pairs forwarded to helpers.

    Raises
    ------
    ValueError
        If inputs are invalid (delegated to helper functions).

    See Also
    --------
    polar_colat, num_collars, ideal_region_list, round_to_naturals, cap_colats

    Notes
    -----
    This draws the final rounded region counts and the final cap colatitudes
    after rounding to integer numbers of regions per collar.

    Examples
    --------
    >>> illustrate_steps_6_7(3, 99)
    """
    gdefault = {"fontsize": 14, "show_title": True, "long_title": False}
    gopt = illustration_options(gdefault, *varargin)

    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * np.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")

    c_polar = polar_colat(dim, N)
    n_collars = num_collars(N, c_polar, ideal_collar_angle(dim, N))
    r_regions = ideal_region_list(dim, N, c_polar, n_collars)
    n_regions = round_to_naturals(N, r_regions)
    s_cap = cap_colats(dim, N, c_polar, n_regions)

    k = np.linspace(-1.0, 1.0, 41)
    j = np.ones_like(k)
    plt.plot(np.sin(c_polar) * k, np.cos(c_polar) * j, "r", linewidth=2)

    plt.plot(np.zeros_like(j), k, "b", linewidth=1)

    for collar_n in range(0, n_collars + 1):
        zone_n = 1 + collar_n
        theta = s_cap[zone_n]
        plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)
        theta_str = r"\theta_{%d}" % zone_n
        plt.text(np.sin(theta) * 1.1, np.cos(theta) * 1.1,
                 theta_str, fontsize=gopt["fontsize"])
        if collar_n != 0:
            plt.plot(np.sin(theta) * k, np.cos(theta) * j, "r",
                     linewidth=2)
            theta_p = s_cap[collar_n]
            arc = theta_p + (theta - theta_p) * h
            plt.plot(np.sin(arc) / 5.0, np.cos(arc) / 5.0, "b",
                     linewidth=1)
            mid = (theta_p + theta) / 2.0
            Delta_str = r"\Delta_{%i}" % collar_n
            plt.text(np.sin(mid) / 2.0, np.cos(mid) / 2.0,
                     Delta_str, fontsize=gopt["fontsize"])
            m_str = "m_{%d} =%3.0f" % (collar_n, n_regions[zone_n])
            plt.text(-np.sin(mid) + 1.0 / 20.0,
                     np.cos(mid) + (mid - np.pi) / 30.0,
                     m_str, fontsize=gopt["fontsize"])
    if gopt.get("show_title", True):
        title_str = f"EQ({dim},{N}) Steps 6 to 7\n"
        plt.title(title_str, fontsize=gopt["fontsize"])


def option_arguments(popt, gopt):
    """
    Convert partition and illustration option dicts to argument list.

    Parameters
    ----------
    popt : dict
        Partition options (as returned by partition_options).
    gopt : dict
        Illustration options (as returned by illustration_options).

    Returns
    -------
    list
        List of alternating option name and value to be used as
        positional arguments (suitable for unpacking with *).

    See Also
    --------
    partition_options, illustration_options

    Notes
    -----
    The returned list mirrors the MATLAB option name/value pair packing:
    e.g. ['offset', 'extra', 'fontsize', 14, 'proj', 'eqarea', ...].

    Examples
    --------
    >>> option_arguments({'extra_offset': True}, {'fontsize': 12})
    ['offset', 'extra', 'fontsize', 12]
    """
    arg = []
    k = 0
    if "extra_offset" in popt:
        arg.append("offset")
        if popt["extra_offset"]:
            arg.append("extra")
        else:
            arg.append("normal")
        k += 2

    if "fontsize" in gopt:
        arg.append("fontsize")
        arg.append(gopt["fontsize"])
        k += 2

    if "stereo" in gopt:
        arg.append("proj")
        if gopt["stereo"]:
            arg.append("stereo")
        else:
            arg.append("eqarea")
        k += 2

    if "show_title" in gopt:
        arg.append("title")
        if gopt["show_title"]:
            if "long_title" in gopt:
                if gopt["long_title"]:
                    arg.append("long")
                else:
                    arg.append("short")
            else:
                arg.append("show")
        else:
            arg.append("none")
        k += 2
    elif "long_title" in gopt:
        arg.append("title")
        if gopt["long_title"]:
            arg.append("long")
        else:
            arg.append("short")
        k += 2

    if "show_points" in gopt:
        arg.append("points")
        if gopt["show_points"]:
            arg.append("show")
        else:
            arg.append("hide")
        k += 2

    if "show_surfaces" in gopt:
        arg.append("surf")
        if gopt["show_surfaces"]:
            arg.append("show")
        else:
            arg.append("hide")

    return arg
