"""
Python translation of illustrate_eq_algorithm.m

This module provides functions to illustrate the recursive zonal equal area
sphere partitioning algorithm. It reproduces the plotting and steps from the
original MATLAB implementation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .partitions import eq_caps
from .utilities import ideal_collar_angle
from ._private._partitions import cap_colats, ideal_region_list, num_collars, polar_colat, round_to_naturals
from .illustrations import project_s2_partition


def illustrate_eq_algorithm(dim, N, *, extra_offset=False, fontsize=16,
                            show_title=True, long_title=False, stereo=False,
                            show_points=True, proj='stereo', **kwargs):
    """
    Illustrate the EQ partition algorithm.

    Parameters
    ----------
    dim : int
        Dimension of the sphere S^dim (unit sphere in R^{dim+1}).
    N : int
        Number of equal-area regions to partition the sphere into.
    extra_offset : bool, optional
        Use extra offsets. Default False.
    fontsize : int, optional
        Font size. Default 16.
    show_title : bool, optional
        Show title. Default True.
    long_title : bool, optional
        Use long title variant. Default False.
    stereo : bool, optional
        Use stereographic projection. Default False.
    show_points : bool, optional
        Show center points. Default True.
    proj : str, optional
        Projection type ('stereo' or 'eqarea'). Default 'stereo'.
    **kwargs
        Passed to underlying plot functions.

    Returns
    -------
    None
    """
    # Consolidate options for easier passing
    opts = {
        'extra_offset': extra_offset,
        'fontsize': fontsize,
        'show_title': show_title,
        'long_title': long_title,
    }

    plt.subplot(2, 2, 1)
    plt.axis("off")
    illustrate_steps_1_2(dim, N, **opts)

    plt.subplot(2, 2, 2)
    plt.axis("off")
    illustrate_steps_3_5(dim, N, **opts)

    plt.subplot(2, 2, 3)
    plt.axis("off")
    illustrate_steps_6_7(dim, N, **opts)

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.cla()

    # Increase font size for lower-dim visualisations
    proj_opts = opts.copy()
    proj_opts["fontsize"] = 32
    # Map stereo/proj options to illustration functions
    if stereo:
        proj_opts['proj'] = 'stereo'
    else:
        proj_opts['proj'] = proj
        
    proj_opts['show_points'] = show_points
    
    # Map title mode
    if show_title:
        proj_opts['title'] = 'long' if long_title else 'short'
    else:
        proj_opts['title'] = 'none'

    if dim == 2:
        project_s2_partition(N, **proj_opts)
    elif dim == 3:
        _, m = eq_caps(dim, N)
        # m is n_regions (1D array)
        max_collar = min(4, m.size - 2)
        for k in range(1, max_collar + 1):
            subn = 9 + 2 * k - ((k - 1) % 2)
            plt.subplot(4, 4, subn)
            plt.axis("off")
            project_s2_partition(int(m[k]), **proj_opts)


def illustrate_steps_1_2(dim, N, *, fontsize=14, show_title=True, long_title=False, **kwargs):
    """
    Illustrate steps 1 and 2 of the EQ partition.
    """
    h = np.linspace(0.0, 1.0, 91)
    Phi = h * 2.0 * np.pi
    plt.plot(np.sin(Phi), np.cos(Phi), "k", linewidth=1)
    plt.axis("equal")
    plt.axis("off")

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

    plt.text(0.05, 2.0 / 3.0, r"$\theta_c$", fontsize=fontsize)

    # Plot the ideal collar angle
    Delta_I = ideal_collar_angle(dim, N)
    theta = c_polar + Delta_I
    plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)

    mid = c_polar + Delta_I / 2.0
    plt.text(np.sin(mid) * 2.0 / 3.0, np.cos(mid) * 2.0 / 3.0, r"$\Delta_I$",
             fontsize=fontsize)

    # Plot an arc to indicate angles
    theta = h * (c_polar + Delta_I)
    plt.plot(np.sin(theta) / 5.0, np.cos(theta) / 5.0, "b", linewidth=1)

    plt.text(-0.9, -0.1,
             "V(\\theta_c) = V_R \n    = \\sigma(S^{%d})/%d" % (dim, N),
             fontsize=fontsize)

    caption_angle = min(mid + 2.0 * Delta_I, np.pi - c_polar)
    plt.text(np.sin(caption_angle) / 3.0,
             np.cos(caption_angle) / 3.0,
             r"$\Delta_I = V_R^{1/%d}$" % dim,
             fontsize=fontsize)

    if show_title:
        title_str = f"EQ({dim},{N}) Steps 1 to 2\n"
        plt.title(title_str, fontsize=fontsize, color='k')


def illustrate_steps_3_5(dim, N, *, fontsize=14, show_title=True, long_title=False, **kwargs):
    """
    Illustrate steps 3 to 5 of the EQ partition.
    """
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
                 theta_str, fontsize=fontsize)
        if collar_n != 0:
            plt.plot(np.sin(theta) * k, np.cos(theta) * j, "r", linewidth=2)
            theta_p = s_cap[collar_n]
            arc = theta_p + (theta - theta_p) * h
            plt.plot(np.sin(arc) / 5.0, np.cos(arc) / 5.0, "b",
                     linewidth=1)
            mid = (theta_p + theta) / 2.0
            plt.text(np.sin(mid) / 2.0, np.cos(mid) / 2.0, r"$\Delta_F$",
                     fontsize=fontsize)
            y_str = "y_{%d} = %3.1f..." % (collar_n, r_regions[zone_n])
            plt.text(-np.sin(mid) + 1.0 / 20.0,
                     np.cos(mid) + (mid - np.pi) / 30.0,
                     y_str, fontsize=fontsize)
    if show_title:
        title_str = f"EQ({dim},{N}) Steps 3 to 5\n"
        plt.title(title_str, fontsize=fontsize, color='k')


def illustrate_steps_6_7(dim, N, *, fontsize=14, show_title=True, long_title=False, **kwargs):
    """
    Illustrate steps 6 and 7 of the EQ partition.
    """
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
                 theta_str, fontsize=fontsize)
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
                     Delta_str, fontsize=fontsize)
            m_str = "m_{%d} =%3.0f" % (collar_n, n_regions[zone_n])
            plt.text(-np.sin(mid) + 1.0 / 20.0,
                     np.cos(mid) + (mid - np.pi) / 30.0,
                     m_str, fontsize=fontsize)
    if show_title:
        title_str = f"EQ({dim},{N}) Steps 6 to 7\n"
        plt.title(title_str, fontsize=fontsize, color='k')
