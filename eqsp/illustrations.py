"""
PyEQSP illustrations module.

Copyright 2026 Paul Leopardi
"""

import matplotlib.pyplot as plt
import numpy as np

from ._private._partitions import (
    cap_colats,
    ideal_region_list,
    num_collars,
    polar_colat,
    round_to_naturals,
)
from .partitions import eq_caps, eq_point_set, eq_regions
from .utilities import (
    ideal_collar_angle,
    polar2cart,
    x2eqarea,
    x2stereo,
)


def project_point_set(
    points, ax=None, proj="stereo", _scale_factor=0.05, color=None, show=None, **_kwargs
):
    """
    Use projection to illustrate a point set of S^2 or S^3.

    Parameters
    ----------
    points : ndarray
        Points in R^3 (S^2) or R^4 (S^3).
    ax : Axes, optional
        Matplotlib axes (2D or 3D). If None, a new figure is created.
    proj : {'stereo', 'eqarea'}, optional
        Projection type. Default is 'stereo'.
    _scale_factor : float, optional
        Scale factor for points (unused in matplotlib implementation, kept for
        compatibility/kwargs).
    color : color spec, optional
        Colour of points. Default is 'k' for 2D, 'r' for 3D.
    show : bool or None, optional
        If True, call `plt.show()`. If None, call `plt.show()` only if `ax` was None.
    **_kwargs
        Passed to `ax.scatter`.

    Returns
    -------
    ax : Axes
        The axes object.

    Examples
    --------
    >>> from eqsp.illustrations import project_point_set
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend('Agg')
    >>> points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    >>> ax = project_point_set(points, proj='eqarea')
    >>> len(ax.collections)  # doctest: +SKIP
    1
    """
    points = np.asarray(points)
    dim = points.shape[0] - 1
    if dim not in [2, 3]:
        raise ValueError("Points must be in R^3 (S^2) or R^4 (S^3)")

    if proj == "stereo":
        projector = x2stereo
    elif proj == "eqarea":
        projector = x2eqarea
    else:
        raise ValueError("proj must be 'stereo' or 'eqarea'")

    if dim == 2:
        t = projector(points)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
            ax.set_axis_off()
        if color is None:
            # Colour based on colatitude (mimic MATLAB)
            # Matlab uses r = pi - acos(z)
            z = points[2, :]
            r = np.pi - np.arccos(np.clip(z, -1.0, 1.0))
            cmap = plt.get_cmap("jet")
            c = cmap(r / np.pi)
        else:
            c = color
        ax.scatter(t[0, :], t[1, :], s=20, c=c, **_kwargs)

    elif dim == 3:
        raise NotImplementedError(
            "3D plotting for S^3 has been moved to eqsp.visualizations."
        )

    if show is True or (show is None and ax is None):
        if plt.get_backend() != "Agg":  # pragma: no cover
            plt.show()  # pragma: no cover
    return ax


def project_s2_partition(
    N,
    *,
    extra_offset=False,
    fontsize=16,
    title="long",
    proj="stereo",
    show_points=False,
    ax=None,
    show=None,
    **_kwargs,
):
    """
    Use projection to illustrate an EQ partition of S^2.

    Parameters
    ----------
    N : int
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.
    fontsize : int, optional
        Font size for title. Default 16.
    title : {'long', 'short', 'none'}, optional
        Title mode. Default 'long'.
    proj : {'stereo', 'eqarea'}, optional
        Projection type. Default 'stereo'.
    show_points : bool, optional
        Show center points of regions. Default False.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    show : bool or None, optional
        If True, call `plt.show()`. If None, call `plt.show()` only if `ax` was None.
    **_kwargs
        Passed to underlying plot functions.

    Returns
    -------
    ax : Axes
        The axes object.

    Examples
    --------
    >>> from eqsp.illustrations import project_s2_partition
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend('Agg')
    >>> ax = project_s2_partition(4, proj='eqarea', title='none', show_points=True)
    >>> ax.get_title()  # doctest: +SKIP
    ''
    """
    show_title = title != "none"

    if proj == "stereo":
        projector = x2stereo
    elif proj == "eqarea":
        projector = x2eqarea
    else:
        raise ValueError("proj must be 'stereo' or 'eqarea'")

    dim = 2
    R = eq_regions(dim, N, extra_offset)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_axis_off()

    # Ensure aspect is equal even if ax provided
    ax.set_aspect("equal")
    ax.set_axis_off()

    if proj == "eqarea":
        circ = plt.Circle((0, 0), np.sqrt(2), color="k", fill=False, linewidth=1)
        ax.add_patch(circ)

    for i in range(1, N):  # Draw all regions 1..N-1?
        region = R[:, :, i]
        t = region[:, 0]
        b = region[:, 1]

        tol = 1e-10
        if abs(b[0]) < tol:
            b[0] = 2 * np.pi
        pseudo = abs(t[0]) < tol and abs(b[0] - 2 * np.pi) < tol

        fidelity = 33
        h = np.linspace(0, 1, fidelity)

        # Color: Black boundaries
        color = "k"

        for k in range(dim):
            if pseudo and k >= 1:
                continue
            j = np.arange(dim)
            j = np.roll(j, -k)

            s_curve = np.zeros((dim, fidelity))
            idx_vary = j[0]
            idx_fixed = j[1:]

            s_curve[idx_vary, :] = t[idx_vary] + (b[idx_vary] - t[idx_vary]) * h
            for i_f in idx_fixed:
                s_curve[i_f, :] = t[i_f]

            x_curve = polar2cart(s_curve)
            p_curve = projector(x_curve)

            mask = np.isfinite(p_curve[0, :])
            ax.plot(p_curve[0, mask], p_curve[1, mask], color=color, linewidth=0.5)

    if show_points:
        points = eq_point_set(dim, N, extra_offset)
        project_point_set(points, ax=ax, proj=proj, color=None, **_kwargs)

    if show_title:
        title_text = f"EQ(2,{N}) {proj} projection"
        ax.set_title(title_text, fontsize=fontsize, color="k")

    if show is True or (show is None and ax is None):
        if plt.get_backend() != "Agg":  # pragma: no cover
            plt.show()  # pragma: no cover

    return ax



def illustrate_eq_algorithm(
    dim,
    N,
    *,
    extra_offset=False,
    fontsize=8,
    show_title=True,
    _long_title=False,
    show_points=True,
    proj="eqarea",
    show=True,
    **_kwargs,
):
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
    _long_title : bool, optional
        Use long title variant. Default False.
    proj : str, optional
        Projection type ('stereo' or 'eqarea'). Default 'eqarea'.
    show : bool, optional
        If True (default), call `plt.show()`.
    **_kwargs
        Passed to underlying plot functions.

    Returns
    -------
    None
    """
    # Consolidate options for easier passing
    opts = {
        "extra_offset": extra_offset,
        "fontsize": fontsize,
        "show_title": show_title,
        "long_title": _long_title,  # Pass the original value, not the prefixed name
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
    proj_opts["fontsize"] = 16
    # Map proj options to illustration functions
    proj_opts["proj"] = proj

    proj_opts["show_points"] = show_points

    # Map title mode
    if show_title:
        proj_opts["title"] = "long" if _long_title else "short"
    else:
        proj_opts["title"] = "none"

    # Remove options not accepted by project_s2_partition/ax.scatter
    proj_opts.pop("show_title", None)
    proj_opts.pop("long_title", None)  # Remove the original name

    if dim == 2:
        ax = plt.subplot(2, 2, 4)
        plt.axis("off")
        project_s2_partition(N, ax=ax, **proj_opts)
    elif dim == 3:
        _, m = eq_caps(dim, N)
        # m is n_regions (1D array)
        max_collar = min(4, m.size - 2)
        plt.subplot(2, 2, 4)
        plt.axis("off")
        for k in range(1, max_collar + 1):
            subn = 9 + 2 * k - ((k - 1) % 2)
            ax_sub = plt.subplot(4, 4, subn)
            plt.axis("off")
            opts_k = proj_opts.copy()
            opts_k["title"] = "none"
            project_s2_partition(int(m[k]), ax=ax_sub, **opts_k)

    if show and plt.get_backend() != "Agg":
        plt.show()  # pragma: no cover


def illustrate_steps_1_2(
    dim, N, *, fontsize=8, show_title=True, _long_title=False, **_kwargs
):
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
    plt.text(
        np.sin(mid) * 2.0 / 3.0,
        np.cos(mid) * 2.0 / 3.0,
        r"$\Delta_I$",
        fontsize=fontsize,
    )

    # Plot an arc to indicate angles
    theta = h * (c_polar + Delta_I)
    plt.plot(np.sin(theta) / 5.0, np.cos(theta) / 5.0, "b", linewidth=1)

    plt.text(
        -0.9,
        -0.1,
        r"$V(\theta_c) = V_R$" + "\n" + r"$= \sigma(S^{%d})/%d$" % (dim, N),
        fontsize=fontsize,
    )

    caption_angle = min(mid + 2.0 * Delta_I, np.pi - c_polar)
    plt.text(
        np.sin(caption_angle) / 3.0,
        np.cos(caption_angle) / 3.0,
        r"$\Delta_I = V_R^{1/%d}$" % dim,
        fontsize=fontsize,
    )

    if show_title:
        title_str = f"EQ({dim},{N}) Steps 1 to 2"
        plt.title(title_str, fontsize=fontsize, color="k")


def illustrate_steps_3_5(
    dim, N, *, fontsize=8, show_title=True, _long_title=False, **_kwargs
):
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
    plt.text(
        np.sin(c_polar) * 1.1,
        np.cos(c_polar) * 1.1,
        r"$\theta_{F,1}$",
        fontsize=fontsize,
    )
    plt.plot(np.sin(c_polar) * h, np.cos(c_polar) * h, "b", linewidth=2)
    plt.plot(np.sin(c_polar) * k, -np.cos(c_polar) * j, "r", linewidth=2)

    plt.plot(np.zeros_like(j), k, "b", linewidth=1)

    for collar_n in range(1, n_collars + 1):
        zone_n = collar_n
        theta = s_cap[zone_n]
        plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)
        theta_str = r"$\theta_{F,%d}$" % (collar_n + 1)
        plt.text(np.sin(theta) * 1.1, np.cos(theta) * 1.1, theta_str, fontsize=fontsize)

        theta_p = s_cap[collar_n - 1]

        if collar_n > 1:
            plt.plot(np.sin(theta_p) * k, np.cos(theta_p) * j, "r", linewidth=2)

        arc = theta_p + (theta - theta_p) * h
        plt.plot(np.sin(arc) / 5.0, np.cos(arc) / 5.0, "b", linewidth=1)
        mid = (theta_p + theta) / 2.0
        plt.text(np.sin(mid) / 2.0, np.cos(mid) / 2.0, r"$\Delta_F$", fontsize=fontsize)
        y_str = r"$y_{%d} = %3.1f...$" % (collar_n, r_regions[zone_n])
        plt.text(
            -np.sin(mid) + 1.0 / 20.0,
            np.cos(mid) + (mid - np.pi) / 30.0,
            y_str,
            fontsize=fontsize,
        )
    if show_title:
        title_str = f"EQ({dim},{N}) Steps 3 to 5"
        plt.title(title_str, fontsize=fontsize, color="k")


def illustrate_steps_6_7(
    dim, N, *, fontsize=8, show_title=True, _long_title=False, **_kwargs
):
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
    plt.text(
        np.sin(c_polar) * 1.1, np.cos(c_polar) * 1.1, r"$\theta_1$", fontsize=fontsize
    )
    plt.plot(np.sin(c_polar) * h, np.cos(c_polar) * h, "b", linewidth=2)
    plt.plot(np.sin(c_polar) * k, -np.cos(c_polar) * j, "r", linewidth=2)

    plt.plot(np.zeros_like(j), k, "b", linewidth=1)

    for collar_n in range(1, n_collars + 1):
        zone_n = collar_n
        theta = s_cap[zone_n]
        plt.plot(np.sin(theta) * h, np.cos(theta) * h, "b", linewidth=2)
        theta_str = r"$\theta_{%d}$" % (collar_n + 1)
        plt.text(np.sin(theta) * 1.1, np.cos(theta) * 1.1, theta_str, fontsize=fontsize)

        theta_p = s_cap[collar_n - 1]

        if collar_n > 1:
            plt.plot(np.sin(theta_p) * k, np.cos(theta_p) * j, "r", linewidth=2)

        arc = theta_p + (theta - theta_p) * h
        plt.plot(np.sin(arc) / 5.0, np.cos(arc) / 5.0, "b", linewidth=1)
        mid = (theta_p + theta) / 2.0
        Delta_str = r"$\Delta_{%i}$" % collar_n
        plt.text(np.sin(mid) / 2.0, np.cos(mid) / 2.0, Delta_str, fontsize=fontsize)
        m_str = r"$m_{%d} =%3.0f$" % (collar_n, n_regions[zone_n])
        plt.text(
            -np.sin(mid) + 1.0 / 20.0,
            np.cos(mid) + (mid - np.pi) / 30.0,
            m_str,
            fontsize=fontsize,
        )
    if show_title:
        title_str = f"EQ({dim},{N}) Steps 6 to 7"
        plt.title(title_str, fontsize=fontsize, color="k")
