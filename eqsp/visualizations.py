"""
PyEQSP Visualizations module.
3D visualizations using Mayavi.

Copyright 2026 Paul Leopardi.
For licensing, see COPYING.
"""

import matplotlib.pyplot as plt
import numpy as np

from .partitions import eq_point_set, eq_regions
from .utilities import (
    TAU,
    polar2cart,
    x2eqarea,
    x2stereo,
)

try:
    from mayavi import mlab
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Mayavi is not installed. Please install it with: pip install 'eqsp[mayavi]'"
    ) from exc

PROJ_NAME = {"eqarea": "equal area", "stereo": "stereographic"}


def show_s2_sphere(opacity=0.95, color=(0, 1, 0)):
    """
    Illustrate the unit sphere S^2.
    """
    u, v = np.mgrid[0:TAU:50j, 0 : np.pi : 50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    mlab.mesh(x, y, z, color=color, opacity=opacity)


def show_r3_point_set(
    points,
    show_sphere=False,
    scale_factor=0.1,
    save_file=None,
    **kwargs,
):
    """
    3D illustration of a point set.
    """
    if show_sphere:
        show_s2_sphere()

    # Mayavi points3d expects x, y, z, s (scalar) - s is optional
    mlab.points3d(
        points[0, :],
        points[1, :],
        points[2, :],
        scale_factor=scale_factor,
        color=(1, 0, 0),
        **kwargs,
    )

    if save_file:
        mlab.savefig(save_file)


def show_s2_region(region, N, fidelity=32):
    """
    Illustrate a region of S^2.
    """
    # pylint: disable=no-member
    tol = np.finfo(float).eps * 32
    dim = region.shape[0]
    t = region[:, 0]
    b = region[:, 1]

    if abs(b[0]) < tol:
        b[0] = TAU  # pragma: no cover
    pseudo = abs(t[0]) < tol and abs(b[0] - TAU) < tol

    h = np.linspace(0, 1, fidelity)
    r = np.sqrt(1.0 / N) / 12.0

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

        # Mayavi plot3d for tube plotting
        mlab.plot3d(x_curve[0], x_curve[1], x_curve[2], tube_radius=r, color=(0, 0, 1))


def show_s2_partition(
    N,
    *,
    extra_offset=False,
    show_points=True,
    show_sphere=True,
    title="long",
    title_pos=(0.2, 0.85),
    show=True,
    save_file=None,
    **_kwargs,
):
    """
    3D illustration of an EQ partition of S^2 into N regions.

    Parameters
    ----------
    N : int
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.
    show_points : bool, optional
        Show centre points. Default True.
    show_sphere : bool, optional
        Show unit sphere. Default True.
    title : str, optional
        Title text. Special values: 'long', 'short', 'none'.
        'long' uses a default multi-line description.
        'short' uses 'EQ(2, N)'.
        'none' shows no title.
        Any other string is used as the title text.
    title_pos : tuple, optional
        (x, y) position of the title in figure coordinates (0 to 1).
        Default is (0.2, 0.85).
    **kwargs
        Passed to Mayavi functions.

    Examples
    --------
    >>> from eqsp.visualizations import show_s2_partition
    >>> from mayavi import mlab
    >>> mlab.options.offscreen = True
    >>> try:
    ...     show_s2_partition(4, title='short', show_points=False)
    ...     print("Success") # Crude check as Mayavi is hard to doctest
    ... except ImportError:
    ...     print("Mayavi not installed")
    Success
    """
    title_text = None
    if title == "none":
        show_title = False
    else:
        show_title = True
        if title == "long":
            title_text = (
                f"Recursive zonal equal area partition of S^2\ninto {N} regions."
            )
        elif title == "short":
            title_text = f"EQ(2, {N})"
        else:
            title_text = title

    # Set default figure size if none exists
    if mlab.get_engine().current_scene is None:
        mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

    if show_sphere:
        show_s2_sphere(opacity=0.95)

    R = eq_regions(2, N, extra_offset)
    for i in range(N - 1, 0, -1):
        show_s2_region(R[:, :, i], N)

    if show_points:
        points = eq_point_set(2, N, extra_offset)
        show_r3_point_set(points, show_sphere=False)

    if show_title:
        # Use mlab.text for precise control over size and position
        mlab.text(title_pos[0], title_pos[1], title_text, width=0.6, color=(0, 0, 0))

    if save_file:
        mlab.savefig(save_file)

    if show:
        mlab.show()


def project_point_set(
    points,
    proj="stereo",
    scale_factor=0.1,
    color=(1, 0, 0),
    show=True,
    save_file=None,
    **kwargs,
):
    """
    Use projection to illustrate a point set of S^2 or S^3.

    Parameters
    ----------
    points : ndarray
        Array of shape (dim+1, N) containing centre points of each region in
        Cartesian coordinates.
    proj : {'stereo', 'eqarea'}, optional
        Projection type. Default 'stereo'.
    scale_factor : float, optional
        Scale factor for points. Default 0.1.
    color : tuple, optional
        Colour of points in RGB format (0 to 1). Default (1, 0, 0).
    **kwargs
        Passed to Mayavi plotting functions.

    Examples
    --------
    >>> from eqsp.visualizations import project_point_set
    >>> import numpy as np
    >>> from mayavi import mlab
    >>> mlab.options.offscreen = True
    >>> points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    >>> try:
    ...     project_point_set(points, proj='eqarea')
    ...     print("Success")
    ... except ImportError:
    ...     print("Mayavi not installed")
    Success
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

    # Set default figure size if none exists
    if mlab.get_engine().current_scene is None:
        mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

    t = projector(points)

    # Mayavi points3d
    # scale_factor and color are explicit arguments now, but kwargs can override?
    # Actually explicit args take precedence in this implementation, assume user passes
    # them.

    if dim == 2:
        # Project to z=0 for S^2 -> R^2
        mlab.points3d(
            t[0, :],
            t[1, :],
            np.zeros_like(t[0, :]),
            scale_factor=scale_factor,
            color=color,
            **kwargs,
        )
    elif dim == 3:
        # S^3 -> R^3
        mlab.points3d(
            t[0, :], t[1, :], t[2, :], scale_factor=scale_factor, color=color, **kwargs
        )

    if save_file:
        mlab.savefig(save_file)

    if show:
        mlab.show()


def project_s3_partition(
    N,
    *,
    extra_offset=False,
    title="long",
    proj="stereo",
    show_points=True,
    show_surfaces=True,
    show=True,
    save_file=None,
    **kwargs,
):
    """
    Use projection to illustrate an EQ partition of S^3.

    Parameters
    ----------
    N : int
        Number of regions.
    extra_offset : bool, optional
        Use extra offsets. Default False.
    title : {'long', 'short', 'none'}, optional
        Title format. Default 'long'.
    proj : {'stereo', 'eqarea'}, optional
        Projection type. Default 'stereo'.
    show_points : bool, optional
        Show center points. Default True.
    show_surfaces : bool, optional
        Show region surfaces. Default True.
    **kwargs
        Passed to Mayavi plotting functions.

    Examples
    --------
    >>> from eqsp.visualizations import project_s3_partition
    >>> from mayavi import mlab
    >>> mlab.options.offscreen = True
    >>> try:
    ...     project_s3_partition(
    ...         4, proj='stereo', show_points=True, show_surfaces=False
    ...     )
    ...     print("Success")
    ... except ImportError:
    ...     print("Mayavi not installed")
    Success
    """
    if proj == "stereo":
        projector = x2stereo
    elif proj == "eqarea":
        projector = x2eqarea
    else:
        raise ValueError("proj must be 'stereo' or 'eqarea'")

    show_title = title != "none"

    # Set default figure size if none exists
    if mlab.get_engine().current_scene is None:
        mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

    dim = 3

    if show_surfaces:
        # Note: Extra offsets for Dim 3 not fully ported
        # (needs rotation matrices return from eq_regions)
        R = eq_regions(dim, N, extra_offset)

        for i in range(1, N):
            region = R[:, :, i]
            dim_reg = 3
            t = region[:, 0]
            b = region[:, 1]
            if abs(b[0]) < 1e-10:
                b[0] = TAU  # pragma: no cover
            pseudo = abs(t[0]) < 1e-10 and abs(b[0] - TAU) < 1e-10

            for k in range(dim_reg):
                if pseudo and k >= 2:
                    continue
                j = np.arange(dim_reg)
                j = np.roll(j, -k)

                h_grid = np.linspace(0, 1, 10)
                H1, H2 = np.meshgrid(h_grid, h_grid)

                s_face = np.zeros((dim_reg, 10, 10))
                idx_vary1, idx_vary2, idx_fixed = j[0], j[1], j[2]

                s_face[idx_vary1, :, :] = (
                    t[idx_vary1] + (b[idx_vary1] - t[idx_vary1]) * H1
                )
                s_face[idx_vary2, :, :] = (
                    t[idx_vary2] + (b[idx_vary2] - t[idx_vary2]) * H2
                )
                s_face[idx_fixed, :, :] = t[idx_fixed]

                s_flat = s_face.reshape(dim_reg, -1)
                x_flat = polar2cart(s_flat)
                p_flat = projector(x_flat)

                PX = p_flat[0, :].reshape(10, 10)
                PY = p_flat[1, :].reshape(10, 10)
                PZ = p_flat[2, :].reshape(10, 10)

                # Check for NaNs (e.g. projection to infinity)
                if np.any(np.isnan(PX)):
                    continue  # pragma: no cover

                # Mimic Matlab: color based on t[2] (jet), opacity = (t[2]/pi)/2
                cmap = plt.get_cmap("jet")
                c_val = t[2] / np.pi
                rgba = cmap(c_val)
                color = rgba[:3]
                opacity = (t[2] / np.pi) / 2.0

                mlab.mesh(PX, PY, PZ, opacity=opacity, color=color)

    if show_points:
        points = eq_point_set(dim, N, extra_offset)
        project_point_set(
            points, proj=proj, color=(1, 0, 0), scale_factor=0.1, show=False, **kwargs
        )

    if show_title:
        title_text = f"EQ(3,{N}) {PROJ_NAME.get(proj, proj)} projection"
        # width=0.6 gives a reasonable size for this longer string
        # x = 0.5 - 0.6/2 = 0.2, y = 0.9
        mlab.text(0.2, 0.9, title_text, width=0.6, color=(0, 0, 0))

    if save_file:
        mlab.savefig(save_file)

    if show:
        mlab.show()
