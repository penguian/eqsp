"""
PyEQSP Visualizations module.
3D visualizations using PyVista.

Copyright 2026 Paul Leopardi.
For licensing, see LICENSE.
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
    import pyvista as pv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyVista is not installed. "
        "Please install it with: pip install 'pyeqsp[pyvista]'"
    ) from exc

PROJ_NAME = {"eqarea": "equal area", "stereo": "stereographic"}


def _get_plotter(plotter=None):
    """
    Return an active PyVista Plotter instance or create a new one.
    """
    if plotter is not None:
        return plotter
    pl = pv.Plotter(lighting="light_kit")
    pl.set_background("white")
    return pl


def show_s2_sphere(opacity=0.95, color=(0.1, 0.85, 0.3), plotter=None):
    """
    Illustrate the unit sphere S^2.
    """
    pl = _get_plotter(plotter)
    sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
    pl.add_mesh(sphere, color=color, opacity=opacity, ambient=0.5, smooth_shading=True)
    return pl


def show_r3_point_set(
    points,
    *,
    show_sphere=False,
    scale_factor=None,
    color=(1.0, 0.0, 0.0),
    opacity=1.0,
    save_file=None,
    plotter=None,
    **kwargs,
):
    """
    3D illustration of a point set.
    """
    pl = _get_plotter(plotter)
    if show_sphere:
        show_s2_sphere(plotter=pl)

    if scale_factor is None:
        num_points = points.shape[1]
        scale_factor = 0.4 / np.sqrt(num_points)

    poly = pv.PolyData(points.T)
    glyphs = poly.glyph(geom=pv.Sphere(radius=scale_factor), scale=False, orient=False)
    glyphs.active_scalars_name = None
    pl.add_mesh(glyphs, color=color, opacity=opacity, ambient=0.5, **kwargs)

    if save_file:
        pl.screenshot(save_file)
    return pl


def show_s2_region(region, N, fidelity=32, opacity=1.0, plotter=None):
    """
    Illustrate a region of S^2.
    """
    pl = _get_plotter(plotter)
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

        poly = pv.PolyData(x_curve.T)
        lines = np.column_stack(
            [
                np.full(fidelity - 1, 2, dtype=int),
                np.arange(fidelity - 1),
                np.arange(1, fidelity),
            ]
        )
        poly.lines = lines
        tube = poly.tube(radius=r)
        pl.add_mesh(tube, color=(0, 0.4, 0.9), opacity=opacity, ambient=0.5)
    return pl


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
    plotter=None,
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
    show : bool, optional
        Display rendering window. Default True.
    save_file : str, optional
        Filename to save screenshot. Default None.
    plotter : pv.Plotter, optional
        Existing PyVista plotter instance.

    Examples
    --------
    >>> from eqsp.visualizations import show_s2_partition
    >>> import pyvista as pv
    >>> pv.OFF_SCREEN = True
    >>> try:
    ...     _ = show_s2_partition(4, title='short', show_points=False, show=False)
    ...     print("Success")
    ... except ImportError:
    ...     print("PyVista not installed")
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

    pl = _get_plotter(plotter)

    if show_sphere:
        show_s2_sphere(opacity=0.95, plotter=pl)

    R = eq_regions(2, N, extra_offset)
    for i in range(N - 1, 0, -1):
        show_s2_region(R[:, :, i], N, plotter=pl)

    if show_points:
        points = eq_point_set(2, N, extra_offset)
        show_r3_point_set(points, show_sphere=False, plotter=pl)

    if show_title:
        # Convert title_pos to window coordinates (pixel offsets from bottom-left)
        win_x = int(title_pos[0] * pl.window_size[0])
        win_y = int(title_pos[1] * pl.window_size[1])
        pl.add_text(title_text, position=(win_x, win_y), font_size=12, color="black")

    if save_file:
        pl.screenshot(save_file)

    if show and not pv.OFF_SCREEN:
        pl.show()
    return pl


def project_point_set(
    points,
    *,
    proj="stereo",
    scale_factor=None,
    color=(1.0, 0.0, 0.0),
    show=True,
    save_file=None,
    plotter=None,
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
        Scale factor for points. Default None (dynamically calculated as 0.4 / sqrt(N)).
    color : tuple, optional
        Colour of points in RGB format (0 to 1). Default (1, 0, 0).
    show : bool, optional
        Display rendering window. Default True.
    save_file : str, optional
        Filename to save screenshot. Default None.
    plotter : pv.Plotter, optional
        Existing PyVista plotter instance.

    Examples
    --------
    >>> from eqsp.visualizations import project_point_set
    >>> import numpy as np
    >>> import pyvista as pv
    >>> pv.OFF_SCREEN = True
    >>> points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    >>> try:
    ...     _ = project_point_set(points, proj='eqarea', show=False)
    ...     print("Success")
    ... except ImportError:
    ...     print("PyVista not installed")
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

    if scale_factor is None:
        num_points = points.shape[1]
        scale_factor = 0.4 / np.sqrt(num_points)

    pl = _get_plotter(plotter)
    t = projector(points)

    if dim == 2:
        proj_pts = np.vstack([t[0, :], t[1, :], np.zeros_like(t[0, :])])
    else:
        proj_pts = t[:3, :]

    poly = pv.PolyData(proj_pts.T)
    glyphs = poly.glyph(geom=pv.Sphere(radius=scale_factor), scale=False, orient=False)
    glyphs.active_scalars_name = None
    pl.add_mesh(glyphs, color=color, ambient=0.5, **kwargs)

    if save_file:
        pl.screenshot(save_file)

    if show and not pv.OFF_SCREEN:
        pl.show()
    return pl


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
    plotter=None,
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
    show : bool, optional
        Display rendering window. Default True.
    save_file : str, optional
        Filename to save screenshot. Default None.
    plotter : pv.Plotter, optional
        Existing PyVista plotter instance.

    Examples
    --------
    >>> from eqsp.visualizations import project_s3_partition
    >>> import pyvista as pv
    >>> pv.OFF_SCREEN = True
    >>> try:
    ...     _ = project_s3_partition(
    ...         4, proj='stereo', show_points=True, show_surfaces=False, show=False
    ...     )
    ...     print("Success")
    ... except ImportError:
    ...     print("PyVista not installed")
    Success
    """
    if proj == "stereo":
        projector = x2stereo
    elif proj == "eqarea":
        projector = x2eqarea
    else:
        raise ValueError("proj must be 'stereo' or 'eqarea'")

    title_text = None
    if title == "none":
        show_title = False
    else:
        show_title = True
        if title == "long":
            title_text = f"EQ(3,{N}) {PROJ_NAME.get(proj, proj)} projection"
        elif title == "short":
            title_text = f"EQ(3,{N})"
        else:
            title_text = title

    pl = _get_plotter(plotter)
    dim = 3

    if show_surfaces:
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

                if np.any(np.isnan(PX)):
                    continue  # pragma: no cover

                cmap = plt.get_cmap("jet")
                c_val = t[2] / np.pi
                rgba = cmap(c_val)
                color = rgba[:3]
                opacity = (t[2] / np.pi) / 2.0

                grid = pv.StructuredGrid(PX, PY, PZ)
                pl.add_mesh(grid, opacity=opacity, color=color)

    if show_points:
        points = eq_point_set(dim, N, extra_offset)
        project_point_set(
            points,
            proj=proj,
            color=(1, 0, 0),
            scale_factor=0.1,
            show=False,
            plotter=pl,
            **kwargs,
        )

    if show_title:
        win_x = int(0.2 * pl.window_size[0])
        win_y = int(0.9 * pl.window_size[1])
        pl.add_text(title_text, position=(win_x, win_y), font_size=12, color="black")

    if save_file:
        pl.screenshot(save_file)

    if show and not pv.OFF_SCREEN:
        pl.show()
    return pl
