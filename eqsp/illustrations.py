"""
EQSP Illustrations module.
Ported from eq_sphere_partitions/eq_illustrations.

Copyright 2025 Paul Leopardi.
For licensing, see COPYING.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

# Ensure 3D projection is available
from mpl_toolkits.mplot3d import Axes3D

from .utilities import polar2cart, area_of_cap, volume_of_ball, area_of_sphere
from .partitions import eq_point_set, eq_regions, eq_caps
from .partition_options import partition_options


def x2stereo(x):
    """
    Stereographic projection of Euclidean points.
    
    Parameters
    ----------
    x : ndarray
        Points in R^{dim+1}, shape (dim+1, N).
        
    Returns
    -------
    result : ndarray
        Projected points in R^dim, shape (dim, N).
    """
    x = np.asarray(x)
    dim = x.shape[0] - 1
    
    last = x[dim, :]
    mask = np.isclose(last, 1.0)
    
    scale = np.ones(x.shape[1])
    scale[~mask] = 1.0 - last[~mask]
    
    with np.errstate(divide='ignore'): 
        result = x[:dim, :] / scale
        
    result[:, mask] = np.nan
    return result


def x2eqarea(x):
    """
    Equal area projection of Euclidean points.
    
    Parameters
    ----------
    x : ndarray
        Points in R^{dim+1}, shape (dim+1, N).
        
    Returns
    -------
    result : ndarray
        Projected points in R^dim, shape (dim, N).
    """
    x = np.asarray(x)
    dim = x.shape[0] - 1
    last = x[dim, :]
    
    theta = np.arccos(np.clip(-last, -1.0, 1.0))
    a_cap = area_of_cap(dim, theta)
    v_ball = volume_of_ball(dim)
    r = (a_cap / v_ball) ** (1.0 / dim)
    
    sin_theta = np.sin(theta)
    mask = np.isclose(sin_theta, 0.0)
    
    scale = np.zeros_like(theta)
    scale[~mask] = r[~mask] / sin_theta[~mask]
    
    result = np.zeros((dim, x.shape[1]))
    result[:, ~mask] = x[:dim, ~mask] * scale[~mask]
    return result


def fatcurve(c, r=0.01, m=8):
    """
    Create a parameterized cylindrical surface at radius r from curve c.
    """
    c = np.asarray(c)
    dim, n = c.shape
    if dim != 3:
        raise ValueError(f"fatcurve called with dim == {dim} but dim must be 3.")
    
    h = np.linspace(0, 1, m + 1)
    phi = h * 2 * np.pi
    
    X = np.zeros((n, m + 1))
    Y = np.zeros((n, m + 1))
    Z = np.zeros((n, m + 1))
    
    circ = None
    v = None
    
    for k in range(n - 1):
        u = c[:, k+1] - c[:, k]
        if np.linalg.norm(u) < 1e-10:
             if circ is not None:
                XYZ = c[:, k][:, np.newaxis] + r * circ
                X[k, :] = XYZ[0, :]
                Y[k, :] = XYZ[1, :]
                Z[k, :] = XYZ[2, :]
             continue

        M = null_space(u.reshape(1, 3))
        if M.shape[1] != 2:
             v_temp = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
             v_curr = v_temp - u * np.dot(u, v_temp) / np.dot(u, u)
             v_curr = v_curr / np.linalg.norm(v_curr)
             w_curr = np.cross(u, v_curr)
             w_curr = w_curr / np.linalg.norm(w_curr)
        else:
            v_curr = M[:, 0]
            w_curr = np.cross(u, v_curr)
            w_curr = w_curr / np.linalg.norm(w_curr)
            
        circ_curr = np.outer(v_curr, np.cos(phi)) + np.outer(w_curr, np.sin(phi))
        circ = circ_curr
        v = v_curr
        
        XYZ = c[:, k][:, np.newaxis] + r * circ
        X[k, :] = XYZ[0, :]
        Y[k, :] = XYZ[1, :]
        Z[k, :] = XYZ[2, :]
        
    if circ is not None:
        XYZ = c[:, n-1][:, np.newaxis] + r * circ
        X[n-1, :] = XYZ[0, :]
        Y[n-1, :] = XYZ[1, :]
        Z[n-1, :] = XYZ[2, :]
        
    return X, Y, Z


def show_s2_sphere(ax=None, alpha=0.3):
    """
    Illustrate the unit sphere S^2.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='b', alpha=alpha, rstride=5, cstride=5, linewidth=0)
    try: ax.set_box_aspect([1,1,1])
    except: pass
    return ax


def show_r3_point_set(points, ax=None, show_sphere=False, **kwargs):
    """
    3D illustration of a point set.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    if show_sphere:
        show_s2_sphere(ax)
        
    ax.scatter(points[0, :], points[1, :], points[2, :], s=20, c='r', **kwargs)
    return ax


def show_s2_region(region, N, ax=None, fidelity=21):
    """
    Illustrate a region of S^2.
    """
    if ax is None:
        ax = plt.gca()
        
    tol = np.finfo(float).eps * 32
    dim = region.shape[0]
    t = region[:, 0]
    b = region[:, 1]
    
    if abs(b[0]) < tol: b[0] = 2 * np.pi
    pseudo = (abs(t[0]) < tol and abs(b[0] - 2 * np.pi) < tol)
        
    h = np.linspace(0, 1, fidelity)
    r = np.sqrt(1.0 / N) / 12.0
    
    for k in range(dim):
        if pseudo and k >= 1: continue
        j = np.arange(dim)
        j = np.roll(j, -k)
        
        s_curve = np.zeros((dim, fidelity))
        idx_vary = j[0]
        idx_fixed = j[1:]
        
        s_curve[idx_vary, :] = t[idx_vary] + (b[idx_vary] - t[idx_vary]) * h
        for i_f in idx_fixed: s_curve[i_f, :] = t[i_f]
            
        x_curve = polar2cart(s_curve)
        X, Y, Z = fatcurve(x_curve, r)
        ax.plot_surface(X, Y, Z, color='k', alpha=1.0, linewidth=0, shade=True)


def show_s2_partition(N, *args, **kwargs):
    """
    3D illustration of an EQ partition of S^2 into N regions.
    """
    flat_args = list(args)
    for k, v in kwargs.items():
        flat_args.extend([k, v])
        
    pdefault = {'extra_offset': False}
    popt = partition_options(pdefault, *flat_args)
    
    show_sphere = kwargs.get('show_sphere', True)
    show_points = kwargs.get('show_points', True)
    show_title = kwargs.get('show_title', True)
    fontsize = kwargs.get('fontsize', 16)
    
    dim = 2
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    try: ax.set_box_aspect([1,1,1])
    except: pass
    
    if show_title:
        title_text = f"Recursive zonal equal area partition of S^2\ninto {N} regions."
        ax.set_title(title_text, fontsize=fontsize)
        
    if show_sphere:
        show_s2_sphere(ax)
        
    R = eq_regions(dim, N, popt['extra_offset'])
    for i in range(N-1, 0, -1):
        show_s2_region(R[:, :, i], N, ax=ax)
        
    if show_points:
        points = eq_point_set(dim, N, popt['extra_offset'])
        show_r3_point_set(points, ax=ax, show_sphere=False)
        
    plt.show()
    return ax


def project_point_set(points, ax=None, proj='stereo', **kwargs):
    """
    Use projection to illustrate a point set of S^2 or S^3.
    """
    points = np.asarray(points)
    dim = points.shape[0] - 1
    if dim not in [2, 3]:
        raise ValueError("Points must be in R^3 (S^2) or R^4 (S^3)")
        
    if proj == 'stereo':
        projector = x2stereo
    elif proj == 'eqarea':
        projector = x2eqarea
    else:
        raise ValueError("proj must be 'stereo' or 'eqarea'")
        
    if dim == 2:
        t = projector(points)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.set_axis_off()
        ax.scatter(t[0, :], t[1, :], s=20, c='k', **kwargs)
        
    elif dim == 3:
        t = projector(points)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(t[0, :], t[1, :], t[2, :], s=20, c='r', **kwargs)
        
    return ax


def project_s2_partition(N, *args, **kwargs):
    """
    Use projection to illustrate an EQ partition of S^2.
    """
    flat_args = list(args)
    for k, v in kwargs.items():
        flat_args.extend([k, v])
        
    pdefault = {'extra_offset': False}
    popt = partition_options(pdefault, *flat_args)
    
    fontsize = kwargs.get('fontsize', 16)
    show_title = kwargs.get('show_title', True)
    proj = kwargs.get('proj', 'stereo')
    
    if proj == 'stereo':
        projector = x2stereo
    elif proj == 'eqarea':
        projector = x2eqarea
    else:
        raise ValueError("proj must be 'stereo' or 'eqarea'")

    dim = 2
    R = eq_regions(dim, N, popt['extra_offset'])
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    if proj == 'eqarea':
        circ = plt.Circle((0, 0), np.sqrt(2), color='k', fill=False, linewidth=1)
        ax.add_patch(circ)
        
    for i in range(1, N): # Draw all regions 1..N-1?
        region = R[:, :, i]
        t = region[:, 0]
        b = region[:, 1]
        
        tol = 1e-10
        if abs(b[0]) < tol: b[0] = 2 * np.pi
        pseudo = (abs(t[0]) < tol and abs(b[0] - 2*np.pi) < tol)
        
        fidelity = 33
        h = np.linspace(0, 1, fidelity)
        
        # Color based on colatitude t[1]
        # t[1] ranges from 0 to pi.
        # Mimic Matlab's project_s3_partition which uses t(dim) for color data and jet colormap.
        cmap = plt.get_cmap('jet')
        c_val = t[1] / np.pi
        color = cmap(c_val)

        for k in range(dim):
             if pseudo and k >= 1: continue
             j = np.arange(dim)
             j = np.roll(j, -k)
             
             s_curve = np.zeros((dim, fidelity))
             idx_vary = j[0]
             idx_fixed = j[1:]
             
             s_curve[idx_vary, :] = t[idx_vary] + (b[idx_vary] - t[idx_vary]) * h
             for i_f in idx_fixed: s_curve[i_f, :] = t[i_f]
             
             x_curve = polar2cart(s_curve)
             p_curve = projector(x_curve)
             
             mask = np.isfinite(p_curve[0, :])
             ax.plot(p_curve[0, mask], p_curve[1, mask], color=color, linewidth=0.5)

    if show_title:
        title_text = f"EQ(2,{N}) {proj} projection"
        ax.set_title(title_text, fontsize=fontsize)
        
    plt.show()
    return ax


def project_s3_partition(N, *args, **kwargs):
    """
    Use projection to illustrate an EQ partition of S^3.
    """
    flat_args = list(args)
    for k, v in kwargs.items():
        flat_args.extend([k, v])
        
    pdefault = {'extra_offset': False}
    popt = partition_options(pdefault, *flat_args)
    
    proj = kwargs.get('proj', 'stereo')
    if proj == 'stereo': projector = x2stereo
    else: projector = x2eqarea
        
    dim = 3
    # Note: Extra offsets for Dim 3 not fully ported (needs rotation matrices return from eq_regions)
    R = eq_regions(dim, N, popt['extra_offset'])
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    try: ax.set_box_aspect([1,1,1])
    except: pass
    
    for i in range(1, N):
         region = R[:, :, i]
         dim_reg = 3
         t = region[:, 0]
         b = region[:, 1]
         if abs(b[0]) < 1e-10: b[0] = 2*np.pi
         pseudo = (abs(t[0]) < 1e-10 and abs(b[0] - 2*np.pi) < 1e-10)
         
         for k in range(dim_reg):
             if pseudo and k >= 2: continue
             j = np.arange(dim_reg)
             j = np.roll(j, -k)
             
             h_grid = np.linspace(0, 1, 10)
             H1, H2 = np.meshgrid(h_grid, h_grid)
             
             s_face = np.zeros((dim_reg, 10, 10))
             idx_vary1, idx_vary2, idx_fixed = j[0], j[1], j[2]
             
             s_face[idx_vary1, :, :] = t[idx_vary1] + (b[idx_vary1] - t[idx_vary1]) * H1
             s_face[idx_vary2, :, :] = t[idx_vary2] + (b[idx_vary2] - t[idx_vary2]) * H2
             s_face[idx_fixed, :, :] = t[idx_fixed]
             
             s_flat = s_face.reshape(dim_reg, -1)
             x_flat = polar2cart(s_flat)
             p_flat = projector(x_flat)
             
             PX = p_flat[0, :].reshape(10, 10)
             PY = p_flat[1, :].reshape(10, 10)
             PZ = p_flat[2, :].reshape(10, 10)
             
             if np.any(np.isnan(PX)): continue
             
             # Mimic Matlab: color based on t[2] (jet), alpha = (t[2]/pi)/2
             cmap = plt.get_cmap('jet')
             # t[2] is effectively polar angle in [0, pi]
             # Map t[2] to [0, 1] for colormap
             c_val = t[2] / np.pi
             color = cmap(c_val)
             alpha = (t[2] / np.pi) / 2.0
             
             ax.plot_surface(PX, PY, PZ, alpha=alpha, color=color)
             
    plt.show()
    return ax
