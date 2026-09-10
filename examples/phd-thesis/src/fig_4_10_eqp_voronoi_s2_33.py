"""
Figure 4.10 (3D): EQ code EQP(2, 33), Voronoi cells on the sphere, and EQ(2, 33).

Computes the spherical Voronoi diagram of EQP(2, 33) directly on S^2 using
scipy.spatial.SphericalVoronoi, then draws each Voronoi edge as a great circle
arc via spherical linear interpolation (SLERP). This correctly represents Voronoi
cell edges, which are great circle arcs, not straight lines.

Requires PyVista.
"""

# pylint: disable=wrong-import-position,import-error

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import SphericalVoronoi

import eqsp
from eqsp.visualizations import show_r3_point_set, show_s2_partition


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    N = 33
    dim = 2
    SAMPLES = 80  # points per great circle arc
    TUBE_R = np.sqrt(1.0 / N) / 12.0  # matches tube radius used by show_s2_region
    pv.OFF_SCREEN = True
    # ---------------------------------------------------------------
    # Step 1: Get EQP(2, 33) code points on the unit sphere.
    # ---------------------------------------------------------------
    points_3d = eqsp.eq_point_set(dim, N)  # shape (3, N)
    points_for_svd = points_3d.T  # SphericalVoronoi wants (N, 3)
    # ---------------------------------------------------------------
    # Step 2: Compute the spherical Voronoi diagram directly on S^2.
    # ---------------------------------------------------------------
    svd = SphericalVoronoi(points_for_svd, radius=1.0, center=np.zeros(3))
    svd.sort_vertices_of_regions()
    # ---------------------------------------------------------------
    # Step 3: Collect unique Voronoi edges (pairs of vertex indices).
    # Each consecutive pair of vertices in a region shares an edge.
    # ---------------------------------------------------------------
    edges = set()
    for region in svd.regions:
        n = len(region)
        for i in range(n):
            a = region[i]
            b = region[(i + 1) % n]
            edges.add((min(a, b), max(a, b)))

    # ---------------------------------------------------------------
    # Step 4: SLERP helper — great circle arc from point a to point b.
    # ---------------------------------------------------------------
    def great_circle_arc(pa, pb, n=SAMPLES):
        """Spherical linear interpolation from pa to pb on the unit sphere."""
        omega = np.arccos(np.clip(np.dot(pa, pb), -1.0, 1.0))
        if omega < 1e-10:
            return None
        t = np.linspace(0, 1, n)
        arc = (
            np.outer(np.sin((1 - t) * omega), pa) + np.outer(np.sin(t * omega), pb)
        ) / np.sin(omega)
        return arc

    # ---------------------------------------------------------------
    # Step 5: Set up PyVista scene with EQ partition regions (blue) and sphere.
    # ---------------------------------------------------------------
    pl = show_s2_partition(
        N,
        show_sphere=True,
        show_points=False,
        title="none",
        show=False,
    )
    # ---------------------------------------------------------------
    # Step 6: Draw each Voronoi edge as a great circle arc (orange tubes).
    # ---------------------------------------------------------------
    for a_idx, b_idx in edges:
        pa = svd.vertices[a_idx]
        pb = svd.vertices[b_idx]
        arc = great_circle_arc(pa, pb)
        if arc is None:
            continue
        poly = pv.PolyData(arc)
        lines = np.column_stack(
            [
                np.full(len(arc) - 1, 2, dtype=int),
                np.arange(len(arc) - 1),
                np.arange(1, len(arc)),
            ]
        )
        poly.lines = lines
        tube = poly.tube(radius=TUBE_R)
        pl.add_mesh(tube, color=(1.0, 0.6, 0.0), opacity=1.0, ambient=0.5)
    # ---------------------------------------------------------------
    # Step 7: Draw the EQ code points (red spheres).
    # ---------------------------------------------------------------
    show_r3_point_set(points_3d, show_sphere=False, plotter=pl)

    raw_file = "fig_4_10_eqp_voronoi_s2_33_raw.png"
    pl.screenshot(raw_file)
    pl.close()

    # Use Matplotlib to add the LaTeX title
    img = plt.imread(raw_file)
    fig_overlay, ax = plt.subplots(figsize=(9, 9), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    # Increase the size of the 3D portion by 50% (1.5x zoom)
    h, w = img.shape[:2]
    zoom = 1.5
    h_new, w_new = h / zoom, w / zoom
    dy, dx = (h - h_new) / 2, (w - w_new) / 2
    ax.set_ylim(h - dy, dy)
    ax.set_xlim(dx, w - dx)
    title_text = (
        r"Figure 4.10: EQ points and Voronoi cells on $S^2$ for $\mathrm{EQP}(2,33)$"
    )
    fig_overlay.text(0.5, 0.05, title_text, ha="center", fontsize=12)
    plt.savefig(
        "fig_4_10_eqp_voronoi_s2_33.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig_overlay)
    if os.path.exists(raw_file):
        os.remove(raw_file)
    if args.show_progress:
        print("Saved fig_4_10_eqp_voronoi_s2_33.png")


if __name__ == "__main__":
    main()
