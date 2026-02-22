"""
Figure 4.10 (3D): EQ code EQP(2, 33), Voronoi cells on the sphere, and EQ(2, 33).

Computes the spherical Voronoi diagram of EQP(2, 33) directly on S^2 using
scipy.spatial.SphericalVoronoi, then draws each Voronoi edge as a great circle
arc via spherical linear interpolation (SLERP). This correctly represents Voronoi
cell edges, which are great circle arcs, not straight lines.

Requires Mayavi. Run with venv_sys:
    ../venv_sys/bin/python fig_4_10_eqp_voronoi_s2_33_3d.py
"""

import numpy as np
from scipy.spatial import SphericalVoronoi
import eqsp
from mayavi import mlab
from eqsp.visualizations import show_s2_partition, show_r3_point_set
import argparse


def main():
    """Generate and save the figure."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    N = 33
    dim = 2
    SAMPLES = 80  # points per great circle arc
    TUBE_R = np.sqrt(1.0 / N) / 12.0  # matches tube radius used by show_s2_region

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
    # Step 5: Set up Mayavi scene with EQ partition regions (blue) and sphere.
    # ---------------------------------------------------------------
    mlab.figure(bgcolor=(1, 1, 1), size=(900, 900))

    show_s2_partition(
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
        mlab.plot3d(
            arc[:, 0],
            arc[:, 1],
            arc[:, 2],
            color=(1.0, 0.6, 0.0),
            tube_radius=TUBE_R,
            opacity=1.0,
        )

    # ---------------------------------------------------------------
    # Step 7: Draw the EQ code points (red spheres).
    # ---------------------------------------------------------------
    show_r3_point_set(points_3d, show_sphere=False, scale_factor=0.06)

    title = (
        f"EQP(2,{N}): Voronoi cells (orange, great circle arcs)"
        f" and EQ(2,{N}) partition (blue)"
    )
    mlab.text(0.05, 0.9, title, width=0.9, color=(0, 0, 0))
    mlab.savefig("fig_4_10_eqp_voronoi_s2_33_3d.png")
    mlab.show()


if __name__ == "__main__":
    main()
