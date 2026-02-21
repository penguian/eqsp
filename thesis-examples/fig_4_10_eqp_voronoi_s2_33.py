"""
Figure 4.10 (2D): EQ code EQP(2, 33), Voronoi cells, and EQ(2, 33).

2D stereographic projection showing Voronoi cells of EQP(2, 33) and the
boundaries of partition EQ(2, 33).

The Voronoi cells are computed via scipy.spatial.Voronoi on the
stereographic projection coordinates. The north polar cap centre (z=1)
is excluded from the projection (it maps to infinity) and rendered
separately.

This script saves a PNG file and does not require a display.
"""

import matplotlib

matplotlib.use("Agg")
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d  # pylint: disable=no-name-in-module
import eqsp
from eqsp.illustrations import project_s2_partition


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    N = 33
    dim = 2

    # Get EQ code points
    points_x = eqsp.eq_point_set(dim, N)

    # Stereographic projection from north pole: (x,y,z) -> (x/(1-z), y/(1-z))
    # Exclude the north pole (z=1) which maps to infinity.
    tol = 1e-10
    north_pole_mask = np.abs(points_x[2] - 1) < tol
    finite_pts = points_x[:, ~north_pole_mask]

    px = finite_pts[0] / (1 - finite_pts[2])
    py = finite_pts[1] / (1 - finite_pts[2])
    proj_2d = np.column_stack([px, py])

    _, ax = plt.subplots(figsize=(10, 10))

    # Voronoi cells from projected code points
    mirror = proj_2d * -10
    all_pts = np.vstack([proj_2d, mirror])
    vor = Voronoi(all_pts)
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        line_colors="orange",
        line_width=1.2,
        line_alpha=0.7,
        point_size=0,
    )

    # Partition boundaries
    project_s2_partition(N, proj="stereo", ax=ax)

    ax.plot(px, py, "r.", markersize=5, zorder=5)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title(f"EQ code EQP(2, {N}), Voronoi cells, and EQ(2, {N})")
    plt.tight_layout()
    plt.savefig("fig_4_10_eqp_voronoi_s2_33.png", dpi=150)
    print("Saved fig_4_10_eqp_voronoi_s2_33.png")


if __name__ == "__main__":
    main()
