"""
Figure 3.4: Maximum diameters of EQ(2, N) (log-log scale).

For each partition EQ(2, N) for N from 1 to 100,000, plot the
diameter bound coefficient N^(1/dim) * diam_bound (red dots) and the
vertex diameter coefficient (blue +), as in Figure 3.4 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 100000).
    --max-points M
        Maximum number of points to plot (default: 1000).
"""

from pathlib import Path
import argparse
import sys

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path so we can import eqsp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eqsp.region_props import eq_diam_bound, eq_vertex_diam
from eqsp.utilities import area_of_ideal_region, sradius_of_cap


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=100000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Maximum number of points to plot (default: %(default)s)",
    )
    args = parser.parse_args()
    dim = 2
    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.geomspace(1, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(1, args.upper_bound + 1)
    diam_bound = eq_diam_bound(dim, N_values)
    coeff_bound = diam_bound * np.power(N_values, 1.0 / dim)
    vertex_diam = eq_vertex_diam(dim, N_values)
    coeff_vertex = vertex_diam * np.power(N_values, 1.0 / dim)

    # Feige-Schechtman bound: 2 * sin(min(pi, 8*theta_c) / 2)
    area_r = area_of_ideal_region(dim, N_values)
    theta_c = sradius_of_cap(dim, area_r)
    fs_angle = np.minimum(np.pi, 8 * theta_c)
    fs_bound = 2 * np.sin(fs_angle / 2.0)
    coeff_fs = fs_bound * np.power(N_values, 1.0 / dim)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(N_values, coeff_fs, "k-", linewidth=1, label="Feige-Schechtman bound")
    ax.loglog(
        N_values, coeff_bound, "r.", markersize=1, label="Diameter bound coefficient"
    )
    ax.loglog(
        N_values, coeff_vertex, "b+", markersize=1, label="Maximum vertex diameter"
    )
    ax.set_xlabel(r"$\mathcal{N}$: number of regions")
    ax.set_ylabel(
        r"$(\mathrm{maxdiam} \ \mathrm{EQ}(2,\mathcal{N})) \times \mathcal{N}^{1/2}$"
    )
    ax.set_xlim(1, 1e5)
    ax.set_ylim(2, 12.5)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12])
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    fig.text(
        0.5,
        0.02,
        r"Figure 3.4: Maximum diameters of $\mathrm{EQ}(2,\mathcal{N})$ "
        r"(log-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_3_4_max_diam_s2.png", dpi=150)
    print("Saved fig_3_4_max_diam_s2.png")


if __name__ == "__main__":
    main()
