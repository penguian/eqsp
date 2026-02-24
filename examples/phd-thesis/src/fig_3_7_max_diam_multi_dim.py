"""
Figure 3.7: Maximum diameters of EQ(d, N), d from 2 to 8 (log-log scale).

For partitions EQ(d, 2^k) for d from 2 to 8 and k from 1 to 20,
plot the diameter bound coefficient N^(1/d) * diam_bound (red dots)
and vertex diameter coefficient (blue +), as in Figure 3.7 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 1048576, i.e. 2^20).
    --max-points M
        Number of points to plot (default: 20). These will be powers of 2.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from eqsp.region_props import eq_diam_bound, eq_vertex_diam


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=2**20,
        help="Maximum number of regions N (default: %(default)s)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20,
        help="Number of points to plot (default: %(default)s)",
    )
    args = parser.parse_args()
    dims = range(2, 9)
    # Generate N values as powers of 2
    k_max = int(np.floor(np.log2(args.upper_bound)))
    k_values = np.linspace(1, k_max, args.max_points).round().astype(int)
    N_values = 2**k_values
    fig, ax = plt.subplots(figsize=(10, 6))
    for dim in dims:
        coeff_bound = eq_diam_bound(dim, N_values) * np.power(N_values, 1.0 / dim)
        coeff_vertex = eq_vertex_diam(dim, N_values) * np.power(N_values, 1.0 / dim)
        ax.loglog(N_values, coeff_bound, "r.", markersize=4)
        ax.loglog(N_values, coeff_vertex, "b+", markersize=4)
    # Legend proxies
    ax.loglog([], [], "r.", markersize=4, label="Diameter bound coefficient")
    ax.loglog([], [], "b+", markersize=4, label="Vertex diameter coefficient")
    ax.set_xlabel(r"$\mathcal{N}$: number of regions")
    ax.set_ylabel(
        r"$(\mathrm{maxdiam} \ \mathrm{EQ}(d,\mathcal{N})) \times \mathcal{N}^{1/d}$"
    )
    ax.set_xlim(1, 1e5)
    ax.set_ylim(2, 12)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    fig.text(
        0.5,
        0.02,
        r"Figure 3.7: Maximum diameters of $\mathrm{EQ}(d,\mathcal{N})$, "
        r"$d$ from 2 to 8 (log-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_3_7_max_diam_multi_dim.png", dpi=150)
    print("Saved fig_3_7_max_diam_multi_dim.png")


if __name__ == "__main__":
    main()
