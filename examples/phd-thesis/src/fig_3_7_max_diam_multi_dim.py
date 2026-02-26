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


from concurrent.futures import ProcessPoolExecutor, as_completed


from eqsp.region_props import eq_diam_bound, eq_vertex_diam


def compute_dim_data(dim, N_values, show_progress=False):
    """Worker function to calculate diameter data for a single dimension."""
    if show_progress:
        print(f"  Dimension {dim}: calculating {len(N_values)} points...", flush=True)
    coeff_bound = eq_diam_bound(dim, N_values) * np.power(N_values, 1.0 / dim)
    coeff_vertex = eq_vertex_diam(dim, N_values) * np.power(N_values, 1.0 / dim)
    return dim, coeff_bound, coeff_vertex


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=2**14,
        help="Maximum number of regions N (default: %(default)s)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20,
        help="Number of points to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()

    # Generate N values as powers of 2
    k_max = int(np.floor(np.log2(args.upper_bound)))
    k_values = np.linspace(1, k_max, args.max_points).round().astype(int)
    N_values = 2**k_values

    dims = sorted(range(2, 9), reverse=True)
    results = {}

    if args.show_progress:
        print(f"Parallelizing calculations for dimensions {list(dims)} using 2 workers...")

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(compute_dim_data, dim, N_values, args.show_progress): dim
            for dim in dims
        }
        for future in as_completed(futures):
            dim, coeff_bound, coeff_vertex = future.result()
            results[dim] = (coeff_bound, coeff_vertex)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot in increasing dimension order for consistency
    for dim in sorted(results.keys()):
        coeff_bound, coeff_vertex = results[dim]
        ax.loglog(N_values, coeff_bound, "b+", markersize=4)
        ax.loglog(N_values, coeff_vertex, "b+", markersize=4)

    # Legend proxies
    ax.loglog([], [], "b+", markersize=4, label="Diameter bound coefficient")
    ax.loglog([], [], "b+", markersize=4, label="Vertex diameter coefficient")
    ax.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax.set_ylabel(r"Maximum diameter multiplied by $\mathcal{N}^{1/2}$")
    ax.set_xlim(1, 2**14)
    ax.set_ylim(2, 8)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([2, 3, 4, 5, 6, 7, 8])
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
    if args.show_progress:
        print("Saved fig_3_7_max_diam_multi_dim.png")


if __name__ == "__main__":
    main()
