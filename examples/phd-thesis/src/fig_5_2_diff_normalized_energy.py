"""
Figure 5.2: 1 minus the normalized energy, showing convergence to 1.

For d=2, 3, 4 and N from 2 to 20,000.
Log-log plot as in Figure 5.2 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
    --max-points M
        Maximum number of points to plot (default: 250).
"""

import argparse

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eqsp.point_set_props import eq_energy_dist, sphere_int_energy

def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=20000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=250,
        help="Maximum number of points to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    dims = [2, 3, 4]
    colors = ["b", "r", "g"]

    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.geomspace(2, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(2, args.upper_bound + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for dim, color in zip(dims, colors):
        if args.show_progress:
            msg = f"  Dimension {dim}: energy for {len(N_values)} points..."
            print(msg, flush=True)
        s = dim - 1
        energy, _ = eq_energy_dist(dim, N_values, s=s, show_progress=args.show_progress)
        I_s = sphere_int_energy(dim, s)
        norm_energy = energy / ((I_s / 2) * N_values**2)

        ax.loglog(
            N_values, 1 - norm_energy, color + "+", markersize=2, label=rf"$d={dim}$"
        )

    ax.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax.set_ylabel("1 - Normalized energy")
    ax.set_xlim(1, 20000)
    ax.set_ylim(0.005, 1)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()

    fig.text(
        0.5,
        0.02,
        r"Figure 5.2: 1 minus normalized energy of "
        r"$\mathrm{EQP}(d,\mathcal{N})$ (log-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_5_2_diff_normalized_energy.png", dpi=150)
    if args.show_progress:
        print("Saved fig_5_2_diff_normalized_energy.png")

if __name__ == "__main__":
    main()
