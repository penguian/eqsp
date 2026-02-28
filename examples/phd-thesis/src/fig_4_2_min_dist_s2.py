"""
Figure 4.2: Minimum distance coefficient of EQP(2, N) (semi-log scale).

For N from 1 to 20,000, plot the minimum distance coefficient
N^(1/dim) * min_dist as a function of N on a semi-log scale,
as in Figure 4.2 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
    --max-points M
        Maximum number of points to plot (default: 1000).
"""

import argparse

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from eqsp.point_set_props import eq_dist_coeff


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
        default=1000,
        help="Maximum number of points to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    dim = 2
    if args.upper_bound <= 100:
        N_values = np.arange(1, args.upper_bound + 1)
    else:
        N_small = np.arange(1, 101)
        N_large = np.geomspace(100, args.upper_bound, 900)
        N_values = np.unique(np.concatenate([N_small, N_large]).round().astype(int))
    coeff_dist = eq_dist_coeff(dim, N_values, show_progress=args.show_progress)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(
        N_values,
        coeff_dist,
        "b+",
        markersize=2,
        label=r"$\mathrm{min dist} \ \mathrm{EQP}(2, \mathcal{N})$",
    )
    ax.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax.set_ylabel(r"Minimum distance multiplied by $\mathcal{N}^{1/2}$")
    ax.set_xlim(1, 2**14)
    ax.set_ylim(2.4, 3.6)

    # Format y-axis to one decimal place
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.text(
        0.5,
        0.02,
        r"Figure 4.2: Minimum distance coefficient of "
        r"$\mathrm{EQP}(2,\mathcal{N})$ (semi-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_4_2_min_dist_s2.png", dpi=150)
    if args.show_progress:
        print("Saved fig_4_2_min_dist_s2.png")


if __name__ == "__main__":
    main()
