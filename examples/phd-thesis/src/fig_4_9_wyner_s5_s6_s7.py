"""
Figure 4.9: Wyner ratios for EQP(5), EQP(6), EQP(7) (log-log scale).

For N from 2 to 1,000, plot the Wyner ratio of EQP(dim, N) for
dim = 5 (blue), 6 (red), 7 (green), as in Figure 4.9 of the thesis.

The denominator is the simple cubic lattice density for R^dim,
as described in Section 4.3 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 1000).
    --max-points M
        Maximum number of points to plot (default: 1000).
"""

from pathlib import Path
import argparse
import math
import sys

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path so we can import eqsp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eqsp.point_set_props import eq_packing_density


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=1000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Maximum number of points to plot (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.geomspace(2, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(2, args.upper_bound + 1)

    def simple_cubic_density(dim):
        return math.pi ** (dim / 2) / (2**dim * math.gamma(dim / 2 + 1))

    dims = [5, 6, 7]

    # Pre-calculate Wyner ratios for each dimension
    wyner_ratios_by_dim = {}
    for dim in dims:
        density = eq_packing_density(dim, N_values)
        wyner_ratios_by_dim[dim] = density / simple_cubic_density(dim)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(
        N_values, wyner_ratios_by_dim[5], "b-", linewidth=1, label=r"$\mathrm{EQP}(5)$"
    )
    ax.loglog(
        N_values, wyner_ratios_by_dim[6], "r-", linewidth=1, label=r"$\mathrm{EQP}(6)$"
    )
    ax.loglog(
        N_values, wyner_ratios_by_dim[7], "g-", linewidth=1, label=r"$\mathrm{EQP}(7)$"
    )
    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\mathcal{N} = \text{number of points}$")
    ax.set_ylabel("Wyner ratio")
    ax.set_ylim(0.6, 1.05)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    fig.text(
        0.5,
        0.02,
        r"Figure 4.9: Wyner ratios for $\mathrm{EQP}(5)$, "
        r"$\mathrm{EQP}(6)$, $\mathrm{EQP}(7)$ (log-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_4_9_wyner_s5_s6_s7.png", dpi=150)
    print("Saved fig_4_9_wyner_s5_s6_s7.png")


if __name__ == "__main__":
    main()
