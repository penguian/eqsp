"""
Figure 4.3: Minimum distance coefficient of EQP(3, N) (semi-log scale).

For N from 1 to 20,000, plot the minimum distance coefficient
N^(1/dim) * min_dist as a function of N on a semi-log scale,
as in Figure 4.3 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
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
    args = parser.parse_args()
    dim = 3
    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.linspace(1, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(1, args.upper_bound + 1)
    coeff_dist = eq_dist_coeff(dim, N_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(N_values, coeff_dist, "r.", markersize=1)
    ax.set_xlabel(r"$\mathcal{N} = \text{number of points}$")
    ax.set_ylabel(
        r"$\operatorname{mindist}(\mathrm{EQP}(3,\mathcal{N})) \times "
        r"\mathcal{N}^{1/3}$"
    )
    ax.set_xlim(1, 20000)
    ax.set_ylim(0.7, 1.15)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.text(
        0.5,
        0.02,
        r"Figure 4.3: Minimum distance coefficient of "
        r"$\mathrm{EQP}(3,\mathcal{N})$ (semi-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_4_3_min_dist_s3.png", dpi=150)
    print("Saved fig_4_3_min_dist_s3.png")


if __name__ == "__main__":
    main()
