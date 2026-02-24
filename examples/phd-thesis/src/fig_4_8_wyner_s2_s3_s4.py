"""
Figure 4.8: Wyner ratios for EQP(2), EQP(3), EQP(4) (semi-log scale).

For N from 2 to 20,000, plot the Wyner ratio of EQP(dim, N) for
dim = 2 (blue), 3 (red), 4 (green), as in Figure 4.8 of the thesis.

The Wyner ratio is the packing density of the code divided by the
density of the optimal packing in R^dim (based on Wyner's bound).
For dim = 2, the Wyner bound uses the packing density of the hexagonal lattice.
For a simpler approximation we use the simple cubic lattice density
as a reference, as described in Section 4.3 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
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
    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.linspace(2, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(2, args.upper_bound + 1)

    # Simple cubic lattice reference densities for dim=2,3,4
    def simple_cubic_density(dim):
        return math.pi ** (dim / 2) / (2**dim * math.gamma(dim / 2 + 1))

    dims = [2, 3, 4]
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = {}
    for dim in dims:
        density = eq_packing_density(dim, N_values)
        ratios[dim] = density / simple_cubic_density(dim)

    ax.semilogx(N_values, ratios[2], "b-", linewidth=1, label=r"$\mathrm{EQP}(2)$")
    ax.semilogx(N_values, ratios[3], "r-", linewidth=1, label=r"$\mathrm{EQP}(3)$")
    ax.semilogx(N_values, ratios[4], "g-", linewidth=1, label=r"$\mathrm{EQP}(4)$")
    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\mathcal{N} = \text{number of points}$")
    ax.set_ylabel("Wyner ratio")
    ax.set_xlim(1, 20000)
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    fig.text(
        0.5,
        0.02,
        r"Figure 4.8: Wyner ratios for $\mathrm{EQP}(2)$, $\mathrm{EQP}(3)$, "
        r"$\mathrm{EQP}(4)$ (semi-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_4_8_wyner_s2_s3_s4.png", dpi=150)
    print("Saved fig_4_8_wyner_s2_s3_s4.png")


if __name__ == "__main__":
    main()
