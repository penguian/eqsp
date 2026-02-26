"""
Figure 4.7: Packing density of EQP(4, N) (semi-log scale).

For N from 1 to 20,000, plot the packing density against N on a
semi-log scale, as in Figure 4.7 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
    --max-points M
        Maximum number of points to plot (default: 1000).
"""

import argparse
import os

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from scipy.special import gamma
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
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    dim = 4
    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.linspace(1, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(1, args.upper_bound + 1)
    density = eq_packing_density(dim, N_values, show_progress=args.show_progress)

    # Simple cubic lattice density: (pi^(d/2)) / (2^d * Gamma(d/2 + 1))
    sc_density = (np.pi**(dim/2)) / (2**dim * gamma(dim/2 + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(N_values, density, "b+", markersize=2)
    ax.axhline(
        y=sc_density,
        color="r",
        linestyle="--",
        linewidth=0.5,
        label="Simple cubic lattice density",
    )
    ax.plot(1, sc_density, 'ro', markersize=4)

    ax.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax.set_ylabel("Packing density")
    ax.set_xlim(1, 20000)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.text(
        0.5,
        0.02,
        r"Figure 4.7: Packing density of $\mathrm{EQP}(4,\mathcal{N})$ "
        r"(semi-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_4_7_packing_s4.png", dpi=150)
    if args.show_progress:
        print(f"Saved {os.path.basename(__file__).replace('.py', '.png')}")


if __name__ == "__main__":
    main()
