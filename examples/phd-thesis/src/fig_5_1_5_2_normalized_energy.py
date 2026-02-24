"""
Figures 5.1 and 5.2: Normalized energy of EQP(d, N) (log-log scale).

Fig 5.1: Normalized energy E(d,N) / (sphere_int_energy * N^2 / 2),
         for d=2,3,4 and N from 2 to 20,000.
Fig 5.2: 1 minus the normalized energy, showing convergence to 1,
         for d=2,3,4 and N from 2 to 20,000.

Both are log-log plots as in Figures 5.1 and 5.2 of the thesis.

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
        default=1000,
        help="Maximum number of points to plot (default: %(default)s)",
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for dim, color in zip(dims, colors):
        s = dim - 1  # Standard Riesz energy parameter
        energy, _ = eq_energy_dist(dim, N_values, s=s)
        I_s = sphere_int_energy(dim, s)
        # Normalized energy: E / (I_s/2 * N^2)
        norm_energy = energy / ((I_s / 2) * N_values**2)
        ax1.loglog(
            N_values, norm_energy, color=color, linewidth=0.5, label=rf"$d={dim}$"
        )
        ax2.loglog(
            N_values, 1 - norm_energy, color=color, linewidth=0.5, label=rf"$d={dim}$"
        )
    ax1.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax1.set_ylabel("Normalized energy")
    ax1.set_xlim(1, 20000)
    ax1.set_ylim(0.5, 1)
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax2.set_ylabel("1 - Normalized energy")
    ax2.set_xlim(1, 20000)
    ax2.set_ylim(0.01, 1)
    ax2.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax2.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    fig.text(
        0.5,
        0.04,
        r"Figure 5.1: Normalized energy of $\mathrm{EQP}(d,\mathcal{N})$ "
        r"(log-log scale)",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        r"Figure 5.2: 1 minus normalized energy of "
        r"$\mathrm{EQP}(d,\mathcal{N})$ (log-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.savefig("fig_5_1_5_2_normalized_energy.png", dpi=150)
    print("Saved fig_5_1_5_2_normalized_energy.png")


if __name__ == "__main__":
    main()
