"""
Figure 5.5: Energy coefficient of EQP(4, N) (semi-log scale).

For N from 2 to 20,000, plot the energy coefficient of EQP(4, N),
as in Figure 5.4 of the thesis. Uses s = dim - 1 = 3 (Riesz energy).

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

from eqsp.point_set_props import eq_energy_coeff


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
    dim = 4
    if args.upper_bound > args.max_points:
        N_values = np.unique(
            np.linspace(2, args.upper_bound, args.max_points).round().astype(int)
        )
    else:
        N_values = np.arange(2, args.upper_bound + 1)
    coeff = eq_energy_coeff(dim, N_values)
    coeff_energy = coeff * (N_values ** (1 / dim))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(N_values, coeff_energy, "r.", markersize=1)
    ax.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax.set_ylabel(r"$ec_4(\mathcal{N}) = (1 - E) \mathcal{N}^{1/4}$")
    ax.set_xlim(1, 20000)
    ax.set_ylim(1, 1.4)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.text(
        0.5,
        0.02,
        r"Figure 5.5: Energy coefficient of $\mathrm{EQP}(4,\mathcal{N})$ "
        r"(semi-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_5_5_energy_coeff_s4.png", dpi=150)
    print("Saved fig_5_5_energy_coeff_s4.png")


if __name__ == "__main__":
    main()
