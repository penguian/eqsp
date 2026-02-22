"""
Figure 4.9: Wyner ratios for EQP(5), EQP(6), EQP(7) (log-log scale).

For N from 2 to 1,000, plot the Wyner ratio of EQP(dim, N) for
dim = 5 (blue), 6 (red), 7 (green), as in Figure 4.9 of the thesis.

The denominator is the simple cubic lattice density for R^dim,
as described in Section 4.3 of the thesis.

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 1000).
"""

import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.point_set_props import eq_packing_density
import argparse


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-max",
        type=int,
        default=1000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    args = parser.parse_args()

    N_values = np.arange(2, args.n_max + 1)

    def simple_cubic_density(dim):
        return math.pi ** (dim / 2) / (2**dim * math.gamma(dim / 2 + 1))

    dims = [5, 6, 7]
    colors = ["b", "r", "g"]

    _, ax = plt.subplots(figsize=(10, 6))
    for dim, color in zip(dims, colors):
        density = eq_packing_density(dim, N_values)
        wyner_ratio = density / simple_cubic_density(dim)
        ax.loglog(
            N_values, wyner_ratio, color=color, linewidth=0.5, label=rf"$d={dim}$"
        )

    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("N")
    ax.set_ylabel("Wyner ratio")
    ax.set_title(
        r"Wyner ratios for $\mathrm{EQP}(5)$, $\mathrm{EQP}(6)$,"
        r" $\mathrm{EQP}(7)$ (log-log scale)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_4_9_wyner_s5_s6_s7.png", dpi=150)
    print("Saved fig_4_9_wyner_s5_s6_s7.png")


if __name__ == "__main__":
    main()
