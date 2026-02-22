"""
Figure 4.4: Minimum distance coefficient of EQP(4, N) (semi-log scale).

For N from 1 to 20,000, plot the minimum distance coefficient
N^(1/dim) * min_dist as a function of N on a semi-log scale,
as in Figure 4.4 of the thesis.

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 20000).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.point_set_props import eq_dist_coeff
import argparse


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-max",
        type=int,
        default=20000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    args = parser.parse_args()

    dim = 4
    N_values = np.arange(1, args.n_max + 1)
    coeff = eq_dist_coeff(dim, N_values)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(N_values, coeff, "b.", markersize=1)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$N^{1/4} \cdot \mathrm{min\_dist}$")
    ax.set_title(
        r"Minimum distance coefficient of $\mathrm{EQP}(4, N)$ (semi-log scale)"
    )
    plt.tight_layout()
    plt.savefig("fig_4_4_min_dist_s4.png", dpi=150)
    print("Saved fig_4_4_min_dist_s4.png")


if __name__ == "__main__":
    main()
