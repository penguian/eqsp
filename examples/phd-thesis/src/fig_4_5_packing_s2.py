"""
Figure 4.5: Packing density of EQP(2, N) (semi-log scale).

For N from 1 to 20,000, plot the packing density against N on a
semi-log scale, as in Figure 4.5 of the thesis.

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 20000).
"""

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
        default=20000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    args = parser.parse_args()

    dim = 2
    N_values = np.arange(1, args.n_max + 1)
    density = eq_packing_density(dim, N_values)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(N_values, density, "b.", markersize=1)
    ax.set_xlabel("N")
    ax.set_ylabel("Packing density")
    ax.set_title(r"Packing density of $\mathrm{EQP}(2, N)$ (semi-log scale)")
    plt.tight_layout()
    plt.savefig("fig_4_5_packing_s2.png", dpi=150)
    print("Saved fig_4_5_packing_s2.png")


if __name__ == "__main__":
    main()
