"""
Figure 5.4: Energy coefficient of EQP(3, N) (semi-log scale).

For N from 2 to 20,000, plot the energy coefficient of EQP(3, N),
as in Figure 5.4 of the thesis. Uses s = dim - 1 = 2 (Riesz energy).

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 20000).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.point_set_props import eq_energy_coeff
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

    dim = 3
    N_values = np.arange(2, args.n_max + 1)
    coeff = eq_energy_coeff(dim, N_values)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(N_values, np.abs(coeff), "b.", markersize=1)
    ax.set_xlabel("N")
    ax.set_ylabel("Energy coefficient")
    ax.set_title(r"Energy coefficient of $\mathrm{EQP}(3, N)$ (semi-log scale)")
    plt.tight_layout()
    plt.savefig("fig_5_4_energy_coeff_s3.png", dpi=150)
    print("Saved fig_5_4_energy_coeff_s3.png")


if __name__ == "__main__":
    main()
