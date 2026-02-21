"""
Figures 5.1 and 5.2: Normalized energy of EQP(d, N) (log-log scale).

Fig 5.1: Normalized energy E(d,N) / (sphere_int_energy * N^2 / 2),
         for d=2,3,4 and N from 2 to 20,000.
Fig 5.2: 1 minus the normalized energy, showing convergence to 1,
         for d=2,3,4 and N from 2 to 20,000.

Both are log-log plots as in Figures 5.1 and 5.2 of the thesis.

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 20000).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.point_set_props import eq_energy_dist, sphere_int_energy
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

    dims = [2, 3, 4]
    colors = ["b", "r", "g"]
    N_values = np.arange(2, args.n_max + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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

    ax1.set_xlabel("N")
    ax1.set_ylabel("Normalized energy")
    ax1.set_title(r"Normalized energy of $\mathrm{EQP}(d, N)$ (log-log scale)")
    ax1.legend()

    ax2.set_xlabel("N")
    ax2.set_ylabel("1 - Normalized energy")
    ax2.set_title(r"$1 -$ Normalized energy of $\mathrm{EQP}(d, N)$ (log-log scale)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("fig_5_1_5_2_normalized_energy.png", dpi=150)
    print("Saved fig_5_1_5_2_normalized_energy.png")


if __name__ == "__main__":
    main()
