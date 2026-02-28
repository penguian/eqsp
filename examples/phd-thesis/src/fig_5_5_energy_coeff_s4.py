"""
Figure 5.5: Energy coefficient of EQP(4, N) (semi-log scale).

For N from 2 to 20,000, plot the energy coefficient of EQP(4, N),
as in Figure 5.5 of the thesis. Uses s = dim - 1 = 3 (Riesz energy).
Plots ec_4(N) = -2 * eq_energy_coeff(4, N) as per Remark, page 198.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
    --max-points M
        Maximum number of points to plot (default: 1000). Uses hybrid sampling:
        N=2..100 linear, then log-spaced points up to the upper bound.
"""

import argparse

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    dim = 4
    if args.upper_bound > 100:
        # Hybrid sampling: linear (1-100) and logarithmic (100-upper_bound)
        n_linear = np.arange(1, 101)
        n_log = np.geomspace(100, args.upper_bound, max(2, args.max_points - 100))
        N_values = np.unique(np.concatenate([n_linear, n_log.round().astype(int)]))
    else:
        N_values = np.arange(1, args.upper_bound + 1)
    coeff = eq_energy_coeff(dim, N_values, show_progress=args.show_progress)
    # ec_d(N) = -2 * eq_energy_coeff(dim, N)
    coeff_energy = -2 * coeff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(N_values, coeff_energy, "b+", markersize=2.5)
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
    if args.show_progress:
        print("Saved fig_5_5_energy_coeff_s4.png")


if __name__ == "__main__":
    main()
