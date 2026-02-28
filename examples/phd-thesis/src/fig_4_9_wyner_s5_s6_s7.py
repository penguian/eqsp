"""
Figure 4.9: Wyner ratios for EQP(5), EQP(6), EQP(7) (log-log scale).

For N from 2 to 1,000, plot the Wyner ratio of EQP(dim, N) for
dim = 5 (blue), 6 (red), 7 (green), as in Figure 4.9 of the thesis.
Uses a log-log scale to match thesis aesthetic.

The denominator is the simple cubic lattice density for R^dim,
as described in Section 4.3 of the thesis.

Command-line arguments:
    --upper-bound N
        Maximum number of regions N to compute (default: 20000).
    --max-points M
        Maximum number of points to plot (default: 1000). Uses hybrid sampling:
        N=1..100 linear, then log-spaced points up to the upper bound.
"""

import argparse

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eqsp.point_set_props import eq_min_dist
from eqsp.utilities import area_of_cap, area_of_sphere, euc2sph_dist


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
    if args.upper_bound > 100:
        # Hybrid sampling: linear (1-100) and logarithmic (100-upper_bound)
        n_linear = np.arange(1, 101)
        n_log = np.geomspace(100, args.upper_bound, max(2, args.max_points - 100))
        N_values = np.unique(np.concatenate([n_linear, n_log.round().astype(int)]))
    else:
        N_values = np.arange(1, args.upper_bound + 1)

    dims = [5, 6, 7]
    wyner_ratios_by_dim = {}
    sphere_areas = {d: area_of_sphere(d) for d in dims}

    for dim in dims:
        if args.show_progress:
            print(f"Calculating Wyner ratio for dim={dim}...")

        # 1. Calculate packing density: N * area_of_cap(dim, rho) / area_of_sphere(dim)
        min_dist_euc = eq_min_dist(dim, N_values, show_progress=args.show_progress)
        rho = euc2sph_dist(min_dist_euc) / 2.0
        pdens = N_values * area_of_cap(dim, rho) / sphere_areas[dim]

        # 2. Calculate CSW lower bound: area_of_cap(dim, rho) / area_of_cap(dim, 2*rho)
        rho2 = np.clip(2.0 * rho, 0, np.pi)
        csw_bound = area_of_cap(dim, rho) / area_of_cap(dim, rho2)

        # 3. Wyner ratio = pdens / csw_bound
        wyner_ratios_by_dim[dim] = pdens / csw_bound

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(
        N_values,
        wyner_ratios_by_dim[5],
        "b+",
        markersize=2,
        label=r"$\mathrm{EQP}(5)$",
    )
    ax.loglog(
        N_values,
        wyner_ratios_by_dim[6],
        "r+",
        markersize=2,
        label=r"$\mathrm{EQP}(6)$",
    )
    ax.loglog(
        N_values,
        wyner_ratios_by_dim[7],
        "g+",
        markersize=2,
        label=r"$\mathrm{EQP}(7)$",
    )
    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\mathcal{N}$: number of codepoints")
    ax.set_ylabel("Wyner ratio")
    ax.set_xlim(1, 20000)
    ax.set_ylim(0.25, 8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    fig.text(
        0.5,
        0.02,
        r"Figure 4.9: Wyner ratios for $\mathrm{EQP}(5)$, "
        r"$\mathrm{EQP}(6)$, $\mathrm{EQP}(7)$ (log-log scale)",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_4_9_wyner_s5_s6_s7.png", dpi=150)
    if args.show_progress:
        print("Saved fig_4_9_wyner_s5_s6_s7.png")


if __name__ == "__main__":
    main()
