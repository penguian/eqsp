"""
Figure 3.5: Maximum diameters of EQ(3, N) (log-log scale).

For each partition EQ(3, N) for N from 1 to 100,000, plot the
diameter bound coefficient N^(1/dim) * diam_bound (red dots) and the
vertex diameter coefficient (blue +), as in Figure 3.5 of the thesis.

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 100000).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.region_props import eq_diam_bound, eq_vertex_diam
import argparse


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-max",
        type=int,
        default=100000,
        help="Maximum number of regions N (default: %(default)s)",
    )
    args = parser.parse_args()

    dim = 3
    N_values = np.arange(1, args.n_max + 1)

    diam_bound = eq_diam_bound(dim, N_values)
    coeff_bound = diam_bound * np.power(N_values, 1.0 / dim)

    vertex_diam = eq_vertex_diam(dim, N_values)
    coeff_vertex = vertex_diam * np.power(N_values, 1.0 / dim)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(
        N_values, coeff_bound, "r.", markersize=1, label="Diameter bound coefficient"
    )
    ax.loglog(
        N_values, coeff_vertex, "b+", markersize=1, label="Vertex diameter coefficient"
    )
    ax.set_xlabel("N")
    ax.set_ylabel(r"$N^{1/3} \cdot \mathrm{diam}$")
    ax.set_title(r"Maximum diameters of $\mathrm{EQ}(3, N)$ (log-log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_3_5_max_diam_s3.png", dpi=150)
    print("Saved fig_3_5_max_diam_s3.png")


if __name__ == "__main__":
    main()
