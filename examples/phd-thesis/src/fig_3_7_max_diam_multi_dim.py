"""
Figure 3.7: Maximum diameters of EQ(d, N), d from 2 to 8 (log-log scale).

For partitions EQ(d, 2^k) for d from 2 to 8 and k from 1 to 20,
plot the diameter bound coefficient 2^(k/d) * diam_bound (red dots)
and vertex diameter coefficient (blue +), as in Figure 3.7 of the thesis.

Command-line arguments:
    --k-max N
        Maximum exponent k: N ranges over 2^1, ..., 2^k (default: 20).
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

from eqsp.region_props import eq_diam_bound, eq_vertex_diam

def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k-max",
        type=int,
        default=20,
        help="Maximum exponent k: N = 2^k, k in 1..k-max (default: %(default)s)",
    )
    args = parser.parse_args()
    dims = range(2, 9)
    k_values = np.arange(1, args.k_max + 1)
    _, ax = plt.subplots(figsize=(10, 6))
    for dim in dims:
        N_values = 2**k_values
        coeff_bound = eq_diam_bound(dim, N_values) * np.power(N_values, 1.0 / dim)
        coeff_vertex = eq_vertex_diam(dim, N_values) * np.power(N_values, 1.0 / dim)
        ax.loglog(N_values, coeff_bound, "r.", markersize=4)
        ax.loglog(N_values, coeff_vertex, "b+", markersize=4)
    # Legend proxies
    ax.loglog([], [], "r.", markersize=4, label="Diameter bound coefficient")
    ax.loglog([], [], "b+", markersize=4, label="Vertex diameter coefficient")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$N^{1/d} \cdot \mathrm{diam}$")
    ax.set_title(
        r"Maximum diameters of $\mathrm{EQ}(d, N)$, $d$ from 2 to 8 (log-log scale)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_3_7_max_diam_multi_dim.png", dpi=150)
    print("Saved fig_3_7_max_diam_multi_dim.png")
if __name__ == "__main__":
    main()
