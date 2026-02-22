"""
Figure 4.8: Wyner ratios for EQP(2), EQP(3), EQP(4) (semi-log scale).

For N from 2 to 20,000, plot the Wyner ratio of EQP(dim, N) for
dim = 2 (blue), 3 (red), 4 (green), as in Figure 4.8 of the thesis.

The Wyner ratio is the packing density of the code divided by the
density of the optimal packing in R^dim (based on Wyner's bound).
For dim = 2, the Wyner bound uses the packing density of the hexagonal lattice.
For a simpler approximation we use the simple cubic lattice density
as a reference, as described in Section 4.3 of the thesis.

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 20000).
"""

from pathlib import Path
import argparse
import math
import sys

import matplotlib
import numpy as np

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path so we can import eqsp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eqsp.point_set_props import eq_packing_density

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
    N_values = np.arange(2, args.n_max + 1)
    # Simple cubic lattice reference densities for dim=2,3,4
    def simple_cubic_density(dim):
        return math.pi ** (dim / 2) / (2**dim * math.gamma(dim / 2 + 1))
    dims = [2, 3, 4]
    colors = ["b", "r", "g"]
    _, ax = plt.subplots(figsize=(10, 6))
    for dim, color in zip(dims, colors):
        density = eq_packing_density(dim, N_values)
        wyner_ratio = density / simple_cubic_density(dim)
        ax.semilogy(
            N_values, wyner_ratio, color=color, linewidth=0.5, label=rf"$d={dim}$"
        )
    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("N")
    ax.set_ylabel("Wyner ratio")
    ax.set_title(
        r"Wyner ratios for $\mathrm{EQP}(2)$, $\mathrm{EQP}(3)$,"
        r" $\mathrm{EQP}(4)$ (semi-log scale)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_4_8_wyner_s2_s3_s4.png", dpi=150)
    print("Saved fig_4_8_wyner_s2_s3_s4.png")
if __name__ == "__main__":
    main()
