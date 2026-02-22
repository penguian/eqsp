"""
Figure 5.5: Energy coefficient of EQP(4, N) (semi-log scale).

For N from 2 to 20,000, plot the energy coefficient of EQP(4, N),
as in Figure 5.4 of the thesis. Uses s = dim - 1 = 3 (Riesz energy).

Command-line arguments:
    --n-max N
        Maximum number of regions N to compute (default: 20000).
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

from eqsp.point_set_props import eq_energy_coeff

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
    N_values = np.arange(2, args.n_max + 1)
    coeff = eq_energy_coeff(dim, N_values)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(N_values, np.abs(coeff), "b.", markersize=1)
    ax.set_xlabel("N")
    ax.set_ylabel("Energy coefficient")
    ax.set_title(r"Energy coefficient of $\mathrm{EQP}(4, N)$ (semi-log scale)")
    plt.tight_layout()
    plt.savefig("fig_5_5_energy_coeff_s4.png", dpi=150)
    print("Saved fig_5_5_energy_coeff_s4.png")
if __name__ == "__main__":
    main()
