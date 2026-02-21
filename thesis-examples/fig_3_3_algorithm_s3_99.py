"""
Figure 3.3: Partition algorithm for EQ(3, 99).

Reproduce the illustration of the EQ partition algorithm steps
for EQ(3, 99), as shown in Figure 3.3 of the PhD thesis.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.illustrations import illustrate_eq_algorithm
import argparse


def main():
    """Generate and save the figure."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    dim = 3
    N = 99
    illustrate_eq_algorithm(dim, N)
    plt.suptitle(f"Partition algorithm for EQ({dim}, {N})", y=1.02)
    plt.tight_layout()
    plt.savefig("fig_3_3_algorithm_s3_99.png", dpi=150)
    print("Saved fig_3_3_algorithm_s3_99.png")


if __name__ == "__main__":
    main()
