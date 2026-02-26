"""
Figure 3.3: Partition algorithm for EQ(3, 99).

Reproduce the illustration of the EQ partition algorithm steps
for EQ(3, 99), as shown in Figure 3.3 of the PhD thesis.
"""

from pathlib import Path
import argparse
import sys

import matplotlib

# pylint: disable=wrong-import-position,ungrouped-imports,import-error
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from eqsp.illustrations import illustrate_eq_algorithm


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-progress", action="store_true", help="Show progress messages")
    args = parser.parse_args()
    dim = 3
    N = 99
    illustrate_eq_algorithm(dim, N)
    fig = plt.gcf()
    fig.text(
        0.5,
        0.02,
        r"Figure 3.3: Partition algorithm for $\mathrm{EQ}(3,99)$",
        ha="center",
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.1, top=0.95)
    plt.savefig("fig_3_3_algorithm_s3_99.png", dpi=150)
    if args.show_progress:
        print("Saved fig_3_3_algorithm_s3_99.png")


if __name__ == "__main__":
    main()
