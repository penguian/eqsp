"""
PyEQSP Tests: Inspect Illustrations features

Copyright Paul Leopardi 2026
"""

import argparse
import os

import matplotlib.pyplot as plt

from eqsp import illustrations
from eqsp.partitions import eq_point_set


def run(save=False):
    """
    Run visual inspection of all eqsp.illustrations functions.

    Parameters
    ----------
    save : bool
        If True, save each figure to a PNG file and close it.
        If False, display each figure interactively.
    """
    if save:
        plt.switch_backend("Agg")
        print("Save mode: figures will be written to the current directory.")
    elif os.environ.get("HEADLESS"):
        plt.switch_backend("Agg")

    # --- illustrate_eq_algorithm ---
    print("Testing illustrate_eq_algorithm(2, 20)...")
    plt.figure()
    illustrations.illustrate_eq_algorithm(2, 20, show=False)
    if save:
        path = "inspect_illustrate_eq_algorithm_2_20.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {path}")
    else:
        plt.show()

    # --- project_s2_partition ---
    print("Testing project_s2_partition(20, proj='eqarea', show_points=True)...")
    fig, ax = plt.subplots()
    illustrations.project_s2_partition(
        20, proj="eqarea", show_points=True, ax=ax, show=False
    )
    if save:
        path = "inspect_project_s2_partition_20_eqarea.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {path}")
    else:
        plt.show()

    # --- project_point_set ---
    print("Testing project_point_set(points, proj='stereo')...")
    points = eq_point_set(2, 20)
    fig, ax = plt.subplots()
    illustrations.project_point_set(points, ax=ax, proj="stereo", show=False)
    if save:
        path = "inspect_project_point_set_stereo.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {path}")
    else:
        plt.show()

    print("2D Illustrations inspection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual inspection of eqsp.illustrations (Matplotlib)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them.",
    )
    args = parser.parse_args()
    run(save=args.save)
