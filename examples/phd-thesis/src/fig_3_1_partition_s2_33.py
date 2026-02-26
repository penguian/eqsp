"""
Figure 3.1: Partition EQ(2, 33).

3D illustration of the recursive zonal equal area partition of S^2
into 33 regions, as shown in Figure 3.1 of the PhD thesis.

Requires Mayavi. Run with venv_sys:
    ../venv_sys/bin/python fig_3_1_partition_s2_33.py
"""

import argparse
import os

import matplotlib.pyplot as plt
from mayavi import mlab

# pylint: disable=wrong-import-position,import-error

from eqsp.visualizations import show_s2_partition


def main():
    """Display and save the 3D figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    N = 33
    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    show_s2_partition(
        N,
        show_points=False,
        title="none",
        show=False,
    )
    # Save the raw 3D scene
    raw_file = "fig_3_1_partition_s2_33_raw.png"
    mlab.savefig(raw_file)

    # Use Matplotlib to add the LaTeX title
    img = plt.imread(raw_file)
    fig_overlay, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    title_text = r"Figure 3.1: Partition $\mathrm{EQ}(2,33)$"
    # Add title at bottom center
    fig_overlay.text(0.5, 0.05, title_text, ha="center", fontsize=12)
    plt.savefig("fig_3_1_partition_s2_33.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig_overlay)

    if os.path.exists(raw_file):
        os.remove(raw_file)

    if args.show_progress:
        print("Saved fig_3_1_partition_s2_33.png")


if __name__ == "__main__":
    main()
