"""
Figure 4.1: EQ code EQP(2, 33), showing the partition EQ(2, 33).

3D illustration of the EQ partition of S^2 into 33 regions, with the
33 EQ code points (region centers) shown in red.
As in Figure 4.1 of the PhD thesis.

Requires PyVista.
"""

# pylint: disable=wrong-import-position,import-error

import argparse
import os

import matplotlib.pyplot as plt
import pyvista as pv

from eqsp.visualizations import show_s2_partition


def main():
    """Generate and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-progress", action="store_true", help="Show progress messages"
    )
    args = parser.parse_args()
    N = 33
    pv.OFF_SCREEN = True
    # show_points=True (default) shows the EQ code points in red
    pl = show_s2_partition(
        N,
        show_points=True,
        title="none",
        show=False,
    )
    # Save the raw 3D scene
    raw_file = "fig_4_1_eqp_s2_33_raw.png"
    pl.screenshot(raw_file)
    pl.close()

    # Use Matplotlib to add the LaTeX title
    img = plt.imread(raw_file)
    fig_overlay, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    # Increase the size of the 3D portion by 50% (1.5x zoom)
    h, w = img.shape[:2]
    zoom = 1.5
    h_new, w_new = h / zoom, w / zoom
    dy, dx = (h - h_new) / 2, (w - w_new) / 2
    ax.set_ylim(h - dy, dy)
    ax.set_xlim(dx, w - dx)
    title_text = (
        r"Figure 4.1: EQ code $\mathrm{EQP}(2,33)$, showing the "
        r"partition $\mathrm{EQ}(2,33)$"
    )
    # Add title at bottom center
    fig_overlay.text(0.5, 0.05, title_text, ha="center", fontsize=12)
    plt.savefig("fig_4_1_eqp_s2_33.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig_overlay)
    if os.path.exists(raw_file):
        os.remove(raw_file)
    if args.show_progress:
        print("Saved fig_4_1_eqp_s2_33.png")


if __name__ == "__main__":
    main()
