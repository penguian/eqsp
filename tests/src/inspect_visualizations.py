"""
EQSP Tests: Inspect Visualizations features

Copyright Paul Leopardi 2026
"""

# pylint: disable=import-outside-toplevel

import argparse
import os

from eqsp.partitions import eq_point_set


def run(save=False):
    """
    Run visual inspection of all eqsp.visualizations functions.

    Parameters
    ----------
    save : bool
        If True, pass a save_file path to each visualization function so
        Mayavi writes a PNG snapshot before showing the scene.
        If False, display each scene interactively.
    """
    try:
        from mayavi import mlab

        from eqsp import visualizations
    except ImportError:
        print("Mayavi not installed; skipping 3D visualizations.")
        return

    if os.environ.get("HEADLESS") or save:
        mlab.options.offscreen = True

    if save:
        print("Save mode: figures will be written to the current directory.")

    # --- show_s2_partition ---
    print("Testing show_s2_partition(20)...")
    save_file = "inspect_show_s2_partition_20.png" if save else None
    visualizations.show_s2_partition(
        20, show_points=True, show_sphere=True, show=not save, save_file=save_file
    )
    if save:
        print(f"  Saved to {save_file}")

    # --- project_s3_partition ---
    print("Testing project_s3_partition(120, proj='stereo')...")
    save_file = "inspect_project_s3_partition_120.png" if save else None
    visualizations.project_s3_partition(
        120,
        proj="stereo",
        show_points=True,
        show_surfaces=True,
        show=not save,
        save_file=save_file,
    )
    if save:
        print(f"  Saved to {save_file}")

    # --- project_point_set ---
    print("Testing project_point_set(points, proj='stereo') for S^3...")
    mlab.clf()
    points = eq_point_set(3, 120)
    save_file = "inspect_project_point_set_stereo.png" if save else None
    visualizations.project_point_set(
        points, proj="stereo", show=not save, save_file=save_file
    )
    if save:
        print(f"  Saved to {save_file}")

    print("3D Visualizations inspection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual inspection of eqsp.visualizations (Mayavi)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them.",
    )
    args = parser.parse_args()
    run(save=args.save)
