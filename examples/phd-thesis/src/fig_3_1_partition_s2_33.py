"""
Figure 3.1: Partition EQ(2, 33).

3D illustration of the recursive zonal equal area partition of S^2
into 33 regions, as shown in Figure 3.1 of the PhD thesis.

Requires Mayavi. Run with venv_sys:
    ../venv_sys/bin/python fig_3_1_partition_s2_33.py
"""

from pathlib import Path
import argparse
import sys

from mayavi import mlab

# pylint: disable=wrong-import-position,import-error
# Add project root to sys.path so we can import eqsp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eqsp.visualizations import show_s2_partition


def main():
    """Display and save the 3D figure."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    N = 33
    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    show_s2_partition(
        N,
        title=r"Figure 3.1: Partition $\mathrm{EQ}(2,33)$",
        title_pos=(0.3, 0.05),
        show=True,
        save_file="fig_3_1_partition_s2_33.png",
    )


if __name__ == "__main__":
    main()
