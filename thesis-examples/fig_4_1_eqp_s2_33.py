"""
Figure 4.1: EQ code EQP(2, 33), showing the partition EQ(2, 33).

3D illustration of the EQ partition of S^2 into 33 regions, with the
33 EQ code points (region centers) shown in red.
As in Figure 4.1 of the PhD thesis.

Requires Mayavi. Run with venv_sys:
    ../venv_sys/bin/python fig_4_1_eqp_s2_33.py
"""

from mayavi import mlab
from eqsp.visualizations import show_s2_partition
import argparse


def main():
    """Generate and save the figure."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    N = 33
    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    # show_points=True (default) shows the EQ code points in red
    show_s2_partition(
        N,
        show_points=True,
        title="short",
        show=True,
        save_file="fig_4_1_eqp_s2_33.png",
    )


if __name__ == "__main__":
    main()
