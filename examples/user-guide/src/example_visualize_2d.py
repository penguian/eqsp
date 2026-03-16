"""
2D Visualization Example for PyEQSP

This script demonstrates 2D projections of the EQ algorithm and point sets
using Matplotlib.
"""
# ruff: noqa: E402

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from eqsp import illustrations  # pylint: disable=wrong-import-position


def main():
    """
    Generate 2D visualizations.

    Example
    -------
    >>> import matplotlib
    >>> matplotlib.use('Agg') # Use non-interactive backend for testing
    >>> main()
    --- Visualizing EQ Algorithm with N=100 ---
    Projections generated. Run plt.show() to view if in an interactive session.
    """
    N = 100

    # Use Agg backend if running in a non-interactive environment
    try:
        plt.figure(figsize=(10, 8))
    except Exception: # pylint: disable=broad-exception-caught  # pragma: no cover
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 8))

    print(f"--- Visualizing EQ Algorithm with N={N} ---")

    # 1. Illustrate the "igloo" partitioning scheme
    # Uses the default equal-area projection
    illustrations.illustrate_eq_algorithm(dim=2, N=N, show=False)
    plt.suptitle(f"EQ Algorithm Illustration (N={N})", fontsize=16)

    # 2. Show a specific projection of a partition
    plt.figure(figsize=(8, 8))
    illustrations.project_s2_partition(N=N, proj='stereo', show_points=True, show=False)
    plt.title(f"Stereographic Projection (N={N})", fontsize=14)

    print("Projections generated. Run plt.show() to view if in an interactive session.")
    # Show everything at the end
    plt.show() # pragma: no cover

if __name__ == "__main__": # pragma: no cover
    import doctest
    doctest.testmod()
    main()
