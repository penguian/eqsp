"""
3D Visualization Example for PyEQSP

This script demonstrates interactive 3D rendering of EQ partitions
using Mayavi.

NOTE: This requires the 'mayavi' and 'PyQt5' packages.
"""

import sys


def main(): # pragma: no cover
    """
    Launch 3D visualization (interactive).

    Example
    -------
    >>> # Skip this in automated tests to avoid opening windows
    >>> pass
    """
    try:
        from eqsp import visualizations  # pylint: disable=import-outside-toplevel
    except ImportError: # pragma: no cover
        print("Error: Mayavi or PyQt5 not found.")
        print("Please install with: pip install 'pyeqsp[mayavi]'")
        sys.exit(1)

    N = 100

    print(f"--- Launching 3D Visualization with N={N} ---")
    print("Rotate with your mouse, zoom with the scroll wheel.")

    # Show a 3D partition of N regions with centre points
    # This will open an interactive Mayavi window
    visualizations.show_s2_partition(
        N, show_points=True, show_sphere=True
    ) # pragma: no cover

if __name__ == "__main__": # pragma: no cover
    import doctest
    doctest.testmod()
    main()
