"""
Symmetric Partition Example for PyEQSP

This script demonstrates how to generate partitions that are perfectly
symmetric about the sphere's equator using the 'even_collars' parameter.
"""

from eqsp.partitions import eq_point_set, eq_regions


def main():
    """
    Demonstrate symmetric partitions.

    Example
    -------
    >>> main()
    --- Generating Symmetric Partition (N=100) ---
    Successfully generated symmetric partition with 100 regions.
    <BLANKLINE>
    Symmetry distribution:
      Northern points: 50
      Southern points: 50
      Equatorial points: 0
    """
    dim = 2
    N = 100  # N must be even for even_collars=True

    print(f"--- Generating Symmetric Partition (N={N}) ---")

    # Force a symmetric partition
    # This ensures a collar boundary aligns exactly with z=0
    regions = eq_regions(dim=dim, N=N, even_collars=True)
    points = eq_point_set(dim=dim, N=N, even_collars=True)

    print(
        f"Successfully generated symmetric partition with {regions.shape[2]} regions."
    )

    # Basic check for symmetry: Half the points should be in the northern hemisphere
    z_coords = points[2, :]
    northern = (z_coords > 0).sum()
    southern = (z_coords < 0).sum()
    equatorial = (z_coords == 0).sum()  # Boundary points

    print("\nSymmetry distribution:")
    print(f"  Northern points: {northern}")
    print(f"  Southern points: {southern}")
    print(f"  Equatorial points: {equatorial}")


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
    main()
