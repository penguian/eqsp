"""
Quick Start Example for PyEQSP

This script demonstrates the basic workflow:
1. Generating equal-area regions.
2. Generating a centre-point set.
3. Measuring geometric properties (minimum distance and Riesz energy).
"""

import eqsp


def main():
    """
    Execute the quick start demonstration.

    Example
    -------
    >>> main() # doctest: +ELLIPSIS
    --- Generating EQ partition for S^2 with N=1000 ---
    Generated 1000 regions.
    ...
    --- Analyzing geometric properties ---
    Minimum distance: 0.1049
    Riesz s-energy (s=2): 822286.8661
    """
    dim = 2
    N = 1000

    print(f"--- Generating EQ partition for S^{dim} with N={N} ---")

    # 1. Generate region boundaries
    regions = eqsp.eq_regions(dim=dim, N=N)
    print(f"Generated {regions.shape[2]} regions.")

    # 2. Generate point set (centre points of regions)
    points = eqsp.eq_point_set(dim=dim, N=N)
    print(f"Point set shape: {points.shape}")  # Should be (dim+1, N)

    # 3. Analyze separation quality
    print("\n--- Analyzing geometric properties ---")

    # Measure minimum Euclidean distance between points
    min_dist = eqsp.point_set_min_dist(points)
    print(f"Minimum distance: {min_dist:.4f}")

    # Measure Riesz s-energy (uniformity metric)
    energy, _ = eqsp.point_set_energy_dist(points, s=2)
    print(f"Riesz s-energy (s=2): {energy:.4f}")


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
    main()
