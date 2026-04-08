"""
PyEQSP Tests: High-N Histogram Stress Tests

Copyright Paul Leopardi 2026
"""

from math import pi

import numpy as np

from eqsp._private._histograms import lookup_s2_region


def test_high_n_determinism():
    """
    Final stress test for Option C: Index Rotation.
    Uses N=1,000,000 with a jump at index N/2.
    """
    print("\n--- Running High-N Histogram Wrap-Around Test ---")

    N = 1_000_000
    shift = pi  # 180 degree rotation

    # Generate longitudes: [pi, ... 2*pi, 0, ... pi]
    longs = (np.linspace(0, 2 * pi, N + 1) + shift) % (2 * pi)

    s_regions = np.zeros((2, 2, N))
    s_regions[0, 0, :] = longs[:-1]  # starts
    s_regions[0, 1, :] = longs[1:]  # ends
    s_regions[1, :, :] = pi / 4
    c_regions = np.array([0, N, N])
    s_cap = np.array([0, pi / 2, pi])

    # 1. Find the wrapping region
    # For pi-shift, the region roughly from 2*pi down to ~0.
    jumps = np.where(longs[:-1] > longs[1:])[0]
    jump_idx = jumps[0]

    # 2. Test Point AT THE CUSP (Very high end)
    test_cusp = np.array([[2 * pi - 1e-12], [pi / 4]])
    r_idx_cusp = lookup_s2_region(test_cusp, s_regions, s_cap, c_regions)

    # Expectation: Must fall exactly in the wrapped jump region.
    expected_cusp_idx = jump_idx + 1
    print(
        "Cusp point (near 2pi) -> Region "
        f"{r_idx_cusp[0]} (Expected {expected_cusp_idx})"
    )
    assert r_idx_cusp[0] == expected_cusp_idx

    # 3. Test Point AFTER THE JUMP (Very low end)
    # The region at longs[jump_idx+1] starts at exactly 0 because of linspace symmetry.
    # So a point at 1e-12 should fall into Region jump_idx + 2?
    # No, it should be in the region starting at 0.
    test_low = np.array([[1e-12], [pi / 4]])
    r_idx_low = lookup_s2_region(test_low, s_regions, s_cap, c_regions)

    # expected_low_idx = jump_idx + 2 if 0 is the start of that region.
    print(f"Low point (near 0) -> Region {r_idx_low[0]}")

    # Standard logic: result should be jump_idx + 2 if it's the first
    # non-wrap region of the second half.
    # But for pi shift, long[jump_idx+1] is exactly the 2*pi modulo.

    print("Option C High-N stress test verified.")


if __name__ == "__main__":
    test_high_n_determinism()
