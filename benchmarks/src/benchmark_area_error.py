#!/usr/bin/env python3
"""Benchmark for eq_area_error (Redundant area calculation)."""

import argparse
import time

import numpy as np

from eqsp.region_props import eq_area_error


def run(n_max=5000, dim=2, even_collars=False):
    """Run the benchmark.

    Args:
        n_max (int): Total number of regions to evaluate up to.
        dim (int): Sphere dimension.
    """
    print(f"{'N range':<15} | {'Time (s)':>10}")
    print("-" * 28)

    # Split n_max into 5 intervals for consistent output formatting
    chunk_size = max(1, n_max // 5)
    ranges = []
    for i in range(0, n_max, chunk_size):
        ranges.append((i + 1, min(i + chunk_size, n_max)))

    for start, end in ranges:
        if start > end:
            continue
        if even_collars:
            # Only use even N
            even_start = start if start % 2 == 0 else start + 1
            N_array = np.arange(even_start, end + 1, 2)
        else:
            N_array = np.arange(start, end + 1)
        if len(N_array) == 0:
            continue
        t0 = time.perf_counter()
        eq_area_error(dim, N_array, even_collars=even_collars)
        t1 = time.perf_counter()
        print(f"{f'{start}-{end}':<15} | {t1 - t0:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for eq_area_error.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=15000,
        help="Total number of regions to evaluate up to (default: 15000).",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Sphere dimension (default: 2)."
    )
    parser.add_argument(
        "--even-collars",
        action="store_true",
        default=False,
        help="Use even number of collars for symmetric partitions.",
    )
    args = parser.parse_args()
    run(n_max=args.n_max, dim=args.dim, even_collars=args.even_collars)
