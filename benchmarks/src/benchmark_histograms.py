#!/usr/bin/env python3
"""Benchmark for vectorized histogram lookups."""

import argparse
import time

import numpy as np

from eqsp.histograms import eq_find_s2_region


def run(n_max=5000000, N=1000):
    """Run the benchmark.

    Args:
        n_max (int): Total number of points to search.
        N (int): Number of regions in the partition.
    """
    print(f"{'Points':<10} | {'Time (s)':>10}")
    print("-" * 23)

    rng = np.random.default_rng(42)

    # Generate sizes dynamically
    sizes = []
    chunk = max(1, n_max // 5)
    for i in range(chunk, n_max + 1, chunk):
        sizes.append(i)

    for size in sizes:
        # Uniformly distributed points on the sphere
        points_s = np.zeros((2, size))
        points_s[0, :] = rng.uniform(0, 2 * np.pi, size)
        points_s[1, :] = np.arccos(rng.uniform(-1, 1, size))

        t0 = time.perf_counter()
        eq_find_s2_region(points_s, N)
        t1 = time.perf_counter()
        print(f"{size:<10} | {t1 - t0:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for histogram lookups.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=80000000,
        help="Total number of points to search (default: 80,000,000).",
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=1000,
        help="Number of regions in the partition (default: 1000).",
    )
    args = parser.parse_args()
    run(n_max=args.n_max, N=args.regions)
