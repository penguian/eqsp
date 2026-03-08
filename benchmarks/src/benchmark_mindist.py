#!/usr/bin/env python3
"""Benchmark for eq_min_dist (Efficient distance calculation)."""

import argparse
import time

from eqsp.point_set_props import eq_min_dist


def run(n_max=16000, dim=2, even_collars=False):
    """Run the benchmark.

    Args:
        n_max (int): Maximum number of regions N to evaluate.
        dim (int): Sphere dimension.
    """
    print(f"{'N range':<15} | {'Time (s)':>10}")
    print("-" * 28)

    # Split n_max into 5 intervals
    chunk_size = max(1, n_max // 5)
    ranges = []
    for i in range(0, n_max, chunk_size):
        ranges.append((i + 1, min(i + chunk_size, n_max)))

    for start, end in ranges:
        if start > end:
            continue
        t0 = time.perf_counter()
        if even_collars:
            even_start = start if start % 2 == 0 else start + 1
            n_range = range(even_start, end + 1, 2)
        else:
            n_range = range(start, end + 1)
        for N in n_range:
            eq_min_dist(dim, N, even_collars=even_collars)
        t1 = time.perf_counter()
        print(f"{f'{start}-{end}':<15} | {t1 - t0:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for eq_min_dist.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=16000,
        help="Maximum number of regions N (default: 16000).",
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
