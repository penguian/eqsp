"""Benchmark for eq_regions (Python loop overhead)."""

import argparse
import time

from eqsp.partitions import eq_regions


def run(n_max=8000, dim=2):
    """Run the benchmark.

    Args:
        n_max (int): Total number of regions to evaluate up to.
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
        for N in range(start, end + 1):
            eq_regions(dim, N)
        t1 = time.perf_counter()
        print(f"{f'{start}-{end}':<15} | {t1 - t0:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for eq_regions.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=16000,
        help="Total number of regions to evaluate up to (default: 16000).",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Sphere dimension (default: 2)."
    )
    args = parser.parse_args()
    run(n_max=args.n_max, dim=args.dim)
