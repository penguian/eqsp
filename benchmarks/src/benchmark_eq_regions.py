#!/usr/bin/env python3
"""Benchmark for the eq_regions function, aligned with Thesis Section 3.10.2."""

import argparse
import time

import numpy as np

from eqsp.partitions import eq_regions


def run_benchmark(max_d=11, max_k=22, iterations=3, show_progress=False):
    """Run the eq_regions benchmark matching thesis parameters."""
    print(f"Running Thesis Benchmark for eq_regions: d=[1..{max_d}], N=2^[1..{max_k}]")
    print(f"Iterations per point: {iterations}")
    if show_progress:
        print(f"{'d':<3} | {'k':<3} | {'N':<10} | {'Mean Time (s)':>15}")
        print("-" * 45)

    results = []

    for d in range(1, max_d + 1):
        for k in range(1, max_k + 1):
            N = 2**k
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                eq_regions(d, N)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            mean_time = np.mean(times)
            results.append((d, k, N, mean_time))
            if show_progress:
                print(f"{d:<3} | {k:<3} | {N:<10} | {mean_time:>15.6f}")

    return results


def run(n_max=16000, dim=2):
    """Bridge for run_benchmarks.py to benchmark eq_regions.

    Translates linear n_max to k powers of 2 for compatibility.
    """
    max_k = int(np.log2(n_max)) if n_max > 0 else 1
    results = run_benchmark(max_d=dim, max_k=max_k, iterations=1, show_progress=False)
    analyze_scaling(results)


def analyze_scaling(results):
    """Analyze the O(N^x) scaling for eq_regions using log-log regression."""
    print("\nScaling Analysis (t ~ N^x) for eq_regions:")

    # Analyze by dimension
    dimensions = sorted(list(set(r[0] for r in results)))
    for d in dimensions:
        d_results = [r for r in results if r[0] == d]
        if len(d_results) < 2:
            continue

        # log(t) = log(C) + x * log(N)
        log_n = np.log([r[2] for r in d_results])
        log_t = np.log([r[3] for r in d_results])

        # Linear regression for x
        x, _ = np.polyfit(log_n, log_t, 1)
        print(f"Dimension {d:<2}: x = {x:.4f} (Thesis baseline: ~0.60)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replicate Thesis Benchmark for eq_regions."
    )
    parser.add_argument(
        "--max-d",
        type=int,
        default=4,
        help="Maximum dimension to test eq_regions (default: 4).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=18,
        help="Maximum power of 2 for N in eq_regions (default: 18).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per point (default: 3).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=False,
        help="Show per-point benchmark results (default: False).",
    )
    args = parser.parse_args()

    start_total = time.perf_counter()
    data = run_benchmark(
        max_d=args.max_d,
        max_k=args.max_k,
        iterations=args.iterations,
        show_progress=args.show_progress,
    )
    end_total = time.perf_counter()

    analyze_scaling(data)

    print(f"\nTotal benchmark wall time: {end_total - start_total:.2f} seconds")
