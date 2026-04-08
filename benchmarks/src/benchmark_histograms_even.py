#!/usr/bin/env python3
"""Benchmark for eq_find_s2_region (Symmetric/Even Collars)."""

import argparse

from benchmark_core import run_histograms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark for histograms (Symmetric)."
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=10000000,
        help="Total points to evaluate (default: 10,000,000).",
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=1000,
        help="Number of regions in the partition (default: 1000).",
    )
    args = parser.parse_args()
    run_histograms(n_max=args.n_max, n_regions=args.regions, even_collars=True)
