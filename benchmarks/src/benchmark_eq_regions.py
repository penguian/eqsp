#!/usr/bin/env python3
"""Benchmark for eq_regions (Recursive zonal partitioning)."""

import argparse

from benchmark_core import run_eq_regions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for eq_regions.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=100000000,
        help="Total number of regions to evaluate up to (default: 100,000,000).",
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
    run_eq_regions(n_max=args.n_max, dim=args.dim, even_collars=args.even_collars)
