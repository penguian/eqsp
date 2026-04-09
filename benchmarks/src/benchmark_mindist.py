#!/usr/bin/env python3
"""Benchmark for eq_min_dist (KDTree-based spatial search)."""

import argparse

from benchmark_core import run_mindist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for eq_min_dist.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=10000000,
        help="Total points to evaluate (default: 10,000,000).",
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
    run_mindist(n_max=args.n_max, dim=args.dim, even_collars=args.even_collars)
