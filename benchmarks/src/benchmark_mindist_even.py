#!/usr/bin/env python3
"""Benchmark for eq_min_dist (Symmetric/Even Collars)."""

import argparse

from benchmark_core import run_mindist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark for eq_min_dist (Symmetric)."
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=10000000,
        help="Total points to evaluate (default: 10,000,000).",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Sphere dimension (default: 2)."
    )
    args = parser.parse_args()
    run_mindist(n_max=args.n_max, dim=args.dim, even_collars=True)
