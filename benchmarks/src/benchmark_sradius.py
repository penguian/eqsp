#!/usr/bin/env python3
"""Benchmark for sradius_of_cap (Root finding loop bottleneck)."""

import argparse

from benchmark_core import run_sradius

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for sradius.")
    parser.add_argument(
        "--dim", type=int, default=3, help="Manifold dimension d+1 (default: 3)."
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=10000000,
        help="Total points to evaluate (default: 10,000,000).",
    )
    parser.add_argument(
        "--even-collars",
        action="store_true",
        default=False,
        help="Use even number of collars for symmetric partitions.",
    )
    args = parser.parse_args()
    run_sradius(dim=args.dim, n_max=args.n_max, _even_collars=args.even_collars)
