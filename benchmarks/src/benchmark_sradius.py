#!/usr/bin/env python3
"""Benchmark for sradius_of_cap (Root finding loop bottleneck)."""

import argparse

from benchmark_core import run_sradius

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for sradius.")
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        help="Sphere manifold dimension d (e.g., S^d) (default: 3).",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=10000000,
        help="Maximum value for N (default: 10^7).",
    )
    args = parser.parse_args()
    run_sradius(args.n_max, dim=args.dim)
