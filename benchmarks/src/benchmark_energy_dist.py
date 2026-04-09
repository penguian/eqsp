#!/usr/bin/env python3
"""Benchmark for point_set_energy_dist (Riesz Energy calculation)."""

import argparse

from benchmark_core import run_energy_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for Riesz energy.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=50000,
        help="Total points to evaluate (default: 50,000).",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Sphere dimension (default: 2)."
    )
    parser.add_argument(
        "--s", type=float, default=None, help="Riesz exponent s (default: d-1)."
    )
    parser.add_argument(
        "--even-collars",
        action="store_true",
        default=False,
        help="Use even number of collars for symmetric partitions.",
    )
    args = parser.parse_args()
    run_energy_dist(
        n_max=args.n_max, dim=args.dim, s=args.s, even_collars=args.even_collars
    )
