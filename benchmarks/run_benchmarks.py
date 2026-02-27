#!/usr/bin/env python3
"""Main script to run all performance benchmarks and log results."""

import argparse
import os
import subprocess
import sys
import time


def run_benchmark(name, script_name, extra_args, env, results_dir, base_dir):
    """Run a single benchmark script via subprocess and log its output."""
    script_path = os.path.join(base_dir, "src", script_name)
    log_file = os.path.join(results_dir, script_name.replace(".py", ".log"))

    print(f"Running benchmark: {name}")

    cmd = [sys.executable, script_path] + extra_args

    try:
        t0 = time.perf_counter()
        # Capture output for both logging and displaying
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        t_elapsed = time.perf_counter() - t0

        # Write to individual log file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(process.stdout)

        # Also print to stdout (captured if the whole thing is redirected)
        print(process.stdout)
        return t_elapsed

    except subprocess.CalledProcessError as e:
        print(f"Error: Benchmark {name} failed with exit code {e.returncode}")
        print(e.stderr)
        return 0


def main():
    """Main execution logic for benchmarks."""
    parser = argparse.ArgumentParser(description="EQSP Performance Benchmarks.")
    parser.add_argument(
        "--n-max",
        type=int,
        help="General n_max to override defaults for all benchmarks.",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Sphere dimension (default: 2)."
    )
    parser.add_argument(
        "--s", type=float, help="Exponent for energy distance benchmark."
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=1000,
        help="Number of regions for histogram benchmark (default: 1000).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to save log files (default: benchmarks/results).",
    )

    args = parser.parse_args()

    # Ensure results directory exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir
    if not results_dir:
        results_dir = os.path.join(base_dir, "results")

    results_dir = os.path.abspath(results_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set up environment: ensure root directory is in PYTHONPATH
    # Since we are in benchmarks/, the root is one level up
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{root_dir}:{current_pythonpath}"
    else:
        env["PYTHONPATH"] = root_dir

    # Define benchmarks and their arguments
    # Note: we use script arguments now instead of function kwargs
    benchmarks = [
        (
            "eq_area_error (Redundant area calculation)",
            "benchmark_area_error.py",
            ["--n-max", str(args.n_max or 15000), "--dim", str(args.dim)],
        ),
        (
            "point_set_energy_dist (O(N^2) memory & broadcasting)",
            "benchmark_energy_dist.py",
            ["--n-max", str(args.n_max or 2400), "--dim", str(args.dim)]
            + (["--s", str(args.s)] if args.s else []),
        ),
        (
            "eq_regions (Python loop overhead)",
            "benchmark_eq_regions.py",
            [
                "--max-k", str(int(os.getenv("MAX_K", "22"))),
                "--max-d", str(args.dim),
                "--iterations", "1"
            ],
        ),
        (
            "eq_min_dist (Efficient distance calculation)",
            "benchmark_mindist.py",
            ["--n-max", str(args.n_max or 6400), "--dim", str(args.dim)],
        ),
        (
            "eq_find_s2_region (Vectorized histogram lookup)",
            "benchmark_histograms.py",
            ["--n-max", str(args.n_max or 200000000), "--regions", str(args.regions)],
        ),
        (
            "sradius_of_cap (Root finding loop bottleneck)",
            "benchmark_sradius.py",
            ["--n-max", str(args.n_max or 100000000), "--dim", str(args.dim + 1)],
        ),
    ]

    print("=======================================")
    print("      EQSP Performance Benchmarks      ")
    print("=======================================\n")

    t_start = time.perf_counter()

    for name, script, extra_args in benchmarks:
        run_benchmark(name, script, extra_args, env, results_dir, base_dir)

    t_end = time.perf_counter()
    print("=======================================")
    print(f"Total benchmark time: {t_end - t_start:.2f} seconds")
    print("=======================================")


if __name__ == "__main__":
    main()
