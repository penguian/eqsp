#!/usr/bin/env python3
"""Main script to run symmetric (even-collar) performance benchmarks and log results."""

import argparse
import os
import subprocess
import sys
import time


class Tee:
    """Redirect stdout to both console and a file."""

    def __init__(self, filename, mode="w"):
        # pylint: disable=consider-using-with
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        """Write data to both streams."""
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        """Flush both streams."""
        self.stdout.flush()
        self.file.flush()


def run_benchmark(
    name, script_name, *, extra_args, env, results_dir, base_dir, log_suffix=""
):
    """Run a single benchmark script via subprocess and log its output."""
    script_path = os.path.join(base_dir, "src", script_name)
    log_file = os.path.join(
        results_dir, script_name.replace(".py", f"{log_suffix}.log")
    )

    print(f"\nRunning benchmark: {name}")

    cmd = [sys.executable, script_path] + extra_args

    try:
        t0 = time.perf_counter()
        # Capture output for both logging and displaying
        process = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=True
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
    parser = argparse.ArgumentParser(
        description="PyEQSP Symmetric Performance Benchmarks."
    )
    parser.add_argument(
        "--n-max",
        type=int,
        help="General n_max to override defaults for all benchmarks.",
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
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([root_dir, current_pythonpath])
    else:
        env["PYTHONPATH"] = root_dir

    # Even-collar run: always force even collars
    suffix = "_even"
    even_args: list[str] = ["--even-collars"]

    # Define benchmarks and their arguments
    benchmarks = [
        (
            "eq_area_error",
            "benchmark_area_error.py",
            [
                "--n-max",
                str(args.n_max or 100000000),
            ]
            + even_args,
        ),
        (
            "eq_regions",
            "benchmark_eq_regions.py",
            [
                "--n-max",
                str(args.n_max or 100000000),
            ]
            + even_args,
        ),
        (
            "eq_find_s2_region",
            "benchmark_histograms.py",
            ["--n-max", str(args.n_max or 10000000), "--regions", str(args.regions)]
            + even_args,
        ),
        (
            "eq_min_dist",
            "benchmark_mindist.py",
            [
                "--n-max",
                str(args.n_max or 10000000),
            ]
            + even_args,
        ),
        (
            "point_set_energy_dist",
            "benchmark_energy_dist.py",
            [
                "--n-max",
                str(args.n_max or 50000),
            ]
            + (["--s", str(args.s)] if args.s else [])
            + even_args,
        ),
    ]

    # Main log file for the run
    main_log_file = os.path.join(results_dir, f"run_benchmarks{suffix}.log")

    with Tee(main_log_file):
        print("=======================================")
        print("    PyEQSP Symmetric Benchmarks")
        print("=======================================")
        print("\nHardware: AMD Ryzen 7 8840HS w/ Radeon 780M Graphics (~2.4 GHz)")
        print("OS:       Linux")
        print(f"Software: Python {sys.version.split()[0]}\n")

        t_start = time.perf_counter()

        for name, script, extra_args in benchmarks:
            run_benchmark(
                name,
                script,
                extra_args=extra_args,
                env=env,
                results_dir=results_dir,
                base_dir=base_dir,
                log_suffix=suffix,
            )

        t_end = time.perf_counter()
        print("\n=======================================")
        print(f"Total benchmark time: {t_end - t_start:.2f} seconds")
        print("=======================================")

    print(f"\nResults saved to {main_log_file}")


if __name__ == "__main__":
    main()
