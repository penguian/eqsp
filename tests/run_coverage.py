#!/usr/bin/env python3
"""
Run project tests with coverage analysis.

Reproduces the logic of tests/run_coverage.sh in Python.
Supports optional inclusion of private implementation tests.

Usage:
    python3 tests/run_coverage.py [--include-private]
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd, env=None):
    """Run a system command and exit if it fails."""
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{' '.join(cmd)}' failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    """Main execution logic for coverage analysis."""
    parser = argparse.ArgumentParser(description="Run eqsp tests with coverage.")
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include tests for private implementation modules.",
    )
    args = parser.parse_args()

    # Set up environment: ensure current directory is in PYTHONPATH
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{os.getcwd()}:{current_pythonpath}"
    else:
        env["PYTHONPATH"] = os.getcwd()

    # Base pytest options
    # --doctest-modules: run doctests in modules
    # --ignore=eqsp/_private: ignore the private package itself by default
    pytest_opts = ["--doctest-modules", "--ignore=eqsp/_private"]

    if args.include_private:
        print("Running coverage including private tests...")
    else:
        print("Running coverage excluding private tests...")
        # Ignore specific private test files in tests/src
        pytest_opts.extend(
            [
                "--ignore=tests/src/test_private_histograms.py",
                "--ignore=tests/src/test_private_partitions.py",
                "--ignore=tests/src/test_private_region_props.py",
            ]
        )

    # Construct the coverage run command
    # -m coverage run --source=eqsp: measure coverage for the eqsp package
    # -m pytest eqsp tests/src: run tests in both the package and tests/src
    coverage_run = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--source=eqsp",
        "-m",
        "pytest",
        "eqsp",
        "tests/src",
    ] + pytest_opts

    # Run the coverage measurement
    run_command(coverage_run, env=env)

    # Run the coverage report
    print("\nCoverage Report:\n")
    coverage_report = [sys.executable, "-m", "coverage", "report"]
    run_command(coverage_report, env=env)


if __name__ == "__main__":
    main()
