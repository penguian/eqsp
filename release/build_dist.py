#!/usr/bin/env python3
"""
scripts/build_dist.py

Orchestrates the clean-build-check cycle for PyEQSP distribution.
1. Calls pypi_readme_fix.py to produce README_dist.md
2. Removes old dist/, build/, and *.egg-info directories
3. Runs python -m build
4. Runs twine check dist/*

Exits non-zero if any step fails.

Usage:
    python scripts/build_dist.py
"""
# pylint: disable=line-too-long,missing-function-docstring,subprocess-run-check

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, description):
    print(f"=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"ERROR: {description} failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print("✓ Success\n")


def clean_build_artifacts():
    print("=== Cleaning build artifacts ===")
    artifacts = ["dist", "build"]
    for path in Path(".").glob("*.egg-info"):
        artifacts.append(str(path))

    for item in artifacts:
        if os.path.exists(item):
            print(f"Removing {item}/")
            shutil.rmtree(item)
    print("✓ Cleaned\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean, build, and check PyEQSP distribution."
    )
    parser.add_argument(
        "--keep-on-fail",
        action="store_true",
        help="Keep stale dist/ and build/ directories on failure",
    )
    args = parser.parse_args()

    # Ensure we run from the project root
    if not os.path.exists("pyproject.toml"):
        print(
            "ERROR: Must be run from the repository root (where pyproject.toml "
            "is located).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Fix README for PyPI
    run_command(
        [sys.executable, "release/pypi_readme_fix.py"],
        "Generating README_dist.md with absolute links",
    )

    # 2. Clean old artifacts
    clean_build_artifacts()

    # 3. Build distribution (sdist and wheel)
    # We swap README.md with README_dist.md temporarily so that the build
    # system picks up the absolute links for the long_description.
    # We use a unique temporary file to ensure the move is atomic and safe.
    print("=== Swapping README for build ===")
    readme_orig = Path("README.md")
    readme_dist = Path("README_dist.md")

    # Generate a unique temporary path in the current directory
    fd, temp_path_str = tempfile.mkstemp(dir=".", prefix="README_orig_", suffix=".md")
    os.close(fd)
    readme_temp = Path(temp_path_str)

    if readme_orig.exists():
        shutil.move(readme_orig, readme_temp)
    shutil.copy(readme_dist, readme_orig)

    try:
        try:
            run_command([sys.executable, "-m", "build"], "Building distribution")
        except (Exception, SystemExit):
            if not args.keep_on_fail:
                print(
                    "ERROR: Build failed. Cleaning up stale artifacts "
                    "(use --keep-on-fail to prevent this).",
                    file=sys.stderr,
                )
                clean_build_artifacts()
            raise
    finally:
        print("=== Restoring original README ===")
        if readme_temp.exists():
            shutil.move(readme_temp, readme_orig)
        if readme_dist.exists():
            readme_dist.unlink()

    # 4. Check distribution with twine
    dist_files = [str(p) for p in Path("dist").glob("*")]
    if not dist_files:
        print("ERROR: No files found in dist/", file=sys.stderr)
        sys.exit(1)

    run_command(
        [sys.executable, "-m", "twine", "check"] + dist_files,
        "Checking distribution with twine",
    )

    print("=== Build Cycle Complete ===")
    print("The packages in dist/ are ready for upload.")


if __name__ == "__main__":
    main()
