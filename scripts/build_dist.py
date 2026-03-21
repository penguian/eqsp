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

import os
import shutil
import subprocess
import sys
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
        [sys.executable, "scripts/pypi_readme_fix.py"],
        "Generating README_dist.md with absolute links",
    )

    # 2. Clean old artifacts
    clean_build_artifacts()

    # 3. Build distribution (sdist and wheel)
    # We swap README.md with README_dist.md temporarily so that the build
    # system picks up the absolute links for the long_description.
    print("=== Swapping README for build ===")
    readme_orig = "README.md"
    readme_dist = "README_dist.md"
    readme_temp = "README_backup_orig.md"

    if os.path.exists(readme_orig):
        shutil.move(readme_orig, readme_temp)
    shutil.copy(readme_dist, readme_orig)

    try:
        run_command([sys.executable, "-m", "build"], "Building distribution")
    finally:
        print("=== Restoring original README ===")
        if os.path.exists(readme_temp):
            shutil.move(readme_temp, readme_orig)
        # We keep README_dist.md for manual inspection

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
