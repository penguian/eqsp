#!/usr/bin/env python3
"""
doc/ci_scripts/upload_sourceforge.py

Automates the SourceForge documentation hosting step:
1. Runs `make html` inside doc/.
2. Derives the current version from pyproject.toml.
3. Prints the full `scp` command to push doc/_build/html/ to SourceForge.

Usage:
    python doc/ci_scripts/upload_sourceforge.py
"""
# pylint: disable=line-too-long,missing-function-docstring,subprocess-run-check

import os
import subprocess
import sys
import tomllib


def get_version():
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        print(
            f"ERROR: {pyproject_path} not found. Run from the project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    return config["project"]["version"]


def main():
    print("=== Building Sphinx Documentation ===")
    # Run make html inside the doc directory
    result = subprocess.run(["make", "-C", "doc", "html"])
    if result.returncode != 0:
        print("ERROR: Sphinx build failed. Aborting.", file=sys.stderr)
        sys.exit(result.returncode)

    version = get_version()

    print("\n=== Build Successful ===")
    print(f"Documentation for version {version} is ready in doc/_build/html/")
    print("\nTo upload to SourceForge, review and run the following command")
    print("(replace 'USER' with your SourceForge username):")
    print("-" * 70)
    print(
        "scp -r doc/_build/html/* "
        "USER@web.sourceforge.net:/home/project-web/eqsp/htdocs/"
    )
    print("-" * 70)


if __name__ == "__main__":
    main()
