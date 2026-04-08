#!/usr/bin/env python3
"""
release/upload_release.py

Automates TestPyPI and PyPI uploads.
Checks for credentials upfront to prevent late failures.
Runs the clean-build-check cycle (via build_dist.py) before uploading.

Usage:
    python release/upload_release.py --testpypi
    python release/upload_release.py --pypi
"""
# pylint: disable=line-too-long,missing-function-docstring,subprocess-run-check

import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_structured_diagnostic(stderr_output):
    print("\n" + "=" * 50)
    print("UPLOAD FAILED - DIAGNOSTIC INFORMATION")
    print("=" * 50)

    # 1. Twine version
    print("\n--- Twine Version ---")
    subprocess.run([sys.executable, "-m", "twine", "--version"])

    # 2. File Sizes
    print("\n--- Distribution File Sizes ---")
    for dist_file in Path("dist").glob("*"):
        size_kb = dist_file.stat().st_size / 1024
        print(f"  {dist_file.name}: {size_kb:.1f} KB")

    # 3. Full Stderr
    print("\n--- Error Output ---")
    print(stderr_output)

    # 4. Hint
    print("\n--- Troubleshooting Hint ---")
    print("If you received an Authentication Error (403), verify that your")
    print("PyPI/TestPyPI token has the correct scope (e.g. project-specific)")
    print("and has not expired.")
    print("=" * 50 + "\n")


def check_credentials():
    has_pypirc = os.path.exists(os.path.expanduser("~/.pypirc"))
    # Twine typically expects TWINE_PASSWORD (and optionally TWINE_USERNAME).
    # Project-specific tokens are often passed via TWINE_PASSWORD with
    # TWINE_USERNAME set to "__token__".
    has_env_creds = "TWINE_PASSWORD" in os.environ

    if not (has_pypirc or has_env_creds):
        print("ERROR: No PyPI credentials found.", file=sys.stderr)
        print(
            "Store them in ~/.pypirc or set TWINE_PASSWORD (and optionally "
            "TWINE_USERNAME if using a username/password pair) environment variables.",
            file=sys.stderr,
        )
        print(
            "See: https://twine.readthedocs.io/en/stable/#configuration",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload PyEQSP to TestPyPI or PyPI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--testpypi", action="store_true", help="Upload to test.pypi.org"
    )
    group.add_argument(
        "--pypi", action="store_true", help="Upload to pypi.org (Production)"
    )
    args = parser.parse_args()

    # 1. Check credentials upfront
    check_credentials()

    # 2. Run clean/build/check
    print("=== Running Pre-Upload Build Cycle ===")
    result = subprocess.run([sys.executable, "release/build_dist.py"])
    if result.returncode != 0:
        print(
            "ERROR: Pre-upload build cycle failed. Aborting upload.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Upload
    print("\n=== Uploading to " + ("TestPyPI" if args.testpypi else "PyPI") + " ===")

    dist_files = [str(p) for p in Path("dist").glob("*")]
    if not dist_files:
        print("ERROR: No files found in dist/ after build.", file=sys.stderr)
        sys.exit(1)

    twine_cmd = [sys.executable, "-m", "twine", "upload"]
    if args.testpypi:
        twine_cmd.extend(["--repository", "testpypi"])
    twine_cmd.extend(dist_files)

    print(f"Running: {' '.join(twine_cmd)}")

    upload_process = subprocess.run(twine_cmd, capture_output=True, text=True)

    if upload_process.returncode == 0:
        print(upload_process.stdout)
        print("✓ Upload successful!")
    else:  # pragma: no cover
        print_structured_diagnostic(upload_process.stderr)
        sys.exit(upload_process.returncode)


if __name__ == "__main__":  # pragma: no cover
    main()
