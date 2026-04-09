#!/usr/bin/env python3
"""
Unified verification script for the PyEQSP project.

This script orchestrates the entire quality control suite including:
- Ruff linter for style and formatting.
- Pylint for deep static analysis.
- Documentation link checks and quality policy audits.
- Sphinx doctest and HTML build with zero-warning enforcement.
- Pytest suite with 100% coverage requirements.
- Pre-release distribution build checks.

Usage:
    python3 validation/verify_all.py [options]

See --help for available environment orchestration options.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(command, name, python_bin_dir, base_env):
    """Run a single verification step and exit on failure."""
    print("========================================")
    print(f"Running {name}...")
    print("========================================")

    # Ensure the target python's bin/ directory is in the PATH
    # so that 'make' can find 'python3' and 'sphinx-build' correctly.
    env = base_env.copy()
    path_list = [python_bin_dir]
    old_path = env.get("PATH", "")
    if old_path:
        path_list.append(old_path)
    env["PATH"] = os.pathsep.join(path_list)

    # Set up PYTHONPATH for development environment
    # This ensures that the codebase under test and the benchmark scripts
    # can resolve their relative imports.
    repo_root = str(REPO_ROOT)
    bench_src = str(REPO_ROOT / "benchmarks" / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    new_paths = [repo_root, bench_src]
    if current_pythonpath:
        new_paths.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(new_paths)

    result = subprocess.run(command, check=False, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"\n[FAILED] {name}\n")
        sys.exit(result.returncode)
    print(f"[PASSED] {name}\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified verification script for PyEQSP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python3 validation/verify_all.py
  python3 validation/verify_all.py --venv VENV --uninstall
  python3 validation/verify_all.py --pre-release

Notes:
  --venv DIR    Activates the specified virtual environment. If a venv is already
                active, it is effectively deactivated for the verification steps
                to ensure audit integrity.
  --uninstall   Uninstalls 'pyeqsp' from the target environment before running
                verification. This prevents stale installations in site-packages
                from shadowing the local source code under test.
""",
    )
    parser.add_argument(
        "--pre-release",
        action="store_true",
        help="Run extra pre-release checks (build distribution).",
    )
    parser.add_argument(
        "--venv",
        metavar="DIR",
        help="Location of a virtual environment to activate for verification.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall pyeqsp from the target environment before starting.",
    )
    return parser.parse_args()


def main():
    """Execute all verification steps."""
    args = parse_args()

    # Base environment setup
    base_env = os.environ.copy()
    py = sys.executable

    # "Deactivate" current venv if it exists.
    # This involves stripping the active venv's bin from PATH and cleaning env vars.
    active_venv = base_env.get("VIRTUAL_ENV")
    if active_venv:
        active_bin = str(Path(active_venv) / "bin")
        paths = base_env.get("PATH", "").split(os.pathsep)
        # Strip the current venv's bin so it doesn't shadow system tools
        new_paths = [p for p in paths if p != active_bin]
        base_env["PATH"] = os.pathsep.join(new_paths)
        base_env.pop("VIRTUAL_ENV", None)
        base_env.pop("PYTHONHOME", None)

        # If we were in a venv, sys.executable is the venv's python.
        # We find the next python3 in the cleaned PATH to ensure audit safety.
        base_py = shutil.which("python3", path=base_env["PATH"])
        if base_py:
            py = base_py

    # Activate requested venv
    if args.venv:
        venv_path = Path(args.venv).resolve()
        if not venv_path.exists():
            print(f"Error: Virtual environment not found at {args.venv}")
            sys.exit(1)

        py = str(venv_path / "bin" / "python3")
        if not os.path.exists(py):
            py = str(venv_path / "bin" / "python")

        if not os.path.exists(py):
            print(f"Error: Python executable not found in venv at {venv_path / 'bin'}")
            sys.exit(1)

        base_env["VIRTUAL_ENV"] = str(venv_path)
        venv_bin = str(venv_path / "bin")
        base_env["PATH"] = os.pathsep.join([venv_bin, base_env.get("PATH", "")])

    python_bin_dir = str(Path(py).parent)

    # Perform uninstallation if requested
    if args.uninstall:
        print("========================================")
        print("Uninstalling pyeqsp...")
        print("========================================")
        subprocess.run(
            [py, "-m", "pip", "uninstall", "-y", "pyeqsp"],
            check=False,
            env=base_env,
            cwd=REPO_ROOT,
        )
        print("[DONE] Uninstallation attempted.\n")

    steps = [
        (
            [
                py,
                "-m",
                "ruff",
                "check",
                "eqsp",
                "tests",
                "examples/phd-thesis",
                "examples/user-guide/src",
                "benchmarks",
                "validation",
                "release",
            ],
            "Ruff Linter",
        ),
        (
            [
                py,
                "-m",
                "pylint",
                "--init-hook=import os, sys; "
                "sys.path.extend(['.', os.path.join('benchmarks', 'src')])",
                "eqsp",
                "tests",
                "examples/phd-thesis",
                "examples/user-guide/src",
                "benchmarks",
                "validation",
                "release",
            ],
            "Pylint",
        ),
        (
            [py, "validation/check_links.py"],
            "Documentation Link Check",
        ),
        (
            [py, "validation/quality_check.py"],
            "Performance Quality Check",
        ),
        (["make", "-C", "doc", "doctest"], "Sphinx Doctest"),
        (
            ["make", "-C", "doc", "html", "SPHINXOPTS=-W"],
            "Sphinx HTML Build (Zero Warning Policy)",
        ),
        (
            [py, "tests/run_coverage.py", "--include-private"],
            "Test Suite & Coverage",
        ),
    ]

    if args.pre_release:
        steps.append(
            ([py, "release/build_dist.py"], "Pre-release Package Build & Check")
        )

    for cmd, name in steps:
        run_step(cmd, name, python_bin_dir, base_env)

    print("All verification steps passed successfully!")


if __name__ == "__main__":  # pragma: no cover
    main()
