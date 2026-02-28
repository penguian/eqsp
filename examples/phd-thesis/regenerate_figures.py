#!/usr/bin/env python3
"""
Regenerate all PhD thesis figures for the eqsp package.

This script scans the 'src' directory for fig_*.py scripts and executes them
to produce PNG files in the 'results' directory. It automatically handles
the environment requirements for 3D Mayavi plots.

Usage:
    python3 regenerate_figures.py [--two-d-only] [--force]
                                [--figure FIG_NAME] [--show-progress]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# --- Configuration ---
QT_ENV = {
    "QT_API": "pyqt5",
    "QT_QPA_PLATFORM": "xcb",
}


def format_duration(seconds):
    """Format duration in H:M:S."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def is_3d_script(script_path):
    """Check if a script imports Mayavi."""
    try:
        content = script_path.read_text(encoding="utf-8")
        return "mayavi" in content or "mlab" in content
    except (OSError, UnicodeDecodeError):
        return False


def run_one_script(script, config, args):
    """Run a single figure generation script."""
    png_name = script.with_suffix(".png").name
    target_png = config["results_dir"] / png_name

    needs_3d = is_3d_script(script)

    # Skip logic
    if (needs_3d and args.two_d_only) or (target_png.exists() and not args.force):
        label = "SKIPPED (3D)" if needs_3d and args.two_d_only else "SKIPPED (exists)"
        print(f"{script.name:40s} : {label}")
        return "skip"

    # Environment Selection
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(config["project_root"])

    python_exe = sys.executable
    mode_label = "[2D]"
    if needs_3d:
        if not config["venv_python"].exists():
            msg = (
                f"{script.name:40s} : FAILED "
                f"(venv_sys not found at {config['venv_python']})"
            )
            print(msg)
            return "fail"
        python_exe = str(config["venv_python"])
        run_env.update(QT_ENV)
        mode_label = "[3D]"

    print(f"Running {mode_label} {script.name}...", end="", flush=True)
    cmd = [python_exe, str(config["run_patched"]), str(script)]
    if args.show_progress:
        cmd.append("--show-progress")

    if args.dry_run:
        print(" DRY RUN")
        return "success"

    return execute_and_report(cmd, config["results_dir"], run_env, args)


def execute_and_report(cmd, results_dir, run_env, args):
    """Execute the command and return the result status."""
    script_start = time.time()
    try:
        with subprocess.Popen(
            cmd,
            cwd=str(results_dir),
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            output_lines = []
            if args.show_progress:
                print(" [OUTPUT BEGIN]")
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        print(f"  {line}", end="", flush=True)
                        output_lines.append(line)
                print(" [OUTPUT END]")
            else:
                stdout, _ = process.communicate(timeout=1200)
                output_lines = [stdout]

            rc = process.poll()
            duration_str = format_duration(time.time() - script_start)

            if rc == 0:
                print(f" DONE ({duration_str})")
                return "success"

            print(f" FAILED (code {rc}) after {duration_str}")
            if not args.show_progress:
                print(f"Error output:\n{''.join(output_lines)}")
            return "fail"
    except (subprocess.SubprocessError, OSError) as e:
        print(f" ERROR after {format_duration(time.time() - script_start)}: {str(e)}")
        return "fail"


def main():
    """Main execution logic for figure regeneration."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--two-d-only", action="store_true", help="Skip 3D Mayavi plots"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing PNG files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without executing"
    )
    parser.add_argument(
        "--figure",
        type=str,
        help="Regenerate specified figure (e.g., fig_3_1_partition_s2_33.py or 3_1)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show progress (default: %(default)s)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="Hide progress",
    )
    args = parser.parse_args()

    # Directory setup
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "src"
    config = {
        "results_dir": base_dir / "results",
        "project_root": base_dir.parent.parent,
        "venv_python": base_dir.parent.parent / "venv_sys" / "bin" / "python3",
        "run_patched": base_dir / "src" / "run_patched.py",
    }
    config["results_dir"].mkdir(parents=True, exist_ok=True)

    scripts = sorted(src_dir.glob("fig_*.py"))
    if args.figure:
        scripts = filter_scripts(scripts, src_dir, args.figure)

    print("--- EQSP Figure Regeneration ---")
    print(f"Source: {src_dir}")
    print(f"Target: {config['results_dir']}")
    print(f"Mode: {'2D-only' if args.two_d_only else 'Full'}")
    print("-" * 32)

    total_start = time.time()
    counts = {"success": 0, "fail": 0, "skip": 0}

    for script in scripts:
        result = run_one_script(script, config, args)
        counts[result] += 1

    total_duration = time.time() - total_start
    print("-" * 32)
    print(
        f"Summary: {counts['success']} succeeded, "
        f"{counts['fail']} failed, {counts['skip']} skipped."
    )
    print(f"Total time taken: {format_duration(total_duration)}")


def filter_scripts(scripts, src_dir, target):
    """Filter scripts based on user input."""
    if not target.endswith(".py"):
        matches = [s for s in scripts if target in s.name]
        if not matches:
            print(f"Error: No script found matching '{target}'")
            sys.exit(1)
        return matches

    script_path = src_dir / target
    if not script_path.exists():
        print(f"Error: Script {target} not found in {src_dir}")
        sys.exit(1)
    return [script_path]


if __name__ == "__main__":
    main()
