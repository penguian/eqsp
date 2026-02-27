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
        content = script_path.read_text()
        return "mayavi" in content or "mlab" in content
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--two-d-only", action="store_true",
        help="Skip 3D Mayavi plots"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing PNG files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without executing"
    )
    parser.add_argument(
        "--figure", type=str,
        help="Regenerate a specific figure (e.g., fig_3_1_partition_s2_33.py or 3_1)"
    )
    parser.add_argument(
        "--show-progress", action="store_true", default=True,
        help="Show progress from figure scripts (default: %(default)s)"
    )
    parser.add_argument(
        "--no-progress", action="store_false", dest="show_progress",
        help="Hide progress from figure scripts"
    )
    args = parser.parse_args()

    # Directory setup
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "src"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = base_dir.parent.parent
    venv_python = project_root / "venv_sys" / "bin" / "python3"
    run_patched = src_dir / "run_patched.py"

    scripts = sorted(src_dir.glob("fig_*.py"))

    if args.figure:
        target = args.figure
        if not target.endswith(".py"):
            # Try to match by number or prefix
            matches = [s for s in scripts if target in s.name]
            if not matches:
                print(f"Error: No script found matching '{target}'")
                sys.exit(1)
            scripts = matches
        else:
            script_path = src_dir / target
            if not script_path.exists():
                print(f"Error: Script {target} not found in {src_dir}")
                sys.exit(1)
            scripts = [script_path]

    print("--- EQSP Figure Regeneration ---")
    print(f"Source: {src_dir}")
    print(f"Target: {results_dir}")
    if args.two_d_only:
        print("Mode: 2D-only (Skipping Mayavi plots)")
    else:
        print("Mode: Full (Including 3D Mayavi plots)")
    print("-" * 32)

    total_start = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0

    for script in scripts:
        png_name = script.with_suffix(".png").name
        target_png = results_dir / png_name

        needs_3d = is_3d_script(script)

        # Skip logic
        if needs_3d and args.two_d_only:
            print(f"{script.name:40s} : SKIPPED (3D)")
            skip_count += 1
            continue

        if target_png.exists() and not args.force:
            print(f"{script.name:40s} : SKIPPED (exists)")
            skip_count += 1
            continue

        # Environment Selection
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(project_root)

        python_exe = sys.executable
        if needs_3d:
            if not venv_python.exists():
                print(
                    f"{script.name:40s} : FAILED "
                    f"(venv_sys not found at {venv_python})"
                )
                fail_count += 1
                continue
            python_exe = str(venv_python)
            run_env.update(QT_ENV)
            mode_label = "[3D]"
        else:
            mode_label = "[2D]"

        print(f"Running {mode_label} {script.name}...", end="", flush=True)
        cmd = [python_exe, str(run_patched), str(script)]
        if args.show_progress:
            cmd.append("--show-progress")

        if args.dry_run:
            print(" DRY RUN")
            success_count += 1
            continue

        script_start = time.time()
        try:
            # Popen to stream output in real-time
            process = subprocess.Popen(
                cmd,
                cwd=str(results_dir),
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

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
            duration = time.time() - script_start
            duration_str = format_duration(duration)

            if rc == 0:
                print(f" DONE ({duration_str})")
                success_count += 1
            else:
                print(f" FAILED (code {rc}) after {duration_str}")
                if not args.show_progress:
                    print(f"Error output:\n{''.join(output_lines)}")
                fail_count += 1
        except Exception as e:
            duration = time.time() - script_start
            print(f" ERROR after {format_duration(duration)}: {str(e)}")
            fail_count += 1

    total_duration = time.time() - total_start
    print("-" * 32)
    print(
        f"Summary: {success_count} succeeded, "
        f"{fail_count} failed, {skip_count} skipped."
    )
    print(f"Total time taken: {format_duration(total_duration)}")

if __name__ == "__main__":
    main()
