#!/usr/bin/env python3
"""
Performance quality checks for configuration, initialization order, and docstrings.
"""
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def check_matplotlib_init():
    """Ensure matplotlib.use('Agg') comes before pyplot import in examples."""
    examples_dir = REPO_ROOT / "examples" / "user-guide" / "src"
    errors = []

    for f in examples_dir.glob("*.py"):
        content = f.read_text(encoding="utf-8")
        lines = content.splitlines()

        pyplot_idx = -1
        agg_idx = -1

        for i, line in enumerate(lines):
            is_py = (
                "import matplotlib.pyplot" in line
                or "from matplotlib import pyplot" in line
            )
            if is_py:
                if pyplot_idx == -1:
                    pyplot_idx = i
            if "matplotlib.use('Agg')" in line or 'matplotlib.use("Agg")' in line:
                if agg_idx == -1:
                    agg_idx = i

        if pyplot_idx != -1 and agg_idx != -1 and agg_idx > pyplot_idx:
            n_rel = f.relative_to(REPO_ROOT)
            msg = (
                f"{n_rel}: matplotlib.use('Agg') "
                "must come before pyplot import."
            )
            errors.append(msg)

    return errors

def check_conf_types():
    """Ensure Sphinx conf.py variables have correct types."""
    conf_py = REPO_ROOT / "doc" / "conf.py"
    if not conf_py.exists():
        return []

    errors = []
    content = conf_py.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                is_myst = (
                    isinstance(target, ast.Name)
                    and target.id == "myst_enable_extensions"
                )
                if is_myst:
                    if isinstance(node.value, ast.Set):
                        msg = (
                            "doc/conf.py: myst_enable_extensions "
                            "should be a list or tuple, not a set."
                        )
                        errors.append(msg)

    return errors

def check_docstring_links():
    """Ensure docstrings use reStructuredText syntax, not Markdown links."""
    errors = []
    # Only check eqsp package
    package_dir = REPO_ROOT / "eqsp"

    # Regex for Markdown links: [text](link)
    markdown_link_re = re.compile(r"\[[^\]]+\]\((?P<link>[^\)]+)\)")

    for f in package_dir.rglob("*.py"):
        content = f.read_text(encoding="utf-8")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    if markdown_link_re.search(docstring):
                        f_rel = f.relative_to(REPO_ROOT)
                        errors.append(
                            f"{f_rel}: Docstring contains Markdown links. "
                            "Use reStructuredText syntax."
                        )

    return errors

def main():
    """Run all quality checks."""
    all_errors = []
    all_errors.extend(check_matplotlib_init())
    all_errors.extend(check_conf_types())
    all_errors.extend(check_docstring_links())

    if all_errors:
        print(f"Found {len(all_errors)} quality issues:")
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)

    print("Quality checks passed!")

if __name__ == "__main__":
    main()
