#!/usr/bin/env python3
"""
release/pypi_readme_fix.py

Converts relative links in README.md to absolute GitHub URLs, producing a
README_dist.md for PyPI. This fixes the broken-links issue on PyPI where
relative paths do not resolve outside the GitHub repository.

Usage:
    python release/pypi_readme_fix.py
"""
# pylint: disable=line-too-long,missing-function-docstring,subprocess-run-check

import os
import re
import sys


def main():
    readme_in = "README.md"
    readme_out = "README_dist.md"
    github_base = "https://github.com/penguian/pyeqsp"

    # We use 'main' for documentation links on PyPI so that they resolve
    # immediately and point to the most recent stable documentation state.
    target_ref = "main"

    if not os.path.exists(readme_in):
        print(f"ERROR: {readme_in} not found.", file=sys.stderr)
        sys.exit(1)

    with open(readme_in, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find standard Markdown links: [link text](url)
    # We ignore urls starting with http, https, mailto, or #
    pattern = r"\[([^\]]+)\]\((?!http|#|mailto)([^)]+)\)"

    def replace_link(match):
        text = match.group(1)
        url = match.group(2)

        # GitHub uses 'tree' for directories and 'blob' for files
        mode = "tree" if url.endswith("/") else "blob"
        absolute_url = f"{github_base}/{mode}/{target_ref}/{url}"
        return f"[{text}]({absolute_url})"

    new_content = re.sub(pattern, replace_link, content)

    with open(readme_out, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(
        f"Successfully generated {readme_out} "
        f"with absolute links pointing to {target_ref}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
