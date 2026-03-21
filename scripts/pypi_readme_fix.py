#!/usr/bin/env python3
"""
scripts/pypi_readme_fix.py

Converts relative links in README.md to absolute GitHub URLs, producing a
README_dist.md for PyPI. This fixes the broken-links issue on PyPI where
relative paths do not resolve outside the GitHub repository.

Usage:
    python scripts/pypi_readme_fix.py
"""
# pylint: disable=line-too-long,missing-function-docstring,subprocess-run-check

import os
import re
import sys
import tomllib


def main():
    pyproject_path = "pyproject.toml"
    readme_in = "README.md"
    readme_out = "README_dist.md"
    github_base = "https://github.com/penguian/pyeqsp"

    if not os.path.exists(pyproject_path):
        print(f"ERROR: {pyproject_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        try:
            config = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            print(f"ERROR: Malformed {pyproject_path}: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        version = config["project"]["version"]
    except KeyError:
        print(
            f"ERROR: 'project.version' key not found in {pyproject_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Convert PyPI version (0.99.5) to tag format (release_0_99_5)
    # Note: Beta versions (0.99.5b1) become release_0_99_5b1
    tag_version = version.replace(".", "_")
    release_tag = f"release_{tag_version}"

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
        absolute_url = f"{github_base}/{mode}/{release_tag}/{url}"
        return f"[{text}]({absolute_url})"

    new_content = re.sub(pattern, replace_link, content)

    with open(readme_out, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(
        f"Successfully generated {readme_out} "
        f"with absolute links for tag {release_tag}."
    )


if __name__ == "__main__":
    main()
