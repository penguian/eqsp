#!/usr/bin/env python3
"""
Check for broken internal and cross-file links in Markdown documentation.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_DIR = REPO_ROOT / "doc"

# Ensure REPO_ROOT is in sys.path for internal lookups if needed
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Regex for MyST targets: (target)=
TARGET_RE = re.compile(r"^\((?P<target>[\w.-]+)\)=", re.MULTILINE)

# Regex for Markdown links: [text](link)
LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<link>[^\)]+)\)")

# Regex for MyST ref: {ref}`target` or {ref}`text <target>`
# Must be exactly {ref}`...` and not preceded by a backtick
REF_RE = re.compile(r"(?<!`)\{ref\}`(?:[^<`\n]+<)?(?P<target>[^>`\n]+)>?`")


def get_all_md_files():
    """Return a list of all .md files in the repository."""
    md_files = list(DOC_DIR.rglob("*.md"))
    md_files.append(REPO_ROOT / "README.md")
    md_files.append(REPO_ROOT / "CONTRIBUTING.md")
    md_files.append(REPO_ROOT / "INSTALL.md")
    return [f for f in md_files if f.exists()]


def parse_content(content):
    r"""
    Extract targets and links from Markdown content.

    >>> targets, links = parse_content("(target)=\n# Header\n[text](#slug)")
    >>> "target" in targets
    True
    >>> "header" in targets
    True
    >>> "#slug" in [link_data[0] for link_data in links]
    True
    """
    targets = set(TARGET_RE.findall(content))
    # Also add headers as targets
    headers = re.findall(r"^#+ (?P<header>.+)$", content, re.MULTILINE)
    for h in headers:
        slug = (
            h.lower()
            .replace(" ", "-")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        targets.add(slug)

    links = []
    for match in LINK_RE.finditer(content):
        links.append((match.group("link"), None))

    for match in REF_RE.finditer(content):
        links.append((f"ref:{match.group('target')}", None))

    return targets, links


def parse_file(file_path):
    """Extract targets and links from a file."""
    content = file_path.read_text(encoding="utf-8")
    targets, links = parse_content(content)
    # Re-wrap links with the file path
    links = [(link_data[0], file_path) for link_data in links]
    return targets, links


def is_link_broken(link, source_file, file_targets):
    # pylint: disable=too-many-return-statements
    """
    Return True if the link is broken.

    >>> file_targets = {Path("f1.md").resolve(): {"t1"}}
    >>> is_link_broken("#t1", Path("f1.md"), file_targets)
    False
    >>> is_link_broken("#t2", Path("f1.md"), file_targets)
    True
    """
    if link.startswith("ref:"):
        target = link[4:]
        all_targets = set()
        for targets in file_targets.values():
            all_targets.update(targets)
        return target not in all_targets

    # Ignore external links
    if link.startswith(("http://", "https://", "mailto:", "ftp:")):
        return False

    # Handle anchors
    parts = link.split("#")
    target_path = parts[0]
    anchor = parts[1] if len(parts) > 1 else None

    resolved_path = source_file.parent / target_path if target_path else source_file
    resolved_path = resolved_path.resolve()

    if not target_path:
        # Internal anchor
        if anchor and anchor not in file_targets.get(source_file.resolve(), set()):
            return True
        return False

    # Cross-file link
    if not resolved_path.exists():
        # Check if it refers to a .md that Sphinx turns into .html
        if target_path.endswith(".html"):
            md_equiv = resolved_path.with_suffix(".md")
            if md_equiv.exists():
                # We should ideally use .md in source,
                # but let's check anchors if provided
                t_set = file_targets.get(md_equiv.resolve(), set())
                if anchor and anchor not in t_set:
                    return True  # pragma: no cover
                return False
        return True  # pragma: no cover

    if anchor:
        local_targets = file_targets.get(resolved_path, set())
        if anchor not in local_targets:
            # Check if this anchor exists in ANY file
            all_targets = set()
            for targets in file_targets.values():
                all_targets.update(targets)
            if anchor in all_targets:
                # Anchor exists but not here! This is likely a broken local link
                # that should be a {ref}.
                return (
                    f"Anchor '#{anchor}' exists in another file. Use {{ref}} instead."
                )
            return True  # pragma: no cover
    return False


def main():  # pragma: no cover
    """Check for broken links in Markdown documentation."""
    md_files = get_all_md_files()
    file_targets = {}
    all_links = []

    for f in md_files:
        targets, links = parse_file(f)
        file_targets[f.resolve()] = targets
        all_links.extend(links)

    broken_links = []
    for link, source_file in all_links:
        status = is_link_broken(link, source_file, file_targets)
        if status:
            broken_links.append((link, source_file, status))

    if broken_links:
        print(f"Found {len(broken_links)} broken links:")
        for link, source, status in broken_links:
            msg = status if isinstance(status, str) else link
            print(f"  {source.relative_to(REPO_ROOT)}: {msg}")
        sys.exit(1)

    print("No broken links found!")


if __name__ == "__main__":  # pragma: no cover
    main()
