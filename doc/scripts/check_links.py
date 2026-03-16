#!/usr/bin/env python3
"""
Check for broken internal and cross-file links in Markdown documentation.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_DIR = REPO_ROOT / "doc"

# Regex for MyST targets: (target)=
TARGET_RE = re.compile(r"^\((?P<target>[\w.-]+)\)=", re.MULTILINE)

# Regex for Markdown links: [text](link)
LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<link>[^\)]+)\)")

# Regex for MyST ref: {ref}`target` or {ref}`text <target>`
REF_RE = re.compile(r"\{ref\}`(?:[^<`]+<)?(?P<target>[^>`]+)>?`")

def get_all_md_files():
    """Return a list of all .md files in the repository."""
    md_files = list(DOC_DIR.rglob("*.md"))
    md_files.append(REPO_ROOT / "README.md")
    md_files.append(REPO_ROOT / "CONTRIBUTING.md")
    md_files.append(REPO_ROOT / "INSTALL.md")
    return [f for f in md_files if f.exists()]

def parse_file(file_path):
    """Extract targets and links from a file."""
    content = file_path.read_text(encoding="utf-8")
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
        links.append((match.group("link"), file_path))

    for match in REF_RE.finditer(content):
        links.append((f"ref:{match.group('target')}", file_path))

    return targets, links

def is_link_broken(link, source_file, file_targets):
    # pylint: disable=too-many-return-statements
    """Return True if the link is broken."""
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
                    return True
                return False
        return True

    if anchor:
        if anchor not in file_targets.get(resolved_path, set()):
            return True
    return False


def main():
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
        if is_link_broken(link, source_file, file_targets):
            broken_links.append((link, source_file))

    if broken_links:
        print(f"Found {len(broken_links)} broken links:")
        for link, source in broken_links:
            print(f"  {source.relative_to(REPO_ROOT)}: {link}")
        sys.exit(1)

    print("No broken links found!")

if __name__ == "__main__":
    main()
