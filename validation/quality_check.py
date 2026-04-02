#!/usr/bin/env python3
"""
Performance quality checks for configuration, initialization order, and docstrings.
"""

import ast
import re
import sys
from pathlib import Path

# Add REPO_ROOT to sys.path to allow importing eqsp from source
REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_environment():
    """Set up the environment for quality checks."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Force headless Matplotlib for diagnostic scripts
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
    except ImportError:  # pragma: no cover
        pass


setup_environment()


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
            msg = f"{n_rel}: matplotlib.use('Agg') must come before pyplot import."
            errors.append(msg)

    return errors


def check_conf_types():
    """Ensure Sphinx conf.py variables have correct types."""
    conf_py = REPO_ROOT / "doc" / "conf.py"
    if not conf_py.exists():
        return []  # pragma: no cover

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


def check_doc_functions():
    """Ensure all eqsp.function calls in docs exist in the package."""
    import eqsp  # pylint: disable=import-outside-toplevel

    exported = set(dir(eqsp))
    errors = []

    # Files to check
    docs_to_check = list((REPO_ROOT / "doc").rglob("*.md"))
    docs_to_check.extend((REPO_ROOT / "doc").rglob("*.rst"))
    docs_to_check.append(REPO_ROOT / "README.md")

    # Regex for eqsp.function_name
    # Handle optional RST backslash escape for underscores: eqsp.region\_props
    func_pattern = r"(?<![\/\.a-zA-Z0-9_-])eqsp\.(?P<func>[a-zA-Z_][a-zA-Z0-9_\\-]*)"
    func_re = re.compile(func_pattern)

    ignore_funcs = {
        "visualizations",
        "illustrations",
        "histograms",
        "partitions",
        "point_set_props",
        "region_props",
        "utilities",
        "sourceforge",
        "region\\_props",
        "point\\_set\\_props",
    }

    for f in docs_to_check:
        if not f.exists():
            continue  # pragma: no cover
        content = f.read_text(encoding="utf-8")
        matches = set(func_re.findall(content))
        for m in matches:
            if m in ignore_funcs:
                continue  # pragma: no cover
            if m not in exported:
                f_rel = f.relative_to(REPO_ROOT)
                errors.append(f"{f_rel}: Referenced non-existent function `eqsp.{m}`")

    return errors


def check_doc_shapes():
    r"""
    Ensure array shape comments follow the (3, N) column-major convention.

    >>> # Mocking a manual check since we don't want to load a file
    >>> wrong_shape_re = re.compile(r"\(N,\s*[234]\)|\(\d+,\s*[234]\)")
    >>> bool(wrong_shape_re.search("(N, 3)"))
    True
    >>> bool(wrong_shape_re.search("(3, N)"))
    False
    """
    errors = []
    docs_to_check = list((REPO_ROOT / "doc").rglob("*.md"))

    # Regex for (1000, 3) or (N, 3) which is usually an error in PyEQSP
    wrong_shape_re = re.compile(r"\(N,\s*[234]\)|\(\d+,\s*[234]\)")

    for f in docs_to_check:
        if not f.exists() or "migration" in f.name:
            continue  # pragma: no cover
        content = f.read_text(encoding="utf-8")

        # Exclude common migration guide "intentional" mentions of other shapes
        if "scikit-learn" in content and "pandas" in content:
            # Simple heuristic: if we are talking about other libs, skip the shape check
            # for the features description line
            content = content.replace("(N, features)", "(EXCLUDED)")  # pragma: no cover

        if wrong_shape_re.search(content):
            f_rel = f.relative_to(REPO_ROOT)
            errors.append(
                f"{f_rel}: Documentation likely uses incorrect (N, dim) shape. "
                "PyEQSP uses (dim+1, N) columns."
            )

    return errors


def check_headings():
    r"""
    Scan for malformed headers (e.g. # Header # SubHeader).

    >>> double_heading_re = re.compile(r"^#+ .+# .+$", re.MULTILINE)
    >>> bool(double_heading_re.search("# Valid Header"))
    False
    >>> bool(double_heading_re.search("# Header # Another"))
    True
    """
    errors = []
    # Files to check
    md_files = list((REPO_ROOT / "doc").rglob("*.md"))
    md_files.append(REPO_ROOT / "README.md")

    # Regex for lines containing multiple # headers
    # Heuristic: Match lines starting with # but containing another # later
    double_heading_re = re.compile(r"^#+ .+# .+$", re.MULTILINE)

    for f in md_files:
        if not f.exists():
            continue  # pragma: no cover
        content = f.read_text(encoding="utf-8")
        if double_heading_re.search(content):
            f_rel = f.relative_to(REPO_ROOT)
            errors.append(f"{f_rel}: Found malformed double-heading.")

    return errors


def check_typos():
    """Scan for common or previously identified typos."""
    errors = []
    # Known typos to scan for
    typo_map = {
        "excitiation": "excitation",
        "Scanning foundational results": "proper metadata entry",
        "2rd": "2nd",
        "3th": "3rd",
        "4th-sphere": "4-sphere",
        "3rd-sphere": "3-sphere",
        "2nd-sphere": "2-sphere",
        "show_s2_partition(N=": "show_s2_partition(100, (N is positional-only)",
    }

    files_to_check = list(REPO_ROOT.rglob("*.md"))
    files_to_check.extend(REPO_ROOT.rglob("*.rst"))
    files_to_check.extend((REPO_ROOT / "eqsp").rglob("*.py"))

    for f in files_to_check:
        ignore_names = [".build", "results.0", "pr_checklist.md"]
        ignore = any(name in str(f) for name in ignore_names)
        if not f.exists() or ignore:
            continue  # pragma: no cover
        content = f.read_text(encoding="utf-8")
        for typo, correction in typo_map.items():
            if typo in content:
                f_rel = f.relative_to(REPO_ROOT)
                msg = f"{f_rel}: Found typo `{typo}`. Should be `{correction}`."
                errors.append(msg)

    return errors


def check_orthography():
    """Ensure Australian -ize English (Oxford spelling) project-wide."""
    errors = []
    # Known non-compliant spellings (Standardisation -> Standardization)
    ortho_map = {
        r"(?<!\w)standardisation(?!\w)": "standardization",
        r"(?<!\w)organise(?!\w)": "organize",
        r"(?<!\w)analyse(?!\w)": "analyze",
        r"(?<!\w)optimise(?!\w)": "optimize",
    }

    files_to_check = list(REPO_ROOT.rglob("*.md"))
    files_to_check.extend(REPO_ROOT.rglob("*.rst"))
    files_to_check.extend((REPO_ROOT / "eqsp").rglob("*.py"))

    for f in files_to_check:
        if not f.exists() or ".build" in str(f) or "results.0" in str(f):
            continue
        content = f.read_text(encoding="utf-8")

        # Skip reference volumes for title quotes (must be verbatim)
        if "references_vol" in f.name and f.suffix == ".md":
            continue

        for pattern, correction in ortho_map.items():
            if re.search(pattern, content, re.IGNORECASE):
                f_rel = f.relative_to(REPO_ROOT)
                msg = f"{f_rel}: Found non-Oxford spelling. Use `{correction}`."
                errors.append(msg)

    return errors


def get_all_refs(repo_root):
    """
    Parse reference keys and metadata from vol1, vol2, and AUTHORS.md.
    Returns {filename: {key: metadata}}.
    """
    refs = {}
    ref_re = re.compile(r"-\s+\*\*\[([^\]]+)\]\*\*\s+(.*)")

    # Check doc/references_vol*.md
    for fname in ["references_vol1.md", "references_vol2.md"]:
        fpath = repo_root / "doc" / fname
        if fpath.exists():
            for line in fpath.read_text(encoding="utf-8").splitlines():
                m = ref_re.match(line.strip())
                if m:
                    refs.setdefault(fname, {})[m.group(1)] = m.group(2)

    # Check AUTHORS.md
    auth_path = repo_root / "AUTHORS.md"
    if auth_path.exists():
        for line in auth_path.read_text(encoding="utf-8").splitlines():
            m = ref_re.match(line.strip())
            if m:
                refs.setdefault("AUTHORS.md", {})[m.group(1)] = m.group(2)
    return refs


def check_reference_consistency(repo_root=REPO_ROOT):
    r"""
    Enforce consistency across vol1, vol2, and AUTHORS.md.
    Flag any shared key whose metadata disagrees between files.

    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as d:
    ...     root = Path(d)
    ...     doc = root / "doc"
    ...     doc.mkdir()
    ...     _ = (doc / "references_vol1.md").write_text(
    ...         "- **[A1]** Author 1\n", encoding="utf-8"
    ...     )
    ...     _ = (root / "AUTHORS.md").write_text(
    ...         "- **[A1]** Author 2\n", encoding="utf-8"
    ...     )
    ...     errors = check_reference_consistency(repo_root=root)
    ...     "Reference mismatch for [A1]" in errors[0]
    True
    """
    errors = []
    all_refs = get_all_refs(repo_root)
    filenames = list(all_refs.keys())

    for i, f1 in enumerate(filenames):
        for f2 in filenames[i + 1 :]:
            refs1, refs2 = all_refs[f1], all_refs[f2]
            for key, meta1 in refs1.items():
                if key in refs2:
                    meta2 = refs2[key]
                    if meta1 != meta2:
                        errors.append(
                            f"Reference mismatch for [{key}] between {f1} and {f2}."
                        )

    return errors


def check_ruff_config_format():
    """Ensure ruff.toml uses the legacy-compatible flat format (no [lint])."""
    ruff_toml = REPO_ROOT / "ruff.toml"
    if not ruff_toml.exists():
        return []  # pragma: no cover

    content = ruff_toml.read_text(encoding="utf-8")
    if "[lint]" in content or "[lint." in content:
        return [
            "ruff.toml: uses modern [lint] section. "
            "Revert to flat format for legacy compatibility."
        ]
    return []


def check_standalone_pragmas():
    """Ensure '# pragma: no cover' is attached to a statement, not on its own line."""
    errors = []
    # Check package and scripts but skip third-party/venv
    python_files = list((REPO_ROOT / "eqsp").rglob("*.py"))
    python_files.extend((REPO_ROOT / "release").glob("*.py"))
    python_files.extend((REPO_ROOT / "validation").glob("*.py"))
    python_files.extend((REPO_ROOT / "tests").rglob("*.py"))

    # Standalone pragma: line consists only of whitespace and the pragma
    standalone_re = re.compile(r"^\s*#\s*pragma:\s*no\s*cover\s*$")

    for f in python_files:
        if not f.exists():
            continue  # pragma: no cover
        content = f.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines()):
            if standalone_re.match(line):
                f_rel = f.relative_to(REPO_ROOT)
                errors.append(
                    f"{f_rel}:{i + 1}: Standalone '# pragma: no cover' is ineffective. "
                    "Attach to the statement."
                )
    return errors


def check_script_paths():
    """Ensure scripts referenced in CI and core docs exist in the repository."""
    errors = []
    # Files to scan for script references
    files_to_scan = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / "CONTRIBUTING.md",
    ]
    # Also scan agent workflows
    agent_workflows = REPO_ROOT / ".agents" / "workflows"
    if agent_workflows.exists():
        files_to_scan.extend(agent_workflows.glob("*.md"))

    # Regex for "python3 path/to/script.py" or "`python3 path/to/script.py`"
    # Matches both literal and backticked paths, focusing on local scripts
    script_re = re.compile(r"python3\s+(?P<path>[\w\-\/]+\.py)")

    for f in files_to_scan:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")
        matches = script_re.findall(content)
        for rel_path in matches:
            # Skip standard library modules or installed packages
            if "/" not in rel_path and rel_path not in [
                "build_dist.py",
                "verify_all.py",
            ]:
                # If no slash, it's either in the root or a module.
                # If it's verify_all.py without validation/, it's likely a regression.
                pass

            script_path = REPO_ROOT / rel_path
            if not script_path.exists():
                f_rel = f.relative_to(REPO_ROOT)
                errors.append(f"{f_rel}: Referenced non-existent script `{rel_path}`")
    return errors


def _check_usage_tokens(usage_part, f_rel):
    """Helper to check tokens extracted from a Usage block."""
    errors = []
    for token in usage_part.split():
        clean_token = token.strip("`'\".,;")
        if clean_token.endswith(".py"):
            target_path = REPO_ROOT / clean_token
            if not target_path.exists():
                errors.append(
                    f"{f_rel}: 'Usage:' block references "
                    f"non-existent file `{clean_token}`"
                )
    return errors


def check_usage_blocks():
    """Ensure file paths mentioned in 'Usage:' docstring blocks exist."""
    errors = []
    python_files = list(REPO_ROOT.rglob("*.py"))
    exclude_dirs = {".venvs", "build", "dist", "pyeqsp.egg-info"}

    for f in python_files:
        if any(ex in f.parts for ex in exclude_dirs) or not f.exists():
            continue

        content = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue  # pragma: no cover

        f_rel = f.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                continue

            docstring = ast.get_docstring(node)
            if docstring and "Usage:" in docstring:
                usage_part = docstring.split("Usage:")[1]
                errors.extend(_check_usage_tokens(usage_part, f_rel))

    return errors


def check_index_rst_toctree():
    """Ensure toctree entries in index.rst have correct paths and indentation."""
    errors = []
    index_rst = REPO_ROOT / "doc" / "index.rst"
    if not index_rst.exists():
        return errors  # pragma: no cover

    content = index_rst.read_text(encoding="utf-8")
    lines = content.splitlines()
    in_toctree = False

    for i, line in enumerate(lines):
        if line.startswith(".. toctree::"):
            in_toctree = True
            continue

        if in_toctree:
            if not line.strip():
                continue

            if not line.startswith(" "):
                in_toctree = False
                continue

            num_spaces = len(line) - len(line.lstrip(" "))
            if num_spaces != 3:
                errors.append(
                    f"doc/index.rst:{i + 1}: toctree entry '{line.strip()}' has "
                    f"incorrect indentation ({num_spaces}). Must be exactly 3."
                )

            stripped = line.strip()
            if not stripped.startswith(":"):
                path_str = stripped
                if "<" in stripped and stripped.endswith(">"):
                    path_str = stripped.split("<")[-1][:-1]

                target = REPO_ROOT / "doc" / path_str
                if target.suffix == "":
                    if not (
                        target.with_suffix(".rst").exists()
                        or target.with_suffix(".md").exists()
                        or target.exists()
                    ):
                        msg = f"doc/index.rst:{i + 1}: path '{path_str}' not found."
                        errors.append(msg)
                else:
                    if not target.exists():
                        msg = f"doc/index.rst:{i + 1}: path '{path_str}' not found."
                        errors.append(msg)

    return errors


def main():
    """Run all quality checks."""
    all_errors = []
    all_errors.extend(check_matplotlib_init())
    all_errors.extend(check_conf_types())
    all_errors.extend(check_docstring_links())
    all_errors.extend(check_doc_functions())
    all_errors.extend(check_doc_shapes())
    all_errors.extend(check_headings())
    all_errors.extend(check_typos())
    all_errors.extend(check_orthography())
    all_errors.extend(check_reference_consistency())
    all_errors.extend(check_ruff_config_format())
    all_errors.extend(check_standalone_pragmas())
    all_errors.extend(check_script_paths())
    all_errors.extend(check_index_rst_toctree())
    all_errors.extend(check_usage_blocks())

    if all_errors:
        print(f"Found {len(all_errors)} quality issues:")
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)

    print("Quality checks passed!")


if __name__ == "__main__":
    main()
