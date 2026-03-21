"""
Unit tests for documentation diagnostic scripts.
"""

# pylint: disable=redefined-outer-name, unused-argument
import sys

import pytest

from doc.maint import check_links, quality_check


@pytest.fixture
def mock_repo(tmp_path, monkeypatch):
    """Create a mock repository structure for testing scripts."""
    # Create doc directory
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()

    # Create conf.py with a set instead of list (to trigger error)
    (doc_dir / "conf.py").write_text(
        "extensions = []\nmyst_enable_extensions = {'dollarmath'}\n", encoding="utf-8"
    )

    # Create some markdown files
    (doc_dir / "valid.md").write_text(
        "# Valid\n\n(target)=\n[link](#target)\n{ref}`target`", encoding="utf-8"
    )
    (doc_dir / "broken.md").write_text(
        "[broken](#nonexistent)\n{ref}`dead-ref`", encoding="utf-8"
    )
    (doc_dir / "cross.md").write_text(
        "[cross](valid.md#target)\n[html](valid.html#target)", encoding="utf-8"
    )

    # Create root files
    (tmp_path / "README.md").write_text("eqsp.func_a", encoding="utf-8")
    (tmp_path / "CONTRIBUTING.md").write_text("docs", encoding="utf-8")
    (tmp_path / "INSTALL.md").write_text("install", encoding="utf-8")

    # Create eqsp dir
    eqsp_dir = tmp_path / "eqsp"
    eqsp_dir.mkdir()
    (eqsp_dir / "code.py").write_text(
        '"""\n[Markdown](link)\n"""\ndef func(): pass', encoding="utf-8"
    )
    (eqsp_dir / "__init__.py").write_text(
        "__version__ = '1.0'\ndef func_a(): pass\n", encoding="utf-8"
    )

    # Create examples dir
    ex_dir = tmp_path / "examples" / "user-guide" / "src"
    ex_dir.mkdir(parents=True)
    (ex_dir / "bad_init.py").write_text(
        "import matplotlib.pyplot\nmatplotlib.use('Agg')", encoding="utf-8"
    )

    # Monkeypatch REPO_ROOT in scripts
    monkeypatch.setattr(check_links, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_links, "DOC_DIR", doc_dir)
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # Ensure the test environment uses the mock eqsp
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "eqsp", raising=False)  # Force reload if imported

    return tmp_path


def test_check_links_functional(mock_repo):
    """Test check_links.py on a mock filesystem."""
    md_files = check_links.get_all_md_files()
    assert len(md_files) >= 5

    file_targets = {}
    all_links = []
    for md_file in md_files:
        targets, links = check_links.parse_file(md_file)
        file_targets[md_file.resolve()] = targets
        all_links.extend(links)

    # Add a check for external links to hit line 92
    targets, links = check_links.parse_content("[ext](http://example.com)")
    all_links.extend(links)

    # Add a check for broken cross-file anchor to hit line 126+
    (mock_repo / "doc" / "other.md").write_text("(other-target)=", encoding="utf-8")
    (mock_repo / "doc" / "broken_cross.md").write_text(
        "[broken](valid.md#other-target)", encoding="utf-8"
    )
    # Refresh md_files
    md_files = check_links.get_all_md_files()
    for md_file in md_files:
        targets, links = check_links.parse_file(md_file)
        file_targets[md_file.resolve()] = targets
        all_links.extend(links)

    broken = []
    for link, source_file in all_links:
        status = check_links.is_link_broken(link, source_file, file_targets)
        if status:
            broken.append((link, status))

    assert any(b[0] == "#nonexistent" for b in broken)
    assert any(b[0] == "ref:dead-ref" for b in broken)
    assert any("exists in another file" in str(b[1]) for b in broken)


def test_quality_check_matplotlib(mock_repo):
    """Test matplotlib initialization check."""
    errors = quality_check.check_matplotlib_init()
    assert any("bad_init.py" in err for err in errors)


def test_quality_check_conf_types(mock_repo):
    """Test check_conf_types."""
    errors = quality_check.check_conf_types()
    assert any("conf.py" in err for err in errors)


def test_quality_check_doc_functions(mock_repo):
    """Test check_doc_functions."""
    (mock_repo / "doc" / "missing_func.md").write_text(
        "eqsp.non_existent_func", encoding="utf-8"
    )
    errors = quality_check.check_doc_functions()
    assert any("non_existent_func" in err for err in errors)
    # Re-verify eq_regions exists (so no error for it)
    assert not any("eq_regions" in err for err in errors)


def test_quality_check_doc_shapes(mock_repo):
    """Test check_doc_shapes."""
    (mock_repo / "doc" / "bad_shape.md").write_text(
        "Array shape (N, 3)", encoding="utf-8"
    )
    errors = quality_check.check_doc_shapes()
    assert any("bad_shape.md" in err for err in errors)


def test_quality_check_headings(mock_repo):
    """Test check_headings."""
    (mock_repo / "doc" / "bad_head.md").write_text("# Head # Double", encoding="utf-8")
    errors = quality_check.check_headings()
    assert any("bad_head.md" in err for err in errors)


def test_quality_check_typos(mock_repo):
    """Test check_typos."""
    (mock_repo / "doc" / "typo.md").write_text(
        "Found excitiation here.", encoding="utf-8"
    )
    errors = quality_check.check_typos()
    assert any("typo.md" in err for err in errors)


def test_quality_check_docstring_links(mock_repo):
    """Test check_docstring_links."""
    errors = quality_check.check_docstring_links()
    assert any("code.py" in err for err in errors)
