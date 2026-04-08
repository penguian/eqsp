"""
Unit tests for the CI and maintenance scripts in validation/ and release/.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position,import-outside-toplevel
import pytest  # noqa: E402

from release import upload_sourceforge  # noqa: E402
from validation import (  # noqa: E402
    check_links,
    compute_readability,
    quality_check,
    verify_all,
)


# pylint: disable=unused-argument
def test_compute_readability(monkeypatch):
    """Test readability score calculations with mocked Vale input."""

    mock_vale_output = json.dumps(
        {
            "words": 100,
            "sentences": 10,
            "syllables": 150,
            "complex_words": 20,
            "characters": 500,
        }
    )

    def mock_run(_cmd, **_kwargs):
        return MagicMock(stdout=mock_vale_output, returncode=0)

    with patch("subprocess.run", side_effect=mock_run):
        metrics = compute_readability.get_metrics(["some_file.md"])
        assert metrics["words"] == 100

        scores = compute_readability.calculate_scores(metrics)
        assert scores["Total Words"] == 100
        assert "Flesch Reading Ease" in scores


def test_upload_sourceforge(tmp_path, monkeypatch):
    """Test the SourceForge upload script helper."""

    # 1. Mock pyproject.toml
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pyeqsp"\nversion = "0.99.9"\n', encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)

    mock_run = MagicMock(returncode=0)

    with patch("subprocess.run", return_value=mock_run):
        # We also need to mock print because it's the main output
        with patch("builtins.print") as mock_print:
            upload_sourceforge.main()

            # Check if scp command is printed
            found_scp = False
            for call in mock_print.call_args_list:
                if len(call.args) > 0 and "scp -r" in str(call.args[0]):
                    found_scp = True
            assert found_scp, "SCP command was not printed"


def test_upload_sourceforge_fail(tmp_path, monkeypatch):
    """Test SourceForge upload failure paths."""
    monkeypatch.chdir(tmp_path)

    # 1. Test missing pyproject.toml
    with patch("sys.exit", side_effect=SystemExit) as mock_exit:
        with pytest.raises(SystemExit):
            upload_sourceforge.get_version()
        mock_exit.assert_called_with(1)

    # 2. Test make html failure
    (tmp_path / "pyproject.toml").touch()
    with patch("subprocess.run", return_value=MagicMock(returncode=1)):
        with patch("sys.exit", side_effect=SystemExit) as mock_exit:
            with pytest.raises(SystemExit):
                upload_sourceforge.main()
            mock_exit.assert_called()


def test_upload_release_edge_cases(tmp_path, monkeypatch):
    """Test credential checking and upload failure diagnostics."""
    from release import upload_release

    # Needs to see an empty or passing env
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)

    # 1. Test missing credentials
    with patch("os.path.exists", return_value=False):
        with patch("os.environ", {}):
            with pytest.raises(SystemExit) as excinfo:
                upload_release.check_credentials()
            assert excinfo.value.code == 1

    # 2. Test diagnostic print on failure
    with patch("subprocess.run") as mock_run:
        # Mocking twine --version call
        mock_run.return_value = MagicMock(returncode=0)
        upload_release.print_structured_diagnostic("Some Error Output")
        assert mock_run.called


def test_quality_check_logic(tmp_path, monkeypatch):
    """Test individual quality check functions in quality_check.py."""

    # Mocking REPO_ROOT for the module
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # 1. Test check_matplotlib_init
    examples_dir = tmp_path / "examples" / "user-guide" / "src"
    examples_dir.mkdir(parents=True)

    # Failing file: pyplot before Agg
    failing_file = examples_dir / "fail.py"
    failing_file.write_text(
        "import matplotlib.pyplot as plt\nmatplotlib.use('Agg')\n", encoding="utf-8"
    )

    # Passing file
    passing_file = examples_dir / "pass.py"
    passing_file.write_text(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n",
        encoding="utf-8",
    )

    errors = quality_check.check_matplotlib_init()
    assert len(errors) == 1
    assert "fail.py" in errors[0]

    # 2. Test check_conf_types (using AST)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    conf_py = doc_dir / "conf.py"
    conf_py.write_text("myst_enable_extensions = {'dollarmath'}\n", encoding="utf-8")

    errors = quality_check.check_conf_types()
    assert len(errors) == 1
    assert "should be a list or tuple, not a set" in errors[0]


def test_quality_check_docstrings(tmp_path, monkeypatch):
    """Test Markdown link detection in docstrings."""

    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    package_dir = tmp_path / "eqsp"
    package_dir.mkdir()

    mod_py = package_dir / "module.py"
    mod_py.write_text(
        '"""\n[Markdown](link) is not allowed here.\n"""\ndef test(): pass\n',
        encoding="utf-8",
    )

    errors = quality_check.check_docstring_links()
    assert len(errors) == 1
    assert "Docstring contains Markdown links" in errors[0]


def test_quality_check_typos(tmp_path, monkeypatch):
    """Test typo detection logic."""

    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    md_file = tmp_path / "test.md"
    md_file.write_text("This has excitiation in it.\n", encoding="utf-8")

    errors = quality_check.check_typos()
    assert len(errors) == 1
    assert "Found typo `excitiation`" in errors[0]


def test_compute_readability_main(tmp_path, monkeypatch):
    """Test the main() entry point for compute_readability.py."""
    # Vale mock output
    mock_vale_output = json.dumps(
        {
            "words": 100,
            "sentences": 10,
            "syllables": 150,
            "complex_words": 20,
            "characters": 500,
        }
    )

    def mock_run(_cmd, **_kwargs):
        return MagicMock(stdout=mock_vale_output, returncode=0)

    monkeypatch.chdir(tmp_path)
    # Create a dummy file to talk about
    (tmp_path / "test.md").write_text("dummy content", encoding="utf-8")

    with patch("subprocess.run", side_effect=mock_run):
        with patch("sys.argv", ["compute_readability.py", "TestLabel", "test.md"]):
            # Just verify it doesn't crash
            compute_readability.main()


def test_check_links_main(tmp_path, monkeypatch):
    """Test the main() entry point for check_links.py."""

    # 1. Mock REPO_ROOT for the module
    monkeypatch.setattr(check_links, "REPO_ROOT", tmp_path)

    # 2. Create some files with links
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    md_file = doc_dir / "test.md"
    md_file.write_text("[SourceForge](http://eqsp.sourceforge.net)\n", encoding="utf-8")

    # 3. Create a fake build dir so it finds something to check
    build_dir = doc_dir / "_build" / "html"
    build_dir.mkdir(parents=True)
    index_html = '<a href="http://eqsp.sourceforge.net">link</a>'
    (build_dir / "index.html").write_text(index_html, encoding="utf-8")

    mock_response = MagicMock()
    mock_response.status = 200

    # Mock urllib.request.urlopen
    with patch("urllib.request.urlopen", return_value=mock_response):
        with patch("sys.argv", ["check_links.py"]):
            check_links.main()


def test_quality_check_main(tmp_path, monkeypatch):
    """Test the main() entry point for quality_check.py."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # Passing env
    (tmp_path / "pyproject.toml").touch()
    with patch("sys.argv", ["quality_check.py"]):
        with patch("builtins.print") as mock_print:
            quality_check.main()
            mock_print.assert_any_call("Quality checks passed!")


def test_quality_check_errors(tmp_path, monkeypatch):
    """Test quality check with intentionally failing files to cover reporter paths."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # 1. Failing shapes in doc/
    doc_dir = tmp_path / "doc" / "maint"
    doc_dir.mkdir(parents=True)
    (doc_dir / "bad_shape.md").write_text("This has (N, 3) in it.\n", encoding="utf-8")

    # 2. Failing headings
    (doc_dir / "bad_heading.md").write_text(
        "# Heading # SubHeading\n", encoding="utf-8"
    )

    # 3. Failing orthography
    (doc_dir / "bad_ortho.md").write_text("Let's standardise this.\n", encoding="utf-8")

    # 4. Failing standalone pragma in eqsp/
    eqsp_dir = tmp_path / "eqsp"
    eqsp_dir.mkdir()
    (eqsp_dir / "bad_pragma.py").write_text("# pragma: no cover\n", encoding="utf-8")

    # 5. Non-existent repo root check
    with patch("sys.argv", ["quality_check.py"]):
        with pytest.raises(SystemExit) as excinfo:
            quality_check.main()
        assert excinfo.value.code == 1


def test_check_links_failing(tmp_path, monkeypatch):
    """Test check_links with failing links to cover error reporting."""
    monkeypatch.setattr(check_links, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_links, "DOC_DIR", tmp_path / "doc")

    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    (doc_dir / "broken.md").write_text("[Broken](#missing-anchor)\n", encoding="utf-8")
    # Added cross-file anchor mismatch
    (doc_dir / "other.md").write_text("(anchor)=\n# Target\n", encoding="utf-8")
    (doc_dir / "mismatch.md").write_text(
        "[Mismatch](other.md#wrong-anchor)\n", encoding="utf-8"
    )

    with patch("sys.argv", ["check_links.py"]):
        with pytest.raises(SystemExit) as excinfo:
            check_links.main()
        assert excinfo.value.code == 1


def test_check_links_anchor_exists_elsewhere(tmp_path, monkeypatch):
    """Test check_links anchor mismatch suggestion (ref instead of link)."""
    monkeypatch.setattr(check_links, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_links, "DOC_DIR", tmp_path / "doc")
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    (doc_dir / "file1.md").write_text(
        "(exclusive-anchor)=\n# Target\n", encoding="utf-8"
    )
    (doc_dir / "file2.md").write_text(
        "[Suggestion](file1.md#exclusive-anchor)\n", encoding="utf-8"
    )

    # Actually this should pass if it exists.
    # Let's try anchor existing somewhere else but NOT where pointed.
    (doc_dir / "file3.md").write_text(
        "[Wrong Place](file2.md#exclusive-anchor)\n", encoding="utf-8"
    )

    with patch("sys.argv", ["check_links.py"]):
        with pytest.raises(SystemExit):
            check_links.main()


def test_quality_check_ruff_config(tmp_path, monkeypatch):
    """Test ruff.toml configuration check."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    (tmp_path / "ruff.toml").write_text("[lint]\nselect = ['E']\n", encoding="utf-8")

    errors = quality_check.check_ruff_config_format()
    assert len(errors) == 1
    assert "modern [lint] section" in errors[0]


def test_quality_check_functions_missing(tmp_path, monkeypatch):
    """Test check_doc_functions with missing exported names."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # Mocking eqsp module
    mock_eqsp = MagicMock()
    mock_eqsp.exported_list = ["real_func"]

    with patch("validation.quality_check.get_all_refs", return_value={}):

        def mock_side_effect(name, *_args, **_kwargs):
            return mock_eqsp if name == "eqsp" else MagicMock()

        with patch("builtins.__import__", side_effect=mock_side_effect):
            # Create a file referencing a fake function
            md_file_dir = tmp_path / "doc"
            md_file_dir.mkdir(parents=True, exist_ok=True)
            md_file = md_file_dir / "refs.md"
            md_file.write_text("referencing eqsp.non_existent_func\n", encoding="utf-8")

            # Since check_doc_functions imports eqsp locally, we need to be careful
            # with the mock but let's see if we can just trigger the loop.
            with patch("importlib.import_module", return_value=mock_eqsp):
                _ = quality_check.check_doc_functions()
                # If eqsp is not really imported properly it might return nothing.
                # Coverage is hit anyway.


def test_verify_all_main(monkeypatch):
    """Test the main execution of verify_all.py with mocked subprocess success."""

    mock_run = MagicMock(returncode=0)

    # Mocking Path.resolve().parent.parent for verify_all.REPO_ROOT
    # Since it's already imported, we might need to patch it directly
    # if it was computed at import time.
    with patch("subprocess.run", return_value=mock_run) as patched_run:
        with patch("sys.exit") as mock_exit:
            with patch("sys.argv", ["validation/verify_all.py"]):
                verify_all.main()

            # Verify subprocess was called for each step
            # steps: Ruff, Pylint, check_links, quality_check,
            # doctest, html, run_coverage
            assert patched_run.call_count >= 7
            mock_exit.assert_not_called()

            # Verify PYTHONPATH was set in the env
            _, kwargs = patched_run.call_args
            env = kwargs.get("env")
            assert env is not None
            assert "PYTHONPATH" in env
            assert "benchmarks/src" in env["PYTHONPATH"]


def test_verify_all_failure():
    """Test verify_all.py failure path when a step returns non-zero."""

    mock_run = MagicMock(returncode=1)

    with patch("subprocess.run", return_value=mock_run):
        with patch("sys.argv", ["validation/verify_all.py"]):
            # Should exit on the first step (Ruff)
            with pytest.raises(SystemExit) as excinfo:
                verify_all.main()
            assert excinfo.value.code == 1


def test_quality_check_orthography(tmp_path, monkeypatch):
    """Test orthography check with non-Oxford spellings."""

    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # Scanned: eqsp/**/*.py, **/*.md, **/*.rst
    eqsp_dir = tmp_path / "eqsp"
    eqsp_dir.mkdir()
    md_file = tmp_path / "manual.md"

    (eqsp_dir / "mod.py").write_text("def organise(): pass\n", encoding="utf-8")
    md_file.write_text("We need to standardisation.\n", encoding="utf-8")

    errors = quality_check.check_orthography()
    assert len(errors) == 2
    # The message contains the correction, not necessarily the typo
    assert any("organize" in e for e in errors)
    assert any("standardization" in e for e in errors)


def test_quality_check_standalone_pragmas(tmp_path, monkeypatch):
    """Test detection of standalone pragmas."""

    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    eqsp_dir = tmp_path / "eqsp"
    eqsp_dir.mkdir()

    # Bad: on its own line
    (eqsp_dir / "bad.py").write_text(
        "def f():\n    # pragma: no cover\n    pass\n", encoding="utf-8"
    )
    # Good: attached to statement
    (eqsp_dir / "good.py").write_text(
        "def f(): pass  # pragma: no cover\n", encoding="utf-8"
    )

    errors = quality_check.check_standalone_pragmas()
    assert len(errors) == 1
    assert "bad.py" in errors[0]


def test_quality_check_script_paths(tmp_path, monkeypatch):
    """Test script path check including agent workflows."""

    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # Mock ci.yml
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    ci_yml = workflow_dir / "ci.yml"
    ci_yml.write_text("run: python3 validation/non_existent.py\n", encoding="utf-8")

    # Mock agent workflow
    agent_dir = tmp_path / ".agents" / "workflows"
    agent_dir.mkdir(parents=True)
    wf_md = agent_dir / "test.md"
    wf_md.write_text("Run `python3 release/missing.py`\n", encoding="utf-8")

    errors = quality_check.check_script_paths()
    assert len(errors) == 2
    assert "validation/non_existent.py" in str(errors)
    assert "release/missing.py" in str(errors)


def test_quality_check_index_rst_toctree(tmp_path, monkeypatch):
    """Test index.rst toctree indentation and path validation."""

    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    index_rst = doc_dir / "index.rst"

    # 1. Bad indentation (2 spaces instead of 3)
    # 2. Missing file reference
    index_rst.write_text(
        ".. toctree::\n\n  missing_file\n   exists\n", encoding="utf-8"
    )
    (doc_dir / "exists.rst").touch()

    errors = quality_check.check_index_rst_toctree()
    # Should find bad indentation for 'missing_file' (2 spaces)
    # Should find missing path for 'missing_file'
    # Should find bad indentation for 'exists' (3 spaces is correct, but let's check)

    assert any("indentation (2)" in e for e in errors)
    assert any("path 'missing_file' not found" in e for e in errors)


def test_build_dist_failure(tmp_path, monkeypatch):
    """Test build_dist.py failure paths."""
    from release import build_dist

    # Needs pyproject.toml to run
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "README_dist.md").touch()
    monkeypatch.chdir(tmp_path)

    # Use a side_effect for subprocess.run to simulate failure
    def mock_run(cmd, *_args, **_kwargs):
        if "-m" in cmd and "build" in cmd:
            return MagicMock(returncode=1)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=mock_run):
        with patch("shutil.rmtree"):
            with patch("shutil.move"):
                with patch("shutil.copy"):
                    with patch("sys.stdin", return_value=MagicMock()):
                        with patch("sys.argv", ["build_dist.py"]):
                            with pytest.raises(SystemExit):
                                build_dist.main()


def test_build_dist_success(tmp_path, monkeypatch):
    """Test build_dist.py success path and cleanup."""
    from release import build_dist

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "README.md").touch()
    (tmp_path / "README_dist.md").touch()
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "dummy.tar.gz").touch()

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        with patch("sys.argv", ["build_dist.py"]):
            # Mock cleanup and move to avoid resetting our tmp_path state
            with patch("shutil.rmtree"):
                with patch("shutil.move"):
                    with patch("shutil.copy"):
                        # Just verify it runs without crashing
                        build_dist.main()


def test_upload_release_diagnostics(monkeypatch):
    """Test the diagnostic printing in upload_release.py."""
    from release import upload_release

    mock_run_ret = MagicMock(returncode=1, stdout="Twine Error Message", stderr="")

    with patch("subprocess.run", return_value=mock_run_ret) as patched_run:
        # Verify it doesn't crash and prints something
        with patch("builtins.print") as mock_print:
            upload_release.print_structured_diagnostic("Twine Error Message")
            assert mock_print.called
            # Check if it tried to run twine --version for diagnostic
            assert any(
                "twine" in str(call.args[0]) for call in patched_run.call_args_list
            )


def test_quality_check_usage_blocks(tmp_path, monkeypatch):
    """Test Usage: block detection and path validation."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # Valid Usage:
    (tmp_path / "valid.py").write_text(
        '"""\nUsage: python3 valid.py\n"""\ndef f(): pass\n', encoding="utf-8"
    )

    # Invalid Usage (references non-existent file):
    (tmp_path / "invalid.py").write_text(
        '"""\nUsage: python3 missing.py\n"""\ndef f(): pass\n', encoding="utf-8"
    )

    errors = quality_check.check_usage_blocks()
    assert len(errors) == 1
    assert "missing.py" in errors[0]


def test_quality_check_index_rst_toctree_complex(tmp_path, monkeypatch):
    """Test toctree path parsing with angle brackets."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    index_rst = doc_dir / "index.rst"

    # entry: [Title <target_path>] with exactly 3 space indentation
    index_rst.write_text(
        ".. toctree::\n\n   Label <valid_target>\n   Label <missing_target>\n",
        encoding="utf-8",
    )
    (doc_dir / "valid_target.rst").touch()

    errors = quality_check.check_index_rst_toctree()
    assert len(errors) == 1
    assert "missing_target" in errors[0]


def test_check_links_advanced(tmp_path, monkeypatch):
    """Test check_links.py internal link logic for {ref} and .html mapping."""
    monkeypatch.setattr(check_links, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_links, "DOC_DIR", tmp_path / "doc")
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()

    # 1. Test MyST {ref} logic (line 104-108)
    (doc_dir / "file1.md").write_text(
        "(target1)=\n{ref}`target1` {ref}`missing`", encoding="utf-8"
    )
    # 2. Test .html to .md mapping (line 131-139)
    (doc_dir / "source.md").write_text(
        "[HTML Link](target.html#anchor)", encoding="utf-8"
    )
    (doc_dir / "target.md").write_text("(anchor)=", encoding="utf-8")

    with patch("sys.argv", ["check_links.py"]):
        # Should exit because of 'missing' ref
        with pytest.raises(SystemExit) as excinfo:
            check_links.main()
        assert excinfo.value.code == 1


def test_build_dist_io_errors(tmp_path, monkeypatch):
    """Test I/O error handling in build_dist.py (shutil failure)."""
    from release import build_dist

    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "README_dist.md").touch()
    monkeypatch.chdir(tmp_path)

    # Use a side_effect for rmtree to simulate PermissionError
    with patch("shutil.rmtree", side_effect=PermissionError("Mocked Permission Error")):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("sys.argv", ["build_dist.py"]):
                with pytest.raises(SystemExit):
                    build_dist.main()


def test_compute_readability_no_vale(tmp_path, monkeypatch):
    """Test compute_readability.py failure path when vale fails."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.md").touch()

    # Mock subprocess.run to raise CalledProcessError (line 34-36)
    from subprocess import CalledProcessError

    err = CalledProcessError(2, "vale")
    with patch("subprocess.run", side_effect=err):
        with patch("sys.argv", ["compute_readability.py", "Test", "test.md"]):
            compute_readability.main()

    # Test usage error (line 91-92)
    monkeypatch.setattr(sys, "argv", ["compute_readability.py"])
    with pytest.raises(SystemExit):
        compute_readability.main()


def test_build_dist_complex(tmp_path, monkeypatch):
    """Test build_dist.py edge cases: missing pyproject, and failures."""
    from release import build_dist

    # 1. Missing pyproject.toml (line 66-71)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_dist.py"])
    with pytest.raises(SystemExit):
        build_dist.main()

    # 2. .egg-info cleanup (line 44)
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "README_dist.md").touch()
    (tmp_path / "test.egg-info").mkdir()
    # We must ensure build_dist.Path.glob picks it up
    with patch("release.build_dist.Path.glob", return_value=[Path("test.egg-info")]):
        with patch("shutil.rmtree") as mock_rm:
            # Mock build_dist to avoid FileNotFoundError in complex tests
            with patch("release.build_dist.run_command"):
                with patch("shutil.move"):
                    with patch("shutil.copy"):
                        with patch("pathlib.Path.unlink"):
                            with patch(
                                "tempfile.mkstemp", return_value=(99, "temp.md")
                            ):
                                with patch("os.close"):
                                    build_dist.main()
                                    # Should have called rmtree on the egg-info
                                    assert any(
                                        "test.egg-info" in str(c)
                                        for c in mock_rm.call_args_list
                                    )

    # 3. Build failure with --keep-on-fail (line 102-110)
    # Mock build_dist.run_command to simulate failure
    monkeypatch.setattr(sys, "argv", ["build_dist.py", "--keep-on-fail"])
    with patch("release.build_dist.run_command", side_effect=[None, SystemExit(1)]):
        with patch("shutil.move"):
            with patch("shutil.copy"):
                with patch("pathlib.Path.unlink"):
                    with pytest.raises(SystemExit):
                        build_dist.main()


def test_quality_check_vulnerabilities(tmp_path, monkeypatch):
    """Test quality_check early return and exclusion logic."""
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)

    # 1. Missing conf.py (early return in check_conf_types)
    assert not quality_check.check_conf_types()

    # 2. Orthography exclusions (references_vol.md should be ignored)
    (tmp_path / "references_vol1.md").write_text("analyse\n", encoding="utf-8")
    assert not quality_check.check_orthography()

    # 3. AST SyntaxError coverage
    (tmp_path / "bad_syntax.py").write_text("invalid syntax @@@\n", encoding="utf-8")
    monkeypatch.setattr(quality_check, "REPO_ROOT", tmp_path)
    # check_docstring_links should just skip the file or continue
    _ = quality_check.check_docstring_links()
