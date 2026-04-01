"""
Test suite covering distribution and packaging automation scripts.
"""

# pylint: disable=wrong-import-position
import os
import sys
from unittest.mock import MagicMock, patch

# Update sys.path to import from the root module without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from release import build_dist, upload_release  # noqa: E402
from release import pypi_readme_fix as readme_fix  # noqa: E402


def test_pypi_readme_fix(tmp_path, monkeypatch):
    """Test generating a README with absolute PyPI links."""
    # Mock files
    readme_md = tmp_path / "README.md"

    readme_md.write_text(
        "Here is a [link](some/file.md) and another [dir](some/dir/).\n"
        "And an external [ext](https://example.com/file).",
        encoding="utf-8",
    )

    # Change working directory so it finds the files
    monkeypatch.chdir(tmp_path)

    # Run
    readme_fix.main()

    readme_dist = tmp_path / "README_dist.md"
    assert readme_dist.exists()

    content = readme_dist.read_text(encoding="utf-8")

    assert (
        "[link](https://github.com/penguian/pyeqsp/blob/main/some/file.md)" in content
    )
    assert "[dir](https://github.com/penguian/pyeqsp/tree/main/some/dir/)" in content
    assert "[ext](https://example.com/file)" in content


@patch("release.build_dist.subprocess.run")
@patch("release.build_dist.shutil.rmtree")
def test_build_dist(mock_rmtree, mock_run, tmp_path, monkeypatch):
    """Test orchestration of the packaging build cycle."""
    # pylint: disable=unused-argument
    # Setup dummy project root setup
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "README.md").touch()
    (tmp_path / "README_dist.md").touch()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "dummy.whl").touch()

    monkeypatch.chdir(tmp_path)

    # Needs to return success
    mock_run.return_value = MagicMock(returncode=0)

    # Mock sys.argv to prevent argparse from seeing pytest args
    with patch("sys.argv", ["build_dist.py"]):
        build_dist.main()

    # Verify calls
    assert mock_run.call_count == 3
    calls = mock_run.call_args_list

    assert "pypi_readme_fix.py" in calls[0][0][0][1]
    assert "build" in calls[1][0][0]

    # The last call should check the dummy file inside dist
    assert "check" in calls[2][0][0]
    assert "dist/dummy.whl" in calls[2][0][0]


@patch("release.upload_release.check_credentials")
@patch("release.upload_release.subprocess.run")
def test_upload_release(mock_run, mock_check, tmp_path, monkeypatch):
    """Test automated PyPI upload cycle and credential detection."""
    # Setup dummy project root and dist contents
    (tmp_path / "pyproject.toml").touch()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "dummy.whl").touch()

    monkeypatch.chdir(tmp_path)

    # subprocess.run is called for build_dist.py and twine upload
    mock_run.return_value = MagicMock(returncode=0, stdout="Mocked Success", stderr="")

    # Simulate passing CLI arguments to argparse
    monkeypatch.setattr("sys.argv", ["upload_release.py", "--testpypi"])

    upload_release.main()

    mock_check.assert_called_once()
    assert mock_run.call_count == 2

    calls = mock_run.call_args_list
    assert "release/build_dist.py" in calls[0][0][0][1]

    # Twine cmd should reflect --testpypi
    twine_cmd = calls[1][0][0]
    assert "upload" in twine_cmd
    assert "--repository" in twine_cmd
    assert "testpypi" in twine_cmd
    assert "dist/dummy.whl" in twine_cmd
