# SourceForge & TestPyPI Upload Guide

This guide details the procedures for uploading PyEQSP distributions and documentation to their respective hosting platforms using the project's automation suite.

## Release Distribution (PyPI / TestPyPI)

To build and upload the package, use the scripts in the `scripts/` directory. These scripts ensure that all documentation links are converted to absolute GitHub URLs for correct rendering on project pages.

### 1. Verification & Build
Before uploading, run the full verification suite with the pre-release build check:
```bash
python3 verify_all.py --pre-release
```
This command builds the distribution into `dist/` and runs `twine check` automatically.

### 2. TestPyPI Upload
To verify the rendering and installation on TestPyPI:
```bash
python3 scripts/upload_release.py --testpypi
```
Verify the installation in a clean environment:
```bash
python3 -m venv test_env && source test_env/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyeqsp
```

### 3. Production PyPI Upload
Once verified on TestPyPI:
```bash
python3 scripts/upload_release.py --pypi
```

## SourceForge Documentation Upload

To update the project website at `http://eqsp.sourceforge.net`:

### 1. Generate & Upload
The documentation upload is semi-automated via the `doc/maint/upload_sourceforge.py` script.
```bash
# This script builds the docs and generates the scp command
python3 doc/maint/upload_sourceforge.py
```
After reviewing the generated command, execute it to upload the `doc/_build/html` contents to your SourceForge `htdocs` directory.

### 2. Verify Web Rendering
Check [http://eqsp.sourceforge.net](http://eqsp.sourceforge.net) to ensure all Markdown, mathematics symbols, and navigation elements render correctly.
