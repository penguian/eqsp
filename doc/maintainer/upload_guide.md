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

### 3. GitHub Synchronisation & CI Verification
Once the TestPyPI rendering is confirmed, commit the changes and trigger a final CI run on GitHub:

1. **Commit and Push**:
   ```bash
   git checkout -b release_v0.99.7
   git add .
   git commit -m "Release 0.99.7: CI hardening and quality guardrails"
   git push -u origin release_v0.99.7
   ```

2. **Create Pull Request**:
   Create a PR from your release branch to `main`.

3. **Verify CI**:
   Ensure that the GitHub Actions "CI" and "Verify Distribution Build" workflows pass 100%. This serves as the final gatekeeper before the production upload.

### 4. Production PyPI Upload
Once the PR is approved and the CI has passed:
```bash
python3 scripts/upload_release.py --pypi
```

## SourceForge Documentation Upload

To update the project website at `http://eqsp.sourceforge.net`:

### 1. Generate & Upload
The documentation upload is semi-automated via the `doc/ci_scripts/upload_sourceforge.py` script.
```bash
# This script builds the docs and generates the scp command
python3 doc/ci_scripts/upload_sourceforge.py
```
After reviewing the generated command, execute it to upload the `doc/_build/html` contents to your SourceForge `htdocs` directory.

### 2. Verify Web Rendering
Check [http://eqsp.sourceforge.net](http://eqsp.sourceforge.net) to ensure all Markdown, mathematics symbols, and navigation elements render correctly.
