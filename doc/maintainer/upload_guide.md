# SourceForge & TestPyPI Upload Guide

This guide details the procedures for uploading PyEQSP distributions and documentation to their respective hosting platforms using the project's automation suite.

## Release Distribution (PyPI / TestPyPI)

To build and upload the package, use the scripts in the `release/` directory. These scripts ensure that all documentation links are converted to absolute GitHub URLs for correct rendering on project pages.

1.  **Run Build with verification**:
    ```bash
    python3 release/upload_release.py --testpypi
    ```
    This script will:
    *   Initialize the `release/pypi_readme_fix.py` logic to swap relative links.
    *   Trigger `release/build_dist.py` for a clean `sdist` and `wheel` creation.
    *   Verify the artifact using `twine check`.
    *   Attempt an upload to the TestPyPI repository.
2.  **Confirm and Upload to Production**:
    ```bash
    python3 release/upload_release.py --pypi
    ```

Check the TestPyPI/PyPI project pages for the updated distribution.

### 3. GitHub Synchronisation & CI Verification
Once the TestPyPI rendering is confirmed, commit the changes and trigger a final CI run on GitHub:

1. **Commit and Push**:
   ```bash
   git checkout -b release_v0.99.8
   git add .
   git commit -m "Release 0.99.8: Infrastructure hardening and JOSS submission finalization"
   git push -u origin release_v0.99.8
   ```

2. **Create Pull Request**:
   Create a PR from your release branch to `main`.

3. **Verify CI**:
   Ensure that the GitHub Actions "CI" and "Verify Distribution Build" workflows pass 100%. This serves as the final gatekeeper before the production upload.

### 4. Production PyPI Upload
Once the PR is approved and the CI has passed:
```bash
python3 release/upload_release.py --pypi
```

## SourceForge Documentation Upload

To update the project website at `http://eqsp.sourceforge.net`:

### 1. Generate & Upload
The documentation upload is semi-automated via the `release/upload_sourceforge.py` script.
```bash
# This script builds the docs and generates the scp command
python3 release/upload_sourceforge.py
```
After reviewing the generated command, execute it to upload the `doc/_build/html` contents to your SourceForge `htdocs` directory.

### 2. Verify Web Rendering
Check [http://eqsp.sourceforge.net](http://eqsp.sourceforge.net) to ensure all Markdown, mathematics symbols, and navigation elements render correctly.
