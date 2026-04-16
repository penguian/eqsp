# Appendix D: Upload Guide

This guide details the procedures for uploading PyEQSP distributions and documentation to their respective hosting platforms using the project's automation suite.

> [!IMPORTANT]
> **Branch Assumption**: All release procedures (Step 1 onwards) assume you are starting from a clean and up-to-date **`main`** branch. Ensure all feature PRs have been merged before proceeding.

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

    > [!CAUTION]
    > **TestPyPI Immutability**: If an upload to TestPyPI fails *after* a successful partial upload, or if you need to fix a bug discovered during the check, you **must** increment the version number (e.g., `1.0b1` to `1.0b1.post1` or `1.0b2`) in `pyproject.toml`. TestPyPI does not allow re-uploading the same version string.

2.  **Troubleshooting & Bug Fixes**:
    If Step 1 (Build/Test) or Step 2 (Production) fails due to a code bug or metadata error:
    1.  **Switch to a new fix branch** from `main`.
    2.  Implement and verify the fix.
    3.  Create a Pull Request, complete the review, and **merge back to `main`**.
    4.  **Restart** the release process from Step 1 on the updated `main` branch.

3.  **Confirm and Upload to Production**:
    ```bash
    python3 release/upload_release.py --pypi
    ```

Check the TestPyPI/PyPI project pages for the updated distribution.

### 3. GitHub Synchronisation & CI Verification
Once the TestPyPI rendering is confirmed, commit the changes and trigger a final CI run on GitHub:

1. **Commit and Push**:
   ```bash
   # Create and switch to a new release branch (-b) starting from main.
   # This ensures the release is built on a stable, merged foundation.
   git checkout main
   git pull origin main
   git checkout -b release_1_0b1
   git add .
   git commit -m "Release 1.0b1: Open Beta Engagement Infrastructure"
   git push -u origin release_1_0b1
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
