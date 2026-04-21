# Appendix D: Upload Guide

This guide details the procedures for uploading PyEQSP distributions and documentation to their respective hosting platforms using the project's automation suite.

:::important
**Protected Branch Policy**: These procedures assume the **`main`** branch is protected. All changes, including version bumps, must be performed on a release branch and merged via Pull Request after verification.
:::

## Release Distribution (PyPI / TestPyPI)

To build and upload the package, use the scripts in the `release/` directory. These scripts ensure that all documentation links are converted to absolute GitHub URLs for correct rendering on project pages.

### 1. Create Release Branch & Version Bump
Start from an up-to-date `main` branch and create a dedicated staging area.
```bash
git checkout main
git pull origin main
git checkout -b release_branch_1_0b1
```
Edit `pyproject.toml` to set the new version (e.g., `1.0b1`).

### 2. Run Build with verification (TestPyPI)
From the **release branch**, run the pre-flight check:
```bash
python3 release/upload_release.py --testpypi
```
This script will:
*   Initialize the `release/pypi_readme_fix.py` logic to swap relative links.
*   Trigger `release/build_dist.py` for a clean `sdist` and `wheel` creation.
*   Verify the artifact using `twine check`.
*   Attempt an upload to the TestPyPI repository.

:::caution
**TestPyPI Immutability**: If an upload to TestPyPI fails *after* a successful partial upload, or if you need to fix a bug discovered during the check, you **must** increment the version number (e.g., `1.0b1` to `1.0b2`) in `pyproject.toml`. TestPyPI does not allow re-uploading the same version string.
:::

### 3. GitHub Synchronization & CI Verification
Once the TestPyPI rendering is confirmed, push your branch and open a PR.

1. **Push Branch**:
   ```bash
   git add .
   git commit -m "Release 1.0b1: Open Beta Engagement Infrastructure"
   git push -u origin release_branch_1_0b1
   ```

2. **Create Pull Request**:
   Create a PR from your release branch to `main`.

3. **Verify CI**:
   Ensure that the GitHub Actions "CI" and "Verify Distribution Build" workflows pass 100%. This serves as the final gatekeeper before the production upload.

### 4. Production PyPI Upload & Tagging
Once the PR is approved and merged into `main`:
```bash
# Switch to the updated main branch
git checkout main
git pull origin main

# Perform production upload
python3 release/upload_release.py --pypi

# Create and push the release tag
git tag release_1_0b1
git push origin release_1_0b1
```

:::caution
**PyPI Immutability & Failure Recovery**: Like TestPyPI, production PyPI is strictly immutable. If the upload fails *after* any artifact (sdist or wheel) has been successfully accepted, you cannot re-upload to that version string. To recover:
1. Fix the underlying issue (e.g., build error or metadata fix) on a new branch.
2. **Increment the version** in `pyproject.toml` (e.g., `1.0b1` → `1.0b2`).
3. Restart the entire release process from Step 1.
:::

## SourceForge Documentation Mirror Upload

To update the project documentation mirror at `http://pyeqsp.sourceforge.io`:

### 1. Generate & Upload
The documentation upload is semi-automated via the `release/upload_sourceforge.py` script.
```bash
# This script builds the docs and generates the scp command
python3 release/upload_sourceforge.py
```
After reviewing the generated command, execute it to upload the `doc/_build/html` contents to your SourceForge `htdocs` directory.

### 2. Verify Web Rendering
Check [http://pyeqsp.sourceforge.io](http://pyeqsp.sourceforge.io) to ensure all Markdown, mathematics symbols, and navigation elements render correctly.
