# SourceForge & TestPyPI Upload Guide

This guide details the procedures for uploading PyEQSP distributions and documentation to their respective hosting platforms.

## TestPyPI Upload Procedure

To verify the distribution before a public PyPI release:

1. **Clean & Build**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python3 -m build
   ```

2. **Upload via Twine**:
   ```bash
   python3 -m twine upload --repository testpypi dist/*
   ```

3. **Verify Installation**:
   ```bash
   python3 -m venv test_env
   source test_env/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyeqsp
   ```

## SourceForge Documentation Upload

To update the project website at `http://eqsp.sourceforge.net`:

1. **Generate Documentation**:
   ```bash
   cd doc
   make html
   ```

2. **Upload via SCP**:
   Upload the contents of `doc/_build/html` to your SourceForge htdocs directory.
   ```bash
   # Replace USER with your SourceForge username
   scp -r doc/_build/html/* USER@web.sourceforge.net:/home/project-web/eqsp/htdocs/
   ```

3. **Verify Web Rendering**:
   Check [http://eqsp.sourceforge.net](http://eqsp.sourceforge.net) to ensure all Markdown and mathematics symbols render correctly.
