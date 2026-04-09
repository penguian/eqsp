---
name: Project Git Policy
description: Standards for interaction with version control in the PyEQSP repository.
---
# PyEQSP Project Policy

This skill defines the strict boundaries for AI assistants when interacting with the PyEQSP repository.

## Git Interaction Restrictions
You MUST NEVER execute `git commit` or `git push` on behalf of the user. Version control state is the sole responsibility of the user.

### Acceptable Actions:
- `git status`
- `git add`
- `git diff`
- Providing specific `git commit` commands in your response for the user to execute manually.
- **Sphinx Documentation**: When building HTML documentation or running doctests, always use the local virtual environment at `.venvs/.venv_sys` as it contains the necessary system dependencies (Mayavi, VTK). Use the `--venv .venvs/.venv_sys` flag when running `validation/verify_all.py` for doc tasks.

## Mandatory Check
Before performing any technical task, you MUST check if an `.antigravityrules` file exists in the root directory. If it does, follow any task-specific or project-wide constraints defined within it.
