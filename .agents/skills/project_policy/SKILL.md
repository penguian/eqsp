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

## Mandatory Check
Before performing any technical task, you MUST check the `.antigravityrules` file in the root directory for any task-specific or project-wide constraints.
