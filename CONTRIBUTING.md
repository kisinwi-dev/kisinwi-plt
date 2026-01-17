# Contributing to Project

## Branch naming conventions
- `feature/<feature-name>` — new feature
- `fix/<bug-name>` — fix bugs
- `hotfix/<issue>` — hotfix
- `chore/<task>` — routine maintenance tasks
- `refactor/<module>` — refactoring

## Commit message conventions
We use conventional commits:

- `feat(module): short description` — new functionality
- `fix(module): short description` — fix bug
- `docs(module): short description` — documentation
- `chore(module): short description` — routine maintenance tasks
- `refactor(module): short description` — refactoring

### Example:

```
feat(data_manipulation/routers): implement create dataset endpoint
fix(data_manipulation/store): handle missing dataset error
```

## Pull Request Rules
- Link the PR to the corresponding issue (if available).
- PR description must include:
  - What was done
  - How to test
  - Related issues (Closes #, Related to #)
- Ensure commit messages follow the conventions above.
