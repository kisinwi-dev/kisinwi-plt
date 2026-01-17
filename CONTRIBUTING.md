# Contributing to Project

## Branch naming conventions
- `<sercice>/feature/<feature-name>` — new feature
- `<sercice>/fix/<bug-name>` — fix bugs
- `<sercice>/hotfix/<issue>` — hotfix
- `<sercice>/chore/<task>` — routine maintenance tasks
- `<sercice>/refactor/<module>` — refactoring

## Commit message conventions
We use conventional commits:

- `feat(module): short description` — new functionality
- `fix(module): short description` — fix bug
- `docs(module): short description` — documentation
- `chore(module): short description` — routine maintenance tasks
- `refactor(module): short description` — refactoring

### Example:

```
feat(routers): implement create dataset endpoint
fix(store): handle missing dataset error
```

## Pull Request Rules
- Link the PR to the corresponding issue (if available).
- PR description must include:
  - What was done
  - How to test
  - Related issues (Closes #, Related to #)
- Ensure commit messages follow the conventions above.
