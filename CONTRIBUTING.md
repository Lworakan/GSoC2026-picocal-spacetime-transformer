# Contributing

This project is developed openly as part of CERN-HSF GSoC 2026. Contributions,
issues, and review from the LHCb group and the wider community are welcome.

## Git workflow

We follow a standard feature-branch workflow:

1. Create a branch off `main`: `git checkout -b feature/short-description`
2. Make small, focused commits with clear messages.
3. Push and open a Pull Request describing the change and the motivation.
4. CI (tests + lint) must pass before merge.
5. At least one reviewer (typically the mentor) approves before merging.

Commit and push **regularly** — incremental updates are far easier to review
than a single large change.

## Commit messages

Use the imperative mood and keep the summary under ~72 characters:

```
Add k-NN graph construction for PicoCal cells

Build edges from the 8 nearest neighbours in (x, y) and attach
cell energy and timing as node features.
```

## Code style

- Format and lint with `ruff` (`ruff check .` and `ruff format .`).
- Add type hints; `mypy src` should pass.
- New functionality needs a unit test under `tests/`.

## Tests

```bash
pytest
```

## Reproducibility

Any result that goes into the report or a blog post must be reproducible from a
committed config file and a tagged release.
