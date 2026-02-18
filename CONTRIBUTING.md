# Contributing

Thanks for contributing.

## Development setup

```bash
make install
```

## Before opening a PR

```bash
make lint
make test
```

## Scope and data policy

- This repo is synthetic-only.
- Do not add real customer data, secrets, or credentials.
- Generated outputs (`data/`, `reports/`, local DBs) must not be committed.

## Style

- Keep changes focused and readable.
- Add/adjust tests for behavior changes.
- Prefer small PRs with clear commit messages.
