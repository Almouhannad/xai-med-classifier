# Agent Workflow & Guidelines

Welcome, Agent. This project is optimized for automated workflows. Please adhere to the following rules and use the canonical commands provided.

## Canonical Commands

- `make setup`: Initialize environment and dependencies.
- `make lint`: Run ruff and black for code quality.
- `make test`: Execute the test suite with pytest.
- `make train-smoke`: Run a fast training cycle (1 epoch, small subset).
- `make explain-smoke`: Generate explanations for a few samples.
- `make report`: Build the final report in `artifacts/report/`.

## Working Rules

1. **Small PRs**: Keep changes focused and modular.
2. **No Large Artifacts**: Do not commit large checkpoints or datasets to the repo. Use `artifacts/` (gitignored).
3. **Test Driven**: Always add or update tests when modifying core logic.
4. **CLI First**: Logic should be in `src/xaimed/` and exposed via `src/xaimed/cli.py`.