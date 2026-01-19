# Streaming Checkpoint Key & Payload Guard Plan

## Goals
- Keep logging compliant with exception logging rules.
- Persist `checkpoint_key` across process boundaries in streaming plan payloads.
- Fix AutoGluon dependency guard usage in Chronos inference paths.

## Plan
1. Add `exc_info=True` to the invalid-payload metric log in `ml/actors/ml_domain_events.py`.
2. Extend streaming plan payload schema/serialization to include optional `checkpoint_key`.
3. Update tests to verify `checkpoint_key` is emitted when present and omitted when absent.
4. Use `check_ml_dependencies(["autogluon"])` in Chronos inference code paths.

## Validation
- `poetry run mypy ml --strict`
- `poetry ruff check ml`
- `make validate-fixtures`
- `poetry run coverage run -m pytest -k "<focused areas>"`
- `poetry run coverage report`
