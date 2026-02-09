from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from ml.data.ingest import common


def test_rate_limiter_wait_applies_sleep_when_called_too_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = iter([10.0, 10.1, 10.2, 20.0, 20.5])
    sleeps: list[float] = []

    def _monotonic() -> float:
        return next(timeline)

    monkeypatch.setattr(common, "monotonic", _monotonic)
    monkeypatch.setattr(common, "sleep", lambda delay: sleeps.append(delay))

    limiter = common.RateLimiter(per_minute=2)  # 30 second interval
    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(20.2, rel=0.0, abs=1e-9)


def test_load_progress_json_handles_missing_invalid_and_valid_files(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    assert common.load_progress_json(missing_path) == {}

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not-json", encoding="utf-8")
    assert common.load_progress_json(invalid_path) == {}

    valid_path = tmp_path / "valid.json"
    valid_path.write_text('{"cursor": 7, "status": "ok"}', encoding="utf-8")
    assert common.load_progress_json(valid_path) == {"cursor": 7, "status": "ok"}


def test_save_progress_json_uses_fallback_write_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "state.json"

    def _replace_raises(self: Path, _: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", _replace_raises)

    common.save_progress_json(target_path, {"cursor": 42})
    assert common.load_progress_json(target_path) == {"cursor": 42}


def test_save_progress_json_swallows_errors_when_both_writes_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "final.json"
    original_write_text = Path.write_text

    def _replace_raises(self: Path, _: Path) -> None:
        raise OSError("replace failed")

    def _write_text_selective(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if self == target_path:
            raise OSError("write failed")
        return cast(
            int,
            original_write_text(
                self,
                data,
                encoding=encoding,
                errors=errors,
                newline=newline,
            ),
        )

    monkeypatch.setattr(Path, "replace", _replace_raises)
    monkeypatch.setattr(Path, "write_text", _write_text_selective)

    common.save_progress_json(target_path, {"cursor": 99})
    assert common.load_progress_json(target_path) == {}
