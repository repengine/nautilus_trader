#!/usr/bin/env python3
"""
Simple HPO sweep for TFT teacher (BCE).

Thin CLI wrapper delegating to canonical training helpers.
"""

from __future__ import annotations

from collections.abc import Callable

from ml._imports import HAS_OPTUNA
from ml.training.teacher import hpo_tft as _hpo


try:
    from ml.training.teacher.tft_cli import main as _imported_teacher_main
except Exception:  # pragma: no cover - optional dependency guard
    _TEACHER_MAIN: Callable[[list[str] | None], int] | None = None
else:
    _TEACHER_MAIN = _imported_teacher_main


def teacher_main(args: list[str] | None = None) -> int:
    """Proxy to the teacher CLI entrypoint with optional dependency guard."""
    if _TEACHER_MAIN is None:
        raise RuntimeError("teacher_main is unavailable in this environment")
    return int(_TEACHER_MAIN(args))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint delegating to the shared task helper."""
    return _hpo.main(argv, teacher_main=teacher_main, has_optuna=HAS_OPTUNA)


__all__ = ["HAS_OPTUNA", "main", "teacher_main"]


if __name__ == "__main__":
    raise SystemExit(main())
