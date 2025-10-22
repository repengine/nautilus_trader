"""Compatibility wrapper for CLI vintage age conversion."""

from __future__ import annotations

from ml.cli.convert_vintage_age import main


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    raise SystemExit(main())
