"""
Compatibility shim for `python -m ml.scripts.apply_migrations`.

Delegates to `ml.cli.apply_migrations` to preserve documented and Makefile
entry points.
"""

from __future__ import annotations

from ml.cli.apply_migrations import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":  # pragma: no cover - thin shim
    raise SystemExit(main())

