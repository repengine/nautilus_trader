from __future__ import annotations

"""
Compatibility wrapper for legacy import path used by tests.

Delegates to ml.cli.build_tft_dataset.main.
"""

from ml.cli.build_tft_dataset import main  # noqa: E402


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    raise SystemExit(main())
