from __future__ import annotations

from ml.validate_imports import probe_import


def test_probe_import_ml_package() -> None:
    ok, msg = probe_import("ml")
    assert ok, f"ml should import: {msg}"
