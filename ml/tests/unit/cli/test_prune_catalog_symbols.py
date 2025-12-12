from __future__ import annotations

from pathlib import Path

import pytest

from ml.cli import prune_catalog_symbols as cli


def _mk(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def test_prune_catalog_symbols_removes_non_matching(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    quote_tick = catalog / "data" / "quote_tick"
    bar = catalog / "data" / "bar"
    _mk(quote_tick / "AAPL.EQUS")
    _mk(quote_tick / "AAPL.XNAS")
    _mk(bar / "AAPL.EQUS-1-MINUTE-LAST-EXTERNAL")
    _mk(bar / "AAPL.XNAS-1-MINUTE-LAST-EXTERNAL")

    result = cli.main(
        [
            str(catalog),
            "--suffix",
            ".EQUS",
            "--apply",
        ],
    )

    assert result == 0
    assert (quote_tick / "AAPL.EQUS").exists()
    assert not (quote_tick / "AAPL.XNAS").exists()
    assert (bar / "AAPL.EQUS-1-MINUTE-LAST-EXTERNAL").exists()
    assert not (bar / "AAPL.XNAS-1-MINUTE-LAST-EXTERNAL").exists()


def test_prune_catalog_symbols_dry_run_keeps_dirs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    quote_tick = catalog / "data" / "quote_tick"
    _mk(quote_tick / "MSFT.XNYS")

    result = cli.main(
        [
            str(catalog),
            "--suffix",
            ".EQUS",
        ],
    )

    assert result == 0
    assert (quote_tick / "MSFT.XNYS").exists()
