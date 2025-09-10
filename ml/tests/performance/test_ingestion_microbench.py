from __future__ import annotations

import time

from ml.data.fixtures import make_tbbo_fixture, make_trades_fixture, make_mbp10_fixture


def test_ingestion_fixture_generation_micro_bench() -> None:
    t0 = time.perf_counter()
    make_tbbo_fixture(rows=120)
    make_trades_fixture(rows=240)
    make_mbp10_fixture(rows=60)
    t1 = time.perf_counter()
    # Keep under a small budget locally to catch regressions
    assert (t1 - t0) < 0.25
