import time

from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text

from ml.stores.strategy_store import StrategyStore
from pathlib import Path


def test_strategy_store_basic_write(tmp_path: Path) -> None:
    db_path = tmp_path / "strategy_store.db"
    store = StrategyStore(connection_string=f"sqlite:///{db_path}")

    now = int(time.time())
    store.write_signal(
        strategy_id="strat1",
        instrument_id="EUR/USD",
        signal_type="BUY",
        strength=0.7,
        model_predictions={"m1": 0.65},
        risk_metrics={"risk_score": 0.2},
        execution_params={"note": "test"},
        ts_event=now,
        is_live=False,
    )
    store.flush()

    with store.engine.connect() as conn:
        try:
            result = conn.execute(
                select(func.count()).select_from(store.strategy_signals_table),
            ).scalar()
        except Exception:
            result = conn.execute(text("SELECT COUNT(*) FROM ml_strategy_signals")).scalar()

    assert int(result or 0) == 1

    # Read-path check (SQLite-friendly fallback)
    start_ns = 0
    end_ns = int(time.time() * 1e9) + 10**6
    df = store.read_signals("strat1", "EUR/USD", start_ns, end_ns)
    assert not df.empty
