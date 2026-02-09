from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

from pathlib import Path

import pytest

from ml.config.replay_harness import ActorReplayConfig
from ml.config.replay_harness import ParquetLiveReplayHarnessConfig
from ml.config.replay_harness import StrategyReplayConfig
from ml.features.config import FeatureConfig
from ml.orchestration.parquet_live_replay_harness import run_parquet_live_replay_harness
from ml.tests.utils.model_artifacts import write_stub_onnx_artifact


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
)


@pytest.mark.integration
def test_parquet_live_replay_harness_smoke(
    tmp_path: Path,
    mock_onnx_runtime,
    onnx_session_stub_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.data import QuoteTick
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.objects import Quantity
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
    except Exception:
        pytest.skip("Backtest dependencies unavailable")

    from ml.core.common.database_lifecycle import DatabaseLifecycleComponent

    monkeypatch.setattr(
        DatabaseLifecycleComponent,
        "is_postgres_running",
        lambda _self: False,
    )

    mock_onnx_runtime.ort.InferenceSession.side_effect = lambda *_args, **_kwargs: (
        onnx_session_stub_factory(prediction=0.9, confidence=0.95)
    )
    mock_onnx_runtime.reload_modules("ml._imports")

    catalog_dir = tmp_path / "catalog"
    catalog = ParquetDataCatalog(str(catalog_dir))

    instrument = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
    bar_type = BarType(instrument.id, bar_spec)

    ts0 = 1_700_000_000_000_000_000
    bars: list[Bar] = []
    quotes: list[QuoteTick] = []
    close = 100.0
    for i in range(20):
        ts = ts0 + i * 60_000_000_000
        prev = close
        close = prev * (1.0 + (0.001 if i % 2 == 0 else -0.001))
        open_price = Price.from_str(f"{prev:.6f}")
        high_price = Price.from_str(f"{max(prev, close):.6f}")
        low_price = Price.from_str(f"{min(prev, close):.6f}")
        close_price = Price.from_str(f"{close:.6f}")
        bars.append(
            Bar(
                bar_type=bar_type,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=Quantity.from_int(1000),
                ts_event=ts,
                ts_init=ts,
            ),
        )
        bid_price = Price.from_str(f"{close - 0.01:.6f}")
        ask_price = Price.from_str(f"{close + 0.01:.6f}")
        quotes.append(
            QuoteTick(
                instrument_id=instrument.id,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=Quantity.from_int(100),
                ask_size=Quantity.from_int(120),
                ts_event=ts,
                ts_init=ts,
            ),
        )
    catalog.write_data(bars)
    catalog.write_data(quotes)

    model_path = tmp_path / "smoke_model.onnx"
    write_stub_onnx_artifact(model_path, content=b"dummy-onnx")

    output_dir = tmp_path / "out"
    actor_config = ActorReplayConfig(
        prediction_threshold=0.5,
        warm_up_period=1,
        feature_config=FeatureConfig(lookback_window=2),
        db_connection="postgresql://postgres:postgres@localhost:1/invalid",
    )
    strategy_config = StrategyReplayConfig(
        execute_trades=True,
        serialize_order_intents=True,
        min_confidence=0.5,
        subscribe_quote_ticks=True,
        quote_schema="mbp-1",
        max_quote_age_ms=120_000,
    )
    config = ParquetLiveReplayHarnessConfig(
        catalog_path=str(catalog_dir),
        instrument_ids=[str(instrument.id)],
        model_id="dummy",
        model_path=str(model_path),
        bar_spec="1-MINUTE-LAST",
        start_time=ts0,
        end_time=ts0 + 19 * 60_000_000_000,
        output_dir=str(output_dir),
        run_id="smoke",
        actor=actor_config,
        strategy=strategy_config,
    )

    result = run_parquet_live_replay_harness(config)

    assert result.bars_loaded == len(bars)
    assert result.quote_ticks_loaded == len(quotes)
    run_path = output_dir / "smoke"
    assert (run_path / "models" / "predictions.jsonl").exists()
    assert (run_path / "strategies" / "signals.jsonl").exists()
    assert (run_path / "orders" / "order_intents.jsonl").exists()
