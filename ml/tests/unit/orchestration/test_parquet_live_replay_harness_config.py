from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ml.config.base import AccountMode
from ml.config.base import ExecutionValidationMode
from ml.config.base import ShortEntryPolicy
from ml.config.replay_harness import ActorReplayConfig
from ml.config.replay_harness import ParquetLiveReplayHarnessConfig
from ml.config.replay_harness import StrategyReplayConfig
from ml.orchestration import parquet_live_replay_harness
from ml.orchestration.parquet_live_replay_harness import _build_actor_config
from ml.orchestration.parquet_live_replay_harness import _build_engine
from ml.orchestration.parquet_live_replay_harness import _build_fallback_equity
from ml.orchestration.parquet_live_replay_harness import _build_strategy_config
from ml.orchestration.parquet_live_replay_harness import _configure_environment
from ml.orchestration.parquet_live_replay_harness import _fetch_replay_summary_counts
from ml.orchestration.parquet_live_replay_harness import _infer_price_precisions
from ml.orchestration.parquet_live_replay_harness import _load_bars
from ml.orchestration.parquet_live_replay_harness import _load_quote_ticks
from ml.orchestration.parquet_live_replay_harness import _normalize_instrument_ids
from ml.orchestration.parquet_live_replay_harness import _persist_backtest_result
from ml.orchestration.parquet_live_replay_harness import _persist_replay_summary
from ml.orchestration.parquet_live_replay_harness import _price_increment_from_precision
from ml.orchestration.parquet_live_replay_harness import _resolve_instruments
from ml.orchestration.parquet_live_replay_harness import _resolve_replay_db_connection
from ml.orchestration.parquet_live_replay_harness import _resolve_risk_config
from ml.orchestration.parquet_live_replay_harness import _resolve_output_path
from ml.orchestration.parquet_live_replay_harness import _sanitize_backtest_payload
from ml.orchestration.parquet_live_replay_harness import _split_symbol_and_venue
from ml.orchestration.parquet_live_replay_harness import _try_catalog_instrument
from ml.orchestration.parquet_live_replay_harness import _update_price_precision
from ml.orchestration.parquet_live_replay_harness import run_parquet_live_replay_harness
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskLiquidationConfig
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
)


def _build_harness_config(
    tmp_path: Path,
    *,
    strategy: StrategyReplayConfig | None = None,
    actor: ActorReplayConfig | None = None,
    run_id: str | None = "run-1",
    output_dir: str | None = None,
) -> ParquetLiveReplayHarnessConfig:
    return ParquetLiveReplayHarnessConfig(
        catalog_path=str(tmp_path / "catalog"),
        instrument_ids=["SPY.XNAS"],
        model_id="model-1",
        model_path=str(tmp_path / "model.onnx"),
        run_id=run_id,
        output_dir=output_dir,
        strategy=strategy or StrategyReplayConfig(),
        actor=actor or ActorReplayConfig(),
    )


def test_build_strategy_config_sets_exit_policy_max_holding_ms() -> None:
    strategy = StrategyReplayConfig(max_holding_ms=120_000)
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.exit_policy_config is not None
    assert cfg.exit_policy_config.max_holding_ms == 120_000
    assert cfg.exit_policy_config.stop_loss_pct == strategy.stop_loss_pct
    assert cfg.exit_policy_config.take_profit_pct == strategy.take_profit_pct


def test_build_strategy_config_includes_risk_config() -> None:
    liquidation = RiskLiquidationConfig(
        enabled=True,
        daily_loss_limit_pct=0.1,
    )
    risk_config = RiskConfig(
        allow_reduce_only_when_halted=False,
        liquidation_config=liquidation,
    )
    strategy = StrategyReplayConfig(risk_config=risk_config)
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.risk_config is not None
    assert cfg.risk_config.allow_reduce_only_when_halted is False
    assert cfg.risk_config.liquidation_config is not None
    assert cfg.risk_config.liquidation_config.daily_loss_limit_pct == 0.1


def test_build_strategy_config_includes_account_and_validation_mode() -> None:
    strategy = StrategyReplayConfig(
        account_mode=AccountMode.MARGIN,
        short_entry_policy=ShortEntryPolicy.ALLOW,
        execution_validation_mode=ExecutionValidationMode.MARKET,
    )
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.account_mode == AccountMode.MARGIN
    assert cfg.short_entry_policy == ShortEntryPolicy.ALLOW
    assert cfg.execution_config is not None
    assert cfg.execution_config.validation_mode == ExecutionValidationMode.MARKET


def test_build_strategy_config_includes_positions_log_degraded_flag() -> None:
    strategy = StrategyReplayConfig(positions_log_degraded_in_backtest=True)
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    cfg = _build_strategy_config(
        strategy_config=strategy,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        strategy_id="STRAT-1",
    )

    assert cfg.positions_config is not None
    assert cfg.positions_config.log_degraded_in_backtest is True


def test_build_actor_config_includes_actor_overrides() -> None:
    actor = ActorReplayConfig(
        publish_signals=False,
        log_predictions=True,
        signal_strategy="momentum",
        min_signal_separation_bars=1,
        use_dummy_stores=True,
    )
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    bar_type = BarType(instrument_id, BarSpecification.from_str("1-MINUTE-LAST"))

    cfg = _build_actor_config(
        actor_config=actor,
        model_id="model-1",
        model_path="model.onnx",
        bar_type=bar_type,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        db_connection=actor.db_connection,
    )

    assert cfg.publish_signals is False
    assert cfg.log_predictions is True
    assert cfg.signal_strategy == "momentum"
    assert cfg.min_signal_separation_bars == 1
    assert cfg.use_dummy_stores is True


def test_build_actor_config_uses_defaults_when_overrides_missing() -> None:
    actor = ActorReplayConfig()
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    bar_type = BarType(instrument_id, BarSpecification.from_str("1-MINUTE-LAST"))

    cfg = _build_actor_config(
        actor_config=actor,
        model_id="model-1",
        model_path="model.onnx",
        bar_type=bar_type,
        instrument_id=instrument_id,
        actor_id="ACTOR-1",
        db_connection=actor.db_connection,
    )

    assert cfg.signal_strategy == "threshold"
    assert cfg.min_signal_separation_bars == 3


def test_strategy_replay_config_rejects_risk_and_liquidation_config() -> None:
    with pytest.raises(ValueError, match="risk_config and liquidation_config"):
        StrategyReplayConfig(
            risk_config=RiskConfig(),
            liquidation_config=RiskLiquidationConfig(enabled=True),
        )


def test_sanitize_backtest_payload_replaces_nan_with_none() -> None:
    payload = {
        "stats_returns": {
            "Sharpe Ratio (252 days)": float("nan"),
            "Returns Average (252 days)": 0.1,
        },
    }

    nan_keys = _sanitize_backtest_payload(payload)

    assert nan_keys == ["Sharpe Ratio (252 days)"]
    assert payload["stats_returns"]["Sharpe Ratio (252 days)"] is None


def test_resolve_output_path_none_when_output_dir_missing(tmp_path: Path) -> None:
    config = _build_harness_config(tmp_path, output_dir=None)

    assert _resolve_output_path(config, "run-1") is None


def test_resolve_output_path_creates_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config = _build_harness_config(tmp_path, output_dir=str(output_dir))

    result = _resolve_output_path(config, "run-2")

    assert result == output_dir / "run-2"
    assert result.exists()


def test_resolve_output_path_raises_when_directory_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _build_harness_config(tmp_path, output_dir=str(tmp_path / "outputs"))

    def _raise_mkdir(self: Path, *, parents: bool = False, exist_ok: bool = False) -> None:
        del parents, exist_ok
        raise OSError("cannot create")

    monkeypatch.setattr(Path, "mkdir", _raise_mkdir)

    with pytest.raises(OSError, match="cannot create"):
        _resolve_output_path(config, "run-3")


def test_configure_environment_sets_expected_variables(tmp_path: Path) -> None:
    config = _build_harness_config(tmp_path, output_dir=str(tmp_path / "store"))
    output_path = tmp_path / "store" / "run-4"

    _configure_environment(config, output_path)

    assert parquet_live_replay_harness.os.environ["ML_FILE_STORE_PATH"] == str(output_path)
    assert parquet_live_replay_harness.os.environ["ML_TFT_ALLOW_PARQUET_FALLBACK"] == "1"


def test_resolve_replay_db_connection_handles_dummy_and_explicit_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dummy_config = _build_harness_config(tmp_path, actor=ActorReplayConfig(use_dummy_stores=True))
    assert _resolve_replay_db_connection(dummy_config) is None

    explicit_config = _build_harness_config(
        tmp_path,
        actor=ActorReplayConfig(db_connection="  postgresql://localhost/test  "),
    )
    assert _resolve_replay_db_connection(explicit_config) == "postgresql://localhost/test"

    dynamic_config = _build_harness_config(tmp_path)
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "collect_postgres_candidates",
        lambda _role: SimpleNamespace(urls=["postgresql://candidate"]),
    )
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "select_first_working_connection",
        lambda urls: urls[0],
    )
    assert _resolve_replay_db_connection(dynamic_config) == "postgresql://candidate"


def test_resolve_replay_db_connection_returns_none_on_candidates_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _build_harness_config(tmp_path)
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "collect_postgres_candidates",
        lambda _role: (_ for _ in ()).throw(RuntimeError("discovery failed")),
    )

    assert _resolve_replay_db_connection(config) is None


@dataclass
class _ScalarResult:
    value: int | None

    def scalar(self) -> int | None:
        return self.value


class _Connection:
    def __init__(self, values: list[int], failure_indices: set[int], index_ref: dict[str, int]) -> None:
        self._values = values
        self._failure_indices = failure_indices
        self._index_ref = index_ref

    def execute(self, _statement: Any, _params: dict[str, str]) -> _ScalarResult:
        idx = self._index_ref["value"]
        self._index_ref["value"] = idx + 1
        if idx in self._failure_indices:
            raise RuntimeError("query failure")
        return _ScalarResult(self._values[idx])


class _ConnectionContext:
    def __init__(self, conn: _Connection) -> None:
        self._conn = conn

    def __enter__(self) -> _Connection:
        return self._conn

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None


class _Engine:
    def __init__(self, values: list[int], failure_indices: set[int] | None = None) -> None:
        self._values = values
        self._failure_indices = failure_indices or set()
        self._index_ref: dict[str, int] = {"value": 0}

    def begin(self) -> _ConnectionContext:
        return _ConnectionContext(_Connection(self._values, self._failure_indices, self._index_ref))


def test_fetch_replay_summary_counts_handles_success_and_failures() -> None:
    counts = _fetch_replay_summary_counts(_Engine([2, 3, 4]), run_id="run-1")
    assert counts == {"fills": 2, "halts": 3, "sizing_rejects": 4}

    counts_with_failure = _fetch_replay_summary_counts(_Engine([2, 3, 4], failure_indices={1}), run_id="run-1")
    assert counts_with_failure == {"fills": 2, "halts": 0, "sizing_rejects": 4}


def test_persist_replay_summary_early_returns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backtest_result = SimpleNamespace(
        run_started=10,
        run_finished=20,
        backtest_start=10,
        backtest_end=20,
        total_orders=1,
        total_positions=0,
    )
    no_store_config = _build_harness_config(
        tmp_path,
        strategy=StrategyReplayConfig(use_strategy_store=False),
    )
    _persist_replay_summary(
        config=no_store_config,
        run_id="run-1",
        instrument_ids=(InstrumentId.from_str("SPY.XNAS"),),
        backtest_result=backtest_result,
        db_connection="postgresql://localhost/test",
    )

    config = _build_harness_config(tmp_path)
    monkeypatch.setattr(
        parquet_live_replay_harness.EngineManager,
        "get_engine",
        lambda _conn: (_ for _ in ()).throw(RuntimeError("no engine")),
    )
    _persist_replay_summary(
        config=config,
        run_id="run-2",
        instrument_ids=(InstrumentId.from_str("SPY.XNAS"),),
        backtest_result=backtest_result,
        db_connection="postgresql://localhost/test",
    )


def test_persist_replay_summary_writes_to_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _build_harness_config(tmp_path)
    engine = object()
    captured: dict[str, Any] = {}

    class _FakeStore:
        def __init__(self, *, connection_string: str, run_id: str) -> None:
            captured["connection_string"] = connection_string
            captured["run_id"] = run_id

        def write_replay_summary(self, summary: Any, *, publish_bus: bool) -> None:
            captured["summary"] = summary
            captured["publish_bus"] = publish_bus

        def flush(self) -> None:
            captured["flushed"] = True

    monkeypatch.setattr(parquet_live_replay_harness.EngineManager, "get_engine", lambda _conn: engine)
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "_fetch_replay_summary_counts",
        lambda _engine, _run_id: {"fills": 9, "halts": 2, "sizing_rejects": 1},
    )
    monkeypatch.setattr(parquet_live_replay_harness, "StrategyStore", _FakeStore)
    monkeypatch.setattr(parquet_live_replay_harness.time, "time_ns", lambda: 77)
    backtest_result = SimpleNamespace(
        run_started=11,
        run_finished=22,
        backtest_start=11,
        backtest_end=22,
        total_orders=5,
        total_positions=3,
    )

    _persist_replay_summary(
        config=config,
        run_id="run-3",
        instrument_ids=(InstrumentId.from_str("SPY.XNAS"),),
        backtest_result=backtest_result,
        db_connection="postgresql://localhost/test",
    )

    assert captured["connection_string"] == "postgresql://localhost/test"
    assert captured["run_id"] == "run-3"
    assert captured["publish_bus"] is False
    assert captured["flushed"] is True
    assert captured["summary"].total_fills == 9


def test_normalize_instrument_ids_validates_inputs() -> None:
    ids = _normalize_instrument_ids(["SPY", "SPY", "QQQ.XNAS"], fallback_venue="XNAS")
    assert tuple(str(inst) for inst in ids) == ("SPY.XNAS", "QQQ.XNAS")

    with pytest.raises(ValueError, match="instrument_ids cannot contain empty values"):
        _normalize_instrument_ids(["  "], fallback_venue="XNAS")

    with pytest.raises(ValueError, match="instrument_ids must contain at least one entry"):
        _normalize_instrument_ids([], fallback_venue="XNAS")


def test_resolve_instruments_uses_catalog_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = InstrumentId.from_str("SPY.XNAS")
    fallback = {"instrument": "fallback"}
    monkeypatch.setattr(parquet_live_replay_harness, "_try_catalog_instrument", lambda _catalog, _inst: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_build_fallback_equity", lambda *_args, **_kwargs: fallback)

    resolved = _resolve_instruments(
        catalog=SimpleNamespace(),
        instrument_ids=(inst,),
        fallback_venue=Venue("XNAS"),
        price_precisions={inst: 4},
    )
    assert resolved[inst] == fallback

    monkeypatch.setattr(
        parquet_live_replay_harness,
        "_build_fallback_equity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad fallback")),
    )
    with pytest.raises(ValueError, match="Unable to resolve instrument"):
        _resolve_instruments(
            catalog=SimpleNamespace(),
            instrument_ids=(inst,),
            fallback_venue=Venue("XNAS"),
            price_precisions={inst: 2},
        )


def test_split_symbol_and_venue_fallback_logic() -> None:
    assert _split_symbol_and_venue("SPY.XNAS", "SIM") == ("SPY", "XNAS")
    assert _split_symbol_and_venue("SPY", "SIM") == ("SPY", "SIM")


def test_update_price_precision_and_infer_precision_from_bars_and_ticks() -> None:
    inst = InstrumentId.from_str("SPY.XNAS")
    precisions: dict[InstrumentId, int] = {}
    _update_price_precision(precisions, inst, None)
    _update_price_precision(precisions, inst, SimpleNamespace(precision=2))
    _update_price_precision(precisions, inst, SimpleNamespace(precision=4))
    assert precisions[inst] == 4

    bars = [
        SimpleNamespace(
            bar_type=SimpleNamespace(instrument_id=inst),
            open=SimpleNamespace(precision=2),
            high=SimpleNamespace(precision=3),
            low=SimpleNamespace(precision=2),
            close=SimpleNamespace(precision=5),
        ),
    ]
    ticks = [
        SimpleNamespace(
            instrument_id=inst,
            bid_price=SimpleNamespace(precision=4),
            ask_price=SimpleNamespace(precision=6),
        ),
    ]
    inferred = _infer_price_precisions(bars, ticks)
    assert inferred[inst] == 6


def test_price_increment_and_fallback_equity_builder() -> None:
    assert str(_price_increment_from_precision(0)) == "1"
    assert str(_price_increment_from_precision(3)) == "0.001"
    with pytest.raises(ValueError, match="price_precision must be >= 0"):
        _price_increment_from_precision(-1)

    inst = InstrumentId.from_str("SPY.XNAS")
    equity = _build_fallback_equity(inst, symbol="SPY", venue="XNAS", price_precision=2)
    assert str(equity.id) == "SPY.XNAS"
    assert equity.price_precision == 2


def test_build_engine_adds_venues_and_instruments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _build_harness_config(tmp_path, strategy=StrategyReplayConfig(account_mode=AccountMode.MARGIN))
    captured: dict[str, Any] = {}

    class _FakeEngine:
        def __init__(self, *, config: Any) -> None:
            captured["config"] = config
            captured["venues"] = []
            captured["instruments"] = []

        def add_venue(self, **kwargs: Any) -> None:
            cast_list = captured["venues"]
            cast_list.append(kwargs)

        def add_instrument(self, instrument: Any) -> None:
            cast_list = captured["instruments"]
            cast_list.append(instrument)

    monkeypatch.setattr(parquet_live_replay_harness, "BacktestEngine", _FakeEngine)
    monkeypatch.setattr(parquet_live_replay_harness, "BacktestEngineConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(parquet_live_replay_harness, "TraderId", lambda value: value)
    monkeypatch.setattr(parquet_live_replay_harness, "LoggingConfig", lambda *, log_level: {"log_level": log_level})
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "PortfolioConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(parquet_live_replay_harness, "Money", lambda amount, currency: (amount, currency))
    monkeypatch.setattr(parquet_live_replay_harness, "Venue", lambda value: value)

    inst = InstrumentId.from_str("SPY.XNAS")
    _build_engine(config=config, instruments={inst: object()})

    assert len(captured["venues"]) == 1
    assert len(captured["instruments"]) == 1


def test_load_bars_and_quote_ticks_sort_and_handle_missing_rows() -> None:
    class _BarTypeProxy:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    bar_a = SimpleNamespace(ts_event=2)
    bar_b = SimpleNamespace(ts_event=1)
    tick_a = SimpleNamespace(ts_event=3)
    tick_b = SimpleNamespace(ts_event=1)

    class _Catalog:
        def bars(self, *, bar_types: list[str], start: Any, end: Any) -> list[Any]:
            del start, end
            if bar_types[0] == "present":
                return [bar_a, bar_b]
            return []

        def quote_ticks(self, *, instrument_ids: list[str], start: Any, end: Any) -> list[Any]:
            del start, end
            if instrument_ids[0] == "SPY.XNAS":
                return [tick_a, tick_b]
            return []

    bars = _load_bars(
        catalog=_Catalog(),
        bar_types={
            InstrumentId.from_str("SPY.XNAS"): _BarTypeProxy("present"),
            InstrumentId.from_str("QQQ.XNAS"): _BarTypeProxy("missing"),
        },
        start_time=None,
        end_time=None,
    )
    ticks = _load_quote_ticks(
        catalog=_Catalog(),
        instrument_ids=(InstrumentId.from_str("SPY.XNAS"), InstrumentId.from_str("QQQ.XNAS")),
        start_time=None,
        end_time=None,
    )

    assert [bar.ts_event for bar in bars] == [1, 2]
    assert [tick.ts_event for tick in ticks] == [1, 3]


def test_resolve_risk_config_uses_liquidation_config_when_risk_config_missing() -> None:
    liquidation = RiskLiquidationConfig(enabled=True, daily_loss_limit_pct=0.02)
    strategy_config = StrategyReplayConfig(
        liquidation_config=liquidation,
        allow_reduce_only_when_halted=False,
    )

    resolved = _resolve_risk_config(strategy_config)

    assert resolved is not None
    assert resolved.liquidation_config is not None
    assert resolved.allow_reduce_only_when_halted is False


def test_attach_components_registers_actor_and_strategy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _build_harness_config(tmp_path)
    inst = InstrumentId.from_str("SPY.XNAS")
    bar_type = BarType(inst, BarSpecification.from_str("1-MINUTE-LAST"))

    class _Actor:
        def __init__(self, *, config: Any) -> None:
            self.config = config
            self.subscribed: list[Any] = []

        def subscribe_bars(self, bar: Any) -> None:
            self.subscribed.append(bar)

    class _Strategy:
        def __init__(self, *, config: Any) -> None:
            self.config = config
            self.subscribed: list[Any] = []

        def subscribe_bars(self, bar: Any) -> None:
            self.subscribed.append(bar)

    class _Engine:
        def __init__(self) -> None:
            self.actors: list[Any] = []
            self.strategies: list[Any] = []

        def add_actor(self, actor: Any) -> None:
            self.actors.append(actor)

        def add_strategy(self, strategy: Any) -> None:
            self.strategies.append(strategy)

    monkeypatch.setattr(parquet_live_replay_harness, "MLSignalActor", _Actor)
    monkeypatch.setattr(parquet_live_replay_harness, "MLTradingStrategy", _Strategy)
    monkeypatch.setattr(parquet_live_replay_harness, "_build_actor_config", lambda **_kwargs: {"actor": True})
    monkeypatch.setattr(parquet_live_replay_harness, "_build_strategy_config", lambda **_kwargs: {"strategy": True})
    engine = _Engine()

    parquet_live_replay_harness._attach_components(
        engine=engine,
        config=config,
        instrument_ids=(inst,),
        bar_types={inst: bar_type},
        db_connection=None,
    )

    assert len(engine.actors) == 1
    assert len(engine.strategies) == 1
    assert engine.actors[0].subscribed == [bar_type]
    assert engine.strategies[0].subscribed == [bar_type]


def test_persist_backtest_result_writes_sanitized_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = SimpleNamespace(backtest_id="run")
    output_path = tmp_path / "out"
    output_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "asdict",
        lambda _result: {"stats_returns": {"x": float("nan")}},
    )

    _persist_backtest_result(result=result, output_path=output_path)

    assert (output_path / "backtest_result.json").exists()


def test_run_parquet_live_replay_harness_happy_path_and_empty_bars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strategy = StrategyReplayConfig(
        serialize_order_intents=True,
        execute_trades=False,
        subscribe_quote_ticks=True,
    )
    config = _build_harness_config(tmp_path, strategy=strategy, output_dir=str(tmp_path / "store"))
    inst = InstrumentId.from_str("SPY.XNAS")
    bar_type_proxy = SimpleNamespace(spec="1-MINUTE-LAST")
    backtest_result = SimpleNamespace(
        run_started=1,
        run_finished=2,
        backtest_start=1,
        backtest_end=2,
        total_orders=1,
        total_positions=0,
    )

    class _Engine:
        def __init__(self) -> None:
            self.data_calls: list[tuple[int, bool]] = []
            self.disposed = False

        def add_data(self, data: list[Any], *, sort: bool) -> None:
            self.data_calls.append((len(data), sort))

        def run(self, *, start: Any, end: Any) -> None:
            del start, end
            return None

        def get_result(self) -> Any:
            return backtest_result

        def dispose(self) -> None:
            self.disposed = True

    engine = _Engine()
    monkeypatch.setattr(parquet_live_replay_harness, "_resolve_output_path", lambda _cfg, _run_id: tmp_path / "store")
    monkeypatch.setattr(parquet_live_replay_harness, "_configure_environment", lambda _cfg, _path: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_resolve_replay_db_connection", lambda _cfg: "postgresql://db")
    monkeypatch.setattr(parquet_live_replay_harness, "_normalize_instrument_ids", lambda _ids, fallback_venue: (inst,))
    monkeypatch.setattr(parquet_live_replay_harness, "BarType", lambda _inst, _spec: bar_type_proxy)
    monkeypatch.setattr(parquet_live_replay_harness, "ParquetDataCatalog", lambda _path: SimpleNamespace())
    monkeypatch.setattr(parquet_live_replay_harness, "_load_bars", lambda *_args, **_kwargs: [SimpleNamespace(ts_event=1)])
    monkeypatch.setattr(parquet_live_replay_harness, "_load_quote_ticks", lambda *_args, **_kwargs: [SimpleNamespace(ts_event=1)])
    monkeypatch.setattr(parquet_live_replay_harness, "_infer_price_precisions", lambda *_args, **_kwargs: {inst: 2})
    monkeypatch.setattr(parquet_live_replay_harness, "_resolve_instruments", lambda *_args, **_kwargs: {inst: object()})
    monkeypatch.setattr(parquet_live_replay_harness, "_build_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(parquet_live_replay_harness, "_attach_components", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_persist_backtest_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_persist_replay_summary", lambda **_kwargs: None)

    result = run_parquet_live_replay_harness(config)

    assert result.run_id == "run-1"
    assert result.instrument_ids == ("SPY.XNAS",)
    assert result.bars_loaded == 1
    assert result.quote_ticks_loaded == 1
    assert engine.disposed is True

    monkeypatch.setattr(parquet_live_replay_harness, "_load_bars", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="No bars loaded for replay"):
        run_parquet_live_replay_harness(config)


def test_persist_backtest_result_returns_when_output_path_none() -> None:
    _persist_backtest_result(result=SimpleNamespace(), output_path=None)


def test_resolve_replay_db_connection_returns_none_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _build_harness_config(tmp_path)
    monkeypatch.setattr(
        parquet_live_replay_harness,
        "collect_postgres_candidates",
        lambda _role: SimpleNamespace(urls=[]),
    )

    assert _resolve_replay_db_connection(config) is None


def test_try_catalog_instrument_returns_none_when_catalog_lookup_raises() -> None:
    class _Catalog:
        def instruments(self, *, instrument_ids: list[str]) -> list[Any]:
            del instrument_ids
            raise RuntimeError("catalog unavailable")

    result = _try_catalog_instrument(
        catalog=_Catalog(),
        instrument_id=InstrumentId.from_str("SPY.XNAS"),
    )

    assert result is None


def test_update_price_precision_ignores_prices_without_precision_attribute() -> None:
    precisions: dict[InstrumentId, int] = {}
    _update_price_precision(
        precisions,
        InstrumentId.from_str("SPY.XNAS"),
        price=SimpleNamespace(),
    )

    assert precisions == {}


def test_run_parquet_live_replay_harness_swallow_engine_dispose_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _build_harness_config(tmp_path, strategy=StrategyReplayConfig(), output_dir=str(tmp_path / "store"))
    inst = InstrumentId.from_str("SPY.XNAS")
    bar_type_proxy = SimpleNamespace(spec="1-MINUTE-LAST")
    backtest_result = SimpleNamespace(
        run_started=1,
        run_finished=2,
        backtest_start=1,
        backtest_end=2,
        total_orders=1,
        total_positions=0,
    )

    class _Engine:
        def add_data(self, data: list[Any], *, sort: bool) -> None:
            del data, sort

        def run(self, *, start: Any, end: Any) -> None:
            del start, end

        def get_result(self) -> Any:
            return backtest_result

        def dispose(self) -> None:
            raise RuntimeError("dispose failure")

    monkeypatch.setattr(parquet_live_replay_harness, "_resolve_output_path", lambda _cfg, _run_id: tmp_path / "store")
    monkeypatch.setattr(parquet_live_replay_harness, "_configure_environment", lambda _cfg, _path: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_resolve_replay_db_connection", lambda _cfg: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_normalize_instrument_ids", lambda _ids, fallback_venue: (inst,))
    monkeypatch.setattr(parquet_live_replay_harness, "BarType", lambda _inst, _spec: bar_type_proxy)
    monkeypatch.setattr(parquet_live_replay_harness, "ParquetDataCatalog", lambda _path: SimpleNamespace())
    monkeypatch.setattr(parquet_live_replay_harness, "_load_bars", lambda *_args, **_kwargs: [SimpleNamespace(ts_event=1)])
    monkeypatch.setattr(parquet_live_replay_harness, "_infer_price_precisions", lambda *_args, **_kwargs: {inst: 2})
    monkeypatch.setattr(parquet_live_replay_harness, "_resolve_instruments", lambda *_args, **_kwargs: {inst: object()})
    monkeypatch.setattr(parquet_live_replay_harness, "_build_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr(parquet_live_replay_harness, "_attach_components", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_persist_backtest_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parquet_live_replay_harness, "_persist_replay_summary", lambda **_kwargs: None)

    result = run_parquet_live_replay_harness(config)

    assert result.bars_loaded == 1
