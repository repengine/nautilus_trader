from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ml.actors.base import MLSignal
from ml.config.base import ReturnsConfig
from ml.config.base import ReturnsPriceSource
from ml.config.base import ReturnsUpdateMode
from ml.strategies.common import ReturnUpdateResult
from ml.strategies.common import ReturnsUpdater
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@dataclass(slots=True)
class _Px:
    value: float

    def as_double(self) -> float:
        return self.value


@dataclass(slots=True)
class _QuoteTick:
    bid_price: _Px
    ask_price: _Px
    ts_event: int


@dataclass(slots=True)
class _TradeTick:
    price: _Px
    ts_event: int


@dataclass(slots=True)
class _Bar:
    bar_type: BarType
    close: _Px
    ts_event: int
    is_revision: bool = False


@dataclass(slots=True)
class _Cache:
    quote: _QuoteTick | None = None
    trade: _TradeTick | None = None

    def quote_tick(self, _instrument_id: InstrumentId) -> _QuoteTick | None:
        return self.quote

    def trade_tick(self, _instrument_id: InstrumentId) -> _TradeTick | None:
        return self.trade


@dataclass(slots=True)
class _Sizer:
    updates: list[float] = field(default_factory=list)
    annualization_factor: float | None = None

    def update_market_data(self, return_pct: float) -> None:
        self.updates.append(return_pct)

    def set_annualization_factor(self, factor: float) -> None:
        self.annualization_factor = factor


@dataclass(slots=True)
class _Portfolio:
    updates: list[tuple[InstrumentId, float]] = field(default_factory=list)
    annualization_factor: float | None = None

    def update_returns(self, instrument: InstrumentId, return_pct: float) -> None:
        self.updates.append((instrument, return_pct))

    def set_annualization_factor(self, factor: float) -> None:
        self.annualization_factor = factor


def _signal(instrument_id: InstrumentId, ts_event: int) -> MLSignal:
    return MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.5,
        confidence=0.9,
        ts_event=ts_event,
        ts_init=ts_event,
    )


def _bar(
    instrument_id: InstrumentId,
    close: float,
    ts_event: int,
    *,
    bar_spec: str = "1-MINUTE-LAST",
    is_revision: bool = False,
) -> _Bar:
    bar_type = BarType(
        instrument_id,
        BarSpecification.from_str(bar_spec),
    )
    return _Bar(
        bar_type=bar_type,
        close=_Px(close),
        ts_event=ts_event,
        is_revision=is_revision,
    )


def test_returns_updater_uses_quote_mid_and_updates_sinks() -> None:
    instrument_id = InstrumentId.from_str("AAA.SIM")
    cache = _Cache()
    sizer = _Sizer()
    portfolio = _Portfolio()
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.QUOTE_MID, ReturnsPriceSource.LAST_TRADE],
        max_price_age_ms=1_000,
    )
    updater = ReturnsUpdater(
        config=cfg,
        position_sizer=sizer,
        portfolio_manager=portfolio,
    )

    cache.quote = _QuoteTick(bid_price=_Px(100.0), ask_price=_Px(100.2), ts_event=1_000)
    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=1_000),
        cache=cache,
        reference_ts=1_000,
    )

    assert isinstance(result, ReturnUpdateResult)
    assert result.source is ReturnsPriceSource.QUOTE_MID
    assert result.return_pct is None
    assert not sizer.updates
    assert not portfolio.updates

    cache.quote = _QuoteTick(bid_price=_Px(101.0), ask_price=_Px(101.2), ts_event=2_000)
    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=2_000),
        cache=cache,
        reference_ts=2_000,
    )

    expected_return = (101.1 / 100.1) - 1.0
    assert isinstance(result, ReturnUpdateResult)
    assert result.source is ReturnsPriceSource.QUOTE_MID
    assert result.return_pct == pytest.approx(expected_return)
    assert sizer.updates == [pytest.approx(expected_return)]
    assert portfolio.updates == [(instrument_id, pytest.approx(expected_return))]


def test_returns_updater_falls_back_to_trade_when_quote_stale() -> None:
    instrument_id = InstrumentId.from_str("BBB.SIM")
    cache = _Cache()
    sizer = _Sizer()
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.QUOTE_MID, ReturnsPriceSource.LAST_TRADE],
        max_price_age_ms=1,
    )
    updater = ReturnsUpdater(config=cfg, position_sizer=sizer)

    cache.quote = _QuoteTick(bid_price=_Px(100.0), ask_price=_Px(100.2), ts_event=1)
    cache.trade = _TradeTick(price=_Px(99.0), ts_event=10_000_000)
    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=10_000_000),
        cache=cache,
        reference_ts=10_000_000,
    )

    assert isinstance(result, ReturnUpdateResult)
    assert result.source is ReturnsPriceSource.LAST_TRADE

    cache.trade = _TradeTick(price=_Px(100.0), ts_event=20_000_000)
    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=20_000_000),
        cache=cache,
        reference_ts=20_000_000,
    )

    assert isinstance(result, ReturnUpdateResult)
    assert result.source is ReturnsPriceSource.LAST_TRADE
    assert sizer.updates == [pytest.approx((100.0 / 99.0) - 1.0)]


def test_returns_updater_respects_cadence() -> None:
    instrument_id = InstrumentId.from_str("CCC.SIM")
    cache = _Cache()
    sizer = _Sizer()
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.QUOTE_MID],
        update_cadence_ms=60_000,
    )
    updater = ReturnsUpdater(config=cfg, position_sizer=sizer)

    cache.quote = _QuoteTick(bid_price=_Px(100.0), ask_price=_Px(100.2), ts_event=1_000)
    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=1_000),
        cache=cache,
        reference_ts=1_000,
    )

    assert isinstance(result, ReturnUpdateResult)
    assert result.updated is True

    cache.quote = _QuoteTick(bid_price=_Px(101.0), ask_price=_Px(101.2), ts_event=10_000)
    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=10_000),
        cache=cache,
        reference_ts=10_000,
    )

    assert isinstance(result, ReturnUpdateResult)
    assert result.updated is False
    assert result.reason == "cadence_skip"
    assert not sizer.updates


def test_returns_updater_sets_annualization_from_bar_spec() -> None:
    sizer = _Sizer()
    portfolio = _Portfolio()
    cfg = ReturnsConfig(
        bar_spec="5-MINUTE-LAST",
    )

    ReturnsUpdater(config=cfg, position_sizer=sizer, portfolio_manager=portfolio)

    expected = (365.25 * 24 * 60 * 60) / (5 * 60)
    assert sizer.annualization_factor == pytest.approx(expected)
    assert portfolio.annualization_factor == pytest.approx(expected)


def test_returns_updater_updates_from_bar_close() -> None:
    instrument_id = InstrumentId.from_str("DDD.SIM")
    sizer = _Sizer()
    portfolio = _Portfolio()
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.BAR_CLOSE],
    )
    updater = ReturnsUpdater(
        config=cfg,
        position_sizer=sizer,
        portfolio_manager=portfolio,
    )

    result = updater.update_from_bar(
        _bar(instrument_id, close=100.0, ts_event=1_000_000_000),
        cache=None,
    )

    assert result.source is ReturnsPriceSource.BAR_CLOSE
    assert result.return_pct is None
    assert not sizer.updates
    assert not portfolio.updates

    result = updater.update_from_bar(
        _bar(instrument_id, close=101.0, ts_event=61_000_000_000),
        cache=None,
    )

    expected_return = (101.0 / 100.0) - 1.0
    assert result.source is ReturnsPriceSource.BAR_CLOSE
    assert result.return_pct == pytest.approx(expected_return)
    assert sizer.updates == [pytest.approx(expected_return)]
    assert portfolio.updates == [(instrument_id, pytest.approx(expected_return))]


def test_returns_updater_skips_revision_bars() -> None:
    instrument_id = InstrumentId.from_str("EEE.SIM")
    sizer = _Sizer()
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.BAR_CLOSE],
    )
    updater = ReturnsUpdater(config=cfg, position_sizer=sizer)

    result = updater.update_from_bar(
        _bar(instrument_id, close=100.0, ts_event=1_000, is_revision=True),
        cache=None,
    )

    assert result.updated is False
    assert result.reason == "bar_revision"
    assert not sizer.updates


def test_returns_updater_respects_update_mode() -> None:
    cfg = ReturnsConfig(update_mode=ReturnsUpdateMode.SIGNAL)
    updater = ReturnsUpdater(config=cfg)

    assert updater.should_update_from_signal() is True
    assert updater.should_update_from_bar() is False


def test_returns_updater_uses_signal_bar_close_and_bar_spec_metadata() -> None:
    instrument_id = InstrumentId.from_str("FFF.SIM")
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.BAR_CLOSE],
    )
    updater = ReturnsUpdater(config=cfg)

    first = MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.5,
        confidence=0.9,
        ts_event=1_000_000_000,
        ts_init=1_000_000_000,
        metadata={"bar_close": 100.0, "bar_spec": "1-MINUTE-LAST"},
    )
    second = MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.5,
        confidence=0.9,
        ts_event=61_000_000_000,
        ts_init=61_000_000_000,
        metadata={"bar_close": 110.0},
    )

    first_result = updater.update_from_signal(first, cache=None, reference_ts=first.ts_event)
    second_result = updater.update_from_signal(second, cache=None, reference_ts=second.ts_event)

    assert first_result.updated is True
    assert first_result.return_pct is None
    assert second_result.updated is True
    assert second_result.return_pct == pytest.approx(0.1)


def test_returns_updater_returns_price_unavailable_when_cache_missing() -> None:
    instrument_id = InstrumentId.from_str("GGG.SIM")
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.QUOTE_MID, ReturnsPriceSource.LAST_TRADE],
    )
    updater = ReturnsUpdater(config=cfg)

    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=1_000),
        cache=None,
        reference_ts=1_000,
    )

    assert result.updated is False
    assert result.reason == "price_unavailable"


def test_returns_updater_falls_back_from_invalid_quote_to_trade() -> None:
    instrument_id = InstrumentId.from_str("HHH.SIM")
    cache = _Cache(
        quote=_QuoteTick(
            bid_price=_Px(101.0),
            ask_price=_Px(100.0),
            ts_event=1_000,
        ),
        trade=_TradeTick(price=_Px(100.0), ts_event=1_000),
    )
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.QUOTE_MID, ReturnsPriceSource.LAST_TRADE],
    )
    updater = ReturnsUpdater(config=cfg)

    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=1_000),
        cache=cache,
        reference_ts=1_000,
    )

    assert result.updated is True
    assert result.source is ReturnsPriceSource.LAST_TRADE
    assert result.reason is None


def test_returns_updater_returns_price_unavailable_when_trade_stale() -> None:
    instrument_id = InstrumentId.from_str("III.SIM")
    cache = _Cache(trade=_TradeTick(price=_Px(100.0), ts_event=1))
    cfg = ReturnsConfig(
        source_priority=[ReturnsPriceSource.LAST_TRADE],
        max_price_age_ms=1,
    )
    updater = ReturnsUpdater(config=cfg)

    result = updater.update_from_signal(
        _signal(instrument_id, ts_event=10_000_000),
        cache=cache,
        reference_ts=10_000_000,
    )

    assert result.updated is False
    assert result.reason == "price_unavailable"


def test_returns_updater_handles_metric_and_annualization_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml.strategies.common import returns_updater as module

    @dataclass(slots=True)
    class _Logger:
        debug_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

        def debug(self, *args: Any, **kwargs: Any) -> None:
            self.debug_calls.append((args, kwargs))

    class _BrokenMetric:
        def labels(self, **kwargs: Any) -> Any:
            del kwargs
            raise RuntimeError("metric error")

    class _BrokenSizer:
        def update_market_data(self, return_pct: float) -> None:
            del return_pct

        def set_annualization_factor(self, factor: float) -> None:
            del factor
            raise RuntimeError("sizer annualization failed")

    class _BrokenPortfolio:
        def update_returns(self, instrument: Any, return_pct: float) -> None:
            del instrument, return_pct

        def set_annualization_factor(self, factor: float) -> None:
            del factor
            raise RuntimeError("portfolio annualization failed")

    logger = _Logger()
    updater = ReturnsUpdater(
        config=ReturnsConfig(annualization_factor=12.0),
        position_sizer=_BrokenSizer(),
        portfolio_manager=_BrokenPortfolio(),
        log=logger,
        strategy_id="strategy",
    )

    broken_metric = _BrokenMetric()
    monkeypatch.setattr(module, "returns_update_total", broken_metric)
    monkeypatch.setattr(module, "returns_update_skipped_total", broken_metric)
    monkeypatch.setattr(module, "returns_update_fallback_total", broken_metric)

    updater._emit_update(ReturnsPriceSource.BAR_CLOSE)
    updater._emit_skipped("reason")
    updater._emit_fallback(ReturnsPriceSource.QUOTE_MID, "fallback")

    assert logger.debug_calls


def test_returns_updater_private_helpers_cover_reference_and_price_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrument_id = InstrumentId.from_str("JJJ.SIM")
    updater = ReturnsUpdater(config=ReturnsConfig())

    signal = MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.5,
        confidence=0.9,
        ts_event=123,
        ts_init=123,
    )
    assert updater._resolve_reference_ts(signal, reference_ts=None) == 123

    monkeypatch.setattr("ml.strategies.common.returns_updater.time.time_ns", lambda: 456)
    zero_ts_signal = MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.5,
        confidence=0.9,
        ts_event=0,
        ts_init=0,
    )
    assert updater._resolve_reference_ts(zero_ts_signal, reference_ts=None) == 456
    assert updater._resolve_reference_ts(signal, reference_ts=789) == 789

    class _FloatOnly:
        def __float__(self) -> float:
            return 7.5

    class _Broken:
        def as_double(self) -> float:
            raise RuntimeError("bad as_double")

        def __float__(self) -> float:
            raise TypeError("bad float")

    assert updater._price_to_float(_FloatOnly()) == pytest.approx(7.5)
    assert updater._price_to_float(_Broken()) is None

    cadence_ns, annualization = updater._resolve_bar_spec("invalid")
    assert cadence_ns is None
    assert annualization is None


def test_returns_updater_bar_path_returns_price_unavailable_when_bar_close_invalid() -> None:
    instrument_id = InstrumentId.from_str("KKK.SIM")
    cfg = ReturnsConfig(source_priority=[ReturnsPriceSource.BAR_CLOSE])
    updater = ReturnsUpdater(config=cfg)

    bar = _bar(instrument_id, close=-1.0, ts_event=1_000_000_000)
    result = updater.update_from_bar(bar, cache=None, reference_ts=bar.ts_event)

    assert result.updated is False
    assert result.reason == "price_unavailable"


def test_returns_updater_resolve_annualization_and_max_age_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml.strategies.common import returns_updater as module

    updater = ReturnsUpdater(config=ReturnsConfig())

    class _BrokenSpec:
        @staticmethod
        def from_str(value: str) -> object:
            del value
            raise RuntimeError("parse failed")

    monkeypatch.setattr(module, "BarSpecification", _BrokenSpec)
    assert updater._resolve_annualization_factor(ReturnsConfig(bar_spec="bad")) is None

    class _ZeroInterval:
        @staticmethod
        def from_str(value: str) -> object:
            del value

            class _Spec:
                @staticmethod
                def get_interval_ns() -> int:
                    return 0

            return _Spec()

    monkeypatch.setattr(module, "BarSpecification", _ZeroInterval)
    assert updater._resolve_annualization_factor(ReturnsConfig(bar_spec="still-bad")) is None

    assert updater._resolve_max_age_ns(ReturnsConfig(max_price_age_ms="oops")) is None  # type: ignore[arg-type]
    assert updater._resolve_max_age_ns(ReturnsConfig(max_price_age_ms=0)) is None


def test_returns_updater_emit_fallback_skips_when_reason_missing() -> None:
    updater = ReturnsUpdater(config=ReturnsConfig())
    updater._emit_fallback(ReturnsPriceSource.QUOTE_MID, None)
    assert updater._price_to_float(None) is None
