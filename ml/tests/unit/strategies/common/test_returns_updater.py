from __future__ import annotations

from dataclasses import dataclass, field

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
