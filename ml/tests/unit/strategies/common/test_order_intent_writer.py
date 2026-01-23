"""
Tests for order intent serialization metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ml.strategies.common.order_submission import OrderIntentWriter
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


def test_order_intent_writer_includes_positions_metadata(tmp_path: Path) -> None:
    path = tmp_path / "order_intents.jsonl"
    writer = OrderIntentWriter(path)
    order = SimpleNamespace(
        strategy_id="STRAT-1",
        trader_id="TRADER-1",
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        client_order_id="OID-1",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        time_in_force=TimeInForce.GTC,
        is_reduce_only=False,
        ts_init=12345,
    )
    positions_metadata = {
        "source": "cache_positions_open",
        "ready": True,
        "degraded": False,
        "reason": None,
        "count": 1,
    }

    writer.write(order, is_live=True, positions_metadata=positions_metadata)

    payload = path.read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(payload)

    assert record["positions"] == positions_metadata


def test_order_intent_writer_when_quote_metadata_provided_returns_quote_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "order_intents.jsonl"
    writer = OrderIntentWriter(path)
    order = SimpleNamespace(
        strategy_id="STRAT-1",
        trader_id="TRADER-1",
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        client_order_id="OID-1",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        time_in_force=TimeInForce.GTC,
        is_reduce_only=False,
        ts_init=12345,
    )
    quote_metadata = {
        "available": True,
        "ts_event": 120,
        "age_ns": 10_000,
        "max_age_ns": 50_000,
        "stale": False,
    }

    writer.write(order, is_live=True, quote_metadata=quote_metadata)

    payload = path.read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(payload)

    assert record["quote"] == quote_metadata


def test_order_intent_writer_when_exit_metadata_provided_returns_exit_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "order_intents.jsonl"
    writer = OrderIntentWriter(path)
    order = SimpleNamespace(
        strategy_id="STRAT-1",
        trader_id="TRADER-1",
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        client_order_id="OID-1",
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("1.0"),
        time_in_force=TimeInForce.GTC,
        is_reduce_only=True,
        ts_init=12345,
    )
    exit_metadata = {
        "reason": "stop_loss",
        "trigger_price": 97.5,
        "time_in_trade_ns": 10_000_000,
    }

    writer.write(order, is_live=True, exit_metadata=exit_metadata)

    payload = path.read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(payload)

    assert record["exit"] == exit_metadata


def test_order_intent_writer_when_execution_metadata_provided_returns_execution_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "order_intents.jsonl"
    writer = OrderIntentWriter(path)
    order = SimpleNamespace(
        strategy_id="STRAT-1",
        trader_id="TRADER-1",
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        client_order_id="OID-1",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        time_in_force=TimeInForce.GTC,
        is_reduce_only=False,
        ts_init=12345,
    )
    execution_metadata = {
        "mode": "market",
        "fallback_reason": "stale_quote",
    }

    writer.write(order, is_live=True, execution_metadata=execution_metadata)

    payload = path.read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(payload)

    assert record["execution"] == execution_metadata
