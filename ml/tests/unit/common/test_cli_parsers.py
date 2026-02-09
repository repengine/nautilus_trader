from __future__ import annotations

import argparse
import json

import pytest

from ml.common.cli_parsers import parse_market_inputs_json


def test_parse_market_inputs_json_when_value_none_returns_none() -> None:
    assert parse_market_inputs_json(None) is None


def test_parse_market_inputs_json_when_valid_payload_returns_inputs() -> None:
    payload = json.dumps(
        [
            "feeds.primary",
            {
                "dataset_id": "equities.main",
                "symbols": ["aapl", "MSFT"],
                "schema": "ohlcv-1m",
                "storage_kind": "postgres",
                "start": "2024-01-01",
                "end": "2024-02-01",
            },
        ],
    )

    parsed = parse_market_inputs_json(payload)

    assert parsed is not None
    assert len(parsed) == 2
    assert parsed[0].descriptor_id == "feeds.primary"
    assert parsed[1].dataset_id == "equities.main"
    assert parsed[1].symbols == ("AAPL", "MSFT")
    assert parsed[1].schema_override == "ohlcv-1m"
    assert parsed[1].storage_kind_override is not None
    assert parsed[1].storage_kind_override.value == "postgres"
    assert parsed[1].start == "2024-01-01"
    assert parsed[1].end == "2024-02-01"


def test_parse_market_inputs_json_when_json_invalid_raises_argument_error() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="market_inputs_json must be valid JSON"):
        parse_market_inputs_json("{not-json")


def test_parse_market_inputs_json_when_symbols_shape_invalid_raises_argument_error() -> None:
    payload = json.dumps([{"descriptor_id": "feeds.primary", "symbols": 42}])

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="symbols in market_inputs_json must be a list or comma-separated string",
    ):
        parse_market_inputs_json(payload)


def test_parse_market_inputs_json_when_storage_kind_invalid_raises_argument_error() -> None:
    payload = json.dumps([{"descriptor_id": "feeds.primary", "storage_kind": "unsupported"}])

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Invalid storage_kind 'unsupported' in market_inputs_json",
    ):
        parse_market_inputs_json(payload)
