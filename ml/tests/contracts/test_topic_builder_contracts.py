from __future__ import annotations

import os

import pytest

from ml.common.message_topics import build_topic_for_stage, map_stage_to_topic_segments
from ml.config.events import Stage


@pytest.mark.contracts
def test_build_topic_for_stage_domain_op_default_prefix() -> None:
    topic = build_topic_for_stage(
        Stage.PREDICTION_EMITTED,
        instrument_id="EUR/USD",
        scheme="domain_op",
        prefix="events.ml",
    )
    # domain_op scheme -> 'ml.{domain}.{operation}.{instrument}'
    # For PREDICTION_EMITTED: domain='models', operation='created'
    assert topic == "ml.models.created.EUR.USD"


@pytest.mark.contracts
def test_build_topic_for_stage_stage_first_custom_prefix() -> None:
    topic = build_topic_for_stage(
        Stage.SIGNAL_EMITTED,
        instrument_id="SPY.NYSE",
        scheme="stage_first",
        prefix="custom.prefix",
    )
    # stage_first -> '{prefix}.{STAGE}[.{instrument}]' (STAGE is enum value)
    assert topic == "custom.prefix.SIGNAL_EMITTED.SPY.NYSE"


@pytest.mark.contracts
def test_build_topic_for_stage_honors_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate environment-driven defaults (MessageBusConfig.from_env)
    monkeypatch.setenv("ML_TOPIC_SCHEME", "stage_first")
    monkeypatch.setenv("ML_TOPIC_PREFIX", "events.ml")

    scheme = os.getenv("ML_TOPIC_SCHEME", "domain_op")
    prefix = os.getenv("ML_TOPIC_PREFIX", "events.ml")

    topic = build_topic_for_stage(
        Stage.CATALOG_WRITTEN,
        instrument_id="QQQ.NASDAQ",
        scheme=scheme,
        prefix=prefix,
    )
    assert topic == "events.ml.CATALOG_WRITTEN.QQQ.NASDAQ"


@pytest.mark.contracts
def test_stage_to_domain_op_mapping_contract() -> None:
    """
    Contract test verifying the canonical mapping from Stage enum to (domain, operation)
    pairs.

    This ensures that routing logic is consistent across the ML pipeline and that stage-
    first schemes can be correctly mapped to domain_op schemes.

    """
    # Define the expected canonical mappings
    expected_mappings = {
        Stage.DATA_INGESTED: ("data", "created"),
        Stage.CATALOG_WRITTEN: ("data", "updated"),
        Stage.FEATURE_COMPUTED: ("features", "updated"),
        Stage.PREDICTION_EMITTED: ("models", "created"),
        Stage.SIGNAL_EMITTED: ("strategies", "created"),
    }

    # Verify each mapping matches the implementation
    for stage, (expected_domain, expected_op) in expected_mappings.items():
        actual_domain, actual_op = map_stage_to_topic_segments(stage)

        assert actual_domain == expected_domain, (
            f"Domain mismatch for {stage.value}: "
            f"expected '{expected_domain}', got '{actual_domain}'"
        )
        assert actual_op == expected_op, (
            f"Operation mismatch for {stage.value}: " f"expected '{expected_op}', got '{actual_op}'"
        )


@pytest.mark.contracts
def test_stage_first_vs_domain_op_equivalence() -> None:
    """
    Contract test verifying that stage-first and domain_op schemes produce equivalent
    routing for the same logical events, just with different topic structures.
    """
    test_instrument = "EUR/USD.SIM"

    for stage in [
        Stage.DATA_INGESTED,
        Stage.CATALOG_WRITTEN,
        Stage.FEATURE_COMPUTED,
        Stage.PREDICTION_EMITTED,
        Stage.SIGNAL_EMITTED,
    ]:

        # Generate topics using both schemes
        domain_op_topic = build_topic_for_stage(
            stage=stage,
            instrument_id=test_instrument,
            scheme="domain_op",
        )

        stage_first_topic = build_topic_for_stage(
            stage=stage,
            instrument_id=test_instrument,
            scheme="stage_first",
            prefix="events.ml",
        )

        # Verify both contain normalized instrument
        normalized_instrument = "EUR.USD.SIM"  # / becomes .
        assert normalized_instrument in domain_op_topic
        assert normalized_instrument in stage_first_topic

        # Verify domain_op follows ml.{domain}.{operation}.{instrument} pattern
        domain_op_parts = domain_op_topic.split(".")
        assert len(domain_op_parts) >= 4
        assert domain_op_parts[0] == "ml"

        # Verify stage_first follows {prefix}.{STAGE}.{instrument} pattern
        stage_first_parts = stage_first_topic.split(".")
        assert len(stage_first_parts) >= 4
        assert stage_first_parts[0] == "events"
        assert stage_first_parts[1] == "ml"
        assert stage_first_parts[2] == stage.value

        # Both should end with normalized instrument
        assert domain_op_topic.endswith(normalized_instrument)
        assert stage_first_topic.endswith(normalized_instrument)


@pytest.mark.contracts
def test_wildcard_instrument_normalization_contract() -> None:
    """
    Contract test ensuring wildcard and special characters are consistently normalized
    across all topic building functions to prevent routing conflicts.
    """
    # Test instruments with various special characters that need normalization
    problematic_instruments = [
        "EUR/USD*",  # Wildcard
        "BTC#USD",  # MQTT topic wildcard
        "GOLD+SILVER",  # Plus
        "OIL$FUTURES",  # Dollar sign
        "GBP//JPY",  # Double slash
        "SPY.NYSE*#",  # Multiple special chars
    ]

    for instrument in problematic_instruments:
        # Test both schemes normalize consistently
        domain_op_topic = build_topic_for_stage(
            Stage.FEATURE_COMPUTED,
            instrument,
            scheme="domain_op",
        )

        stage_first_topic = build_topic_for_stage(
            Stage.FEATURE_COMPUTED,
            instrument,
            scheme="stage_first",
            prefix="events.ml",
        )

        # Verify no reserved characters remain in either topic
        for reserved_char in ["*", "#", "+", "$", "/"]:
            assert (
                reserved_char not in domain_op_topic
            ), f"Reserved char '{reserved_char}' found in domain_op topic: {domain_op_topic}"
            assert (
                reserved_char not in stage_first_topic
            ), f"Reserved char '{reserved_char}' found in stage_first topic: {stage_first_topic}"

        # Both should contain only allowed characters: A-Za-z0-9_.-
        import re

        allowed_pattern = re.compile(r"^[A-Za-z0-9_.-]+$")

        # Extract instrument segment from each topic
        domain_op_instrument = domain_op_topic.split(".")[-1] if "." in domain_op_topic else ""
        stage_first_instrument = (
            stage_first_topic.split(".")[-1] if "." in stage_first_topic else ""
        )

        if domain_op_instrument:
            assert allowed_pattern.match(
                domain_op_instrument,
            ), f"Invalid chars in domain_op instrument segment: '{domain_op_instrument}'"

        if stage_first_instrument:
            assert allowed_pattern.match(
                stage_first_instrument,
            ), f"Invalid chars in stage_first instrument segment: '{stage_first_instrument}'"
