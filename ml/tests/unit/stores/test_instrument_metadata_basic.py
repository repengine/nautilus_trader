"""
Basic unit tests for InstrumentMetadataStore and DummyInstrumentMetadataStore.

Tests basic CRUD operations without importing heavy dependencies.
"""

import time
from typing import Any

import pytest


# Avoid importing from ml.stores which triggers heavy dependency chain
# Import directly from the module
import sys
from pathlib import Path

# Add ml to path
ml_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ml_path))

from ml.stores.instrument_metadata_store import DummyInstrumentMetadataStore  # noqa: E402
from ml.stores.protocols import InstrumentMetadataStoreProtocol  # noqa: E402


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_dummy_store_implements_protocol() -> None:
    """Test that DummyInstrumentMetadataStore implements the protocol."""
    store = DummyInstrumentMetadataStore()
    assert isinstance(store, InstrumentMetadataStoreProtocol)


# =============================================================================
# DummyInstrumentMetadataStore Tests
# =============================================================================


class TestDummyInstrumentMetadataStore:
    """Test suite for DummyInstrumentMetadataStore (in-memory fallback)."""

    @pytest.fixture
    def store(self) -> DummyInstrumentMetadataStore:
        """Fixture providing a fresh dummy store instance."""
        return DummyInstrumentMetadataStore()

    def test_write_and_get_metadata(self, store: DummyInstrumentMetadataStore) -> None:
        """Test basic write and retrieval of metadata."""
        ts_event = time.time_ns()
        ts_init = time.time_ns()

        store.write_metadata(
            instrument_id="US10Y.BOND",
            duration_bucket=2,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=ts_event,
            ts_init=ts_init,
            region="US",
            sector="TREASURY",
            rating="AAA",
        )

        metadata = store.get_metadata("US10Y.BOND")
        assert metadata is not None
        assert metadata["instrument_id"] == "US10Y.BOND"
        assert metadata["duration_bucket"] == 2
        assert metadata["issuer_type"] == 0
        assert metadata["liquidity_tier"] == 1
        assert metadata["region"] == "US"
        assert metadata["sector"] == "TREASURY"
        assert metadata["rating"] == "AAA"

    def test_get_metadata_not_found(self, store: DummyInstrumentMetadataStore) -> None:
        """Test getting metadata for non-existent instrument."""
        metadata = store.get_metadata("NONEXISTENT.BOND")
        assert metadata is None

    def test_temporal_queries(self, store: DummyInstrumentMetadataStore) -> None:
        """Test point-in-time metadata queries."""
        t1 = time.time_ns()
        t2 = t1 + 1_000_000_000  # 1 second later
        t3 = t2 + 1_000_000_000  # 2 seconds later

        # Write metadata at t1 (short duration)
        store.write_metadata(
            instrument_id="US5Y.BOND",
            duration_bucket=0,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=t1,
            ts_init=t1,
        )

        # Write updated metadata at t2 (medium duration)
        store.write_metadata(
            instrument_id="US5Y.BOND",
            duration_bucket=1,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=t2,
            ts_init=t2,
        )

        # Query at t1 should get first version
        metadata_t1 = store.get_metadata("US5Y.BOND", ts_event=t1)
        assert metadata_t1 is not None
        assert metadata_t1["duration_bucket"] == 0

        # Query at t2 should get second version
        metadata_t2 = store.get_metadata("US5Y.BOND", ts_event=t2)
        assert metadata_t2 is not None
        assert metadata_t2["duration_bucket"] == 1

        # Query at t3 should get second version (latest)
        metadata_t3 = store.get_metadata("US5Y.BOND", ts_event=t3)
        assert metadata_t3 is not None
        assert metadata_t3["duration_bucket"] == 1

    def test_get_instruments_by_factors(self, store: DummyInstrumentMetadataStore) -> None:
        """Test filtering instruments by factor criteria."""
        ts = time.time_ns()

        # Add sovereign bonds with different durations
        store.write_metadata(
            instrument_id="US2Y.BOND",
            duration_bucket=0,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=ts,
            ts_init=ts,
        )

        store.write_metadata(
            instrument_id="US10Y.BOND",
            duration_bucket=2,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=ts,
            ts_init=ts,
        )

        # Add corporate bond
        store.write_metadata(
            instrument_id="AAPL.NASDAQ",
            duration_bucket=1,
            issuer_type=2,
            liquidity_tier=1,
            ts_event=ts,
            ts_init=ts,
        )

        # Filter by sovereign issuer
        sovereigns = store.get_instruments_by_factors(issuer_type=0)
        assert set(sovereigns) == {"US2Y.BOND", "US10Y.BOND"}

        # Filter by long duration
        long_duration = store.get_instruments_by_factors(duration_bucket=2)
        assert long_duration == ["US10Y.BOND"]

        # Filter by corporate + high liquidity
        corporate_liquid = store.get_instruments_by_factors(
            issuer_type=2,
            liquidity_tier=1,
        )
        assert corporate_liquid == ["AAPL.NASDAQ"]

    def test_flush_no_op(self, store: DummyInstrumentMetadataStore) -> None:
        """Test that flush is a no-op for in-memory store."""
        store.flush()  # Should not raise

    def test_health_status(self, store: DummyInstrumentMetadataStore) -> None:
        """Test health status reporting."""
        health = store.get_health_status()
        assert health["status"] == "degraded"
        assert health["component"] == "DummyInstrumentMetadataStore"
        assert health["persistence"] == "in-memory-only"
        assert "instruments_cached" in health

    def test_valid_until_filtering(self, store: DummyInstrumentMetadataStore) -> None:
        """Test validity period filtering."""
        t1 = time.time_ns()
        t2 = t1 + 1_000_000_000
        t3 = t2 + 1_000_000_000

        # Write metadata valid from t1 to t2
        store.write_metadata(
            instrument_id="TEMP.BOND",
            duration_bucket=0,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=t1,
            ts_init=t1,
            valid_from_ns=t1,
            valid_until_ns=t2,
        )

        # Write new metadata valid from t2 onwards
        store.write_metadata(
            instrument_id="TEMP.BOND",
            duration_bucket=1,
            issuer_type=0,
            liquidity_tier=1,
            ts_event=t2,
            ts_init=t2,
            valid_from_ns=t2,
            valid_until_ns=None,
        )

        # Query at t1 should get first version
        metadata_t1 = store.get_metadata("TEMP.BOND", ts_event=t1)
        assert metadata_t1 is not None
        assert metadata_t1["duration_bucket"] == 0

        # Query at t2 should get second version
        metadata_t2 = store.get_metadata("TEMP.BOND", ts_event=t2)
        assert metadata_t2 is not None
        assert metadata_t2["duration_bucket"] == 1

        # Query at t3 should get second version
        metadata_t3 = store.get_metadata("TEMP.BOND", ts_event=t3)
        assert metadata_t3 is not None
        assert metadata_t3["duration_bucket"] == 1


# =============================================================================
# Type Safety Tests (compile-time via mypy)
# =============================================================================


def test_protocol_type_safety() -> None:
    """
    Test that protocol typing works correctly.

    This function demonstrates proper protocol usage and will be checked by mypy --strict.
    """

    def process_metadata(store: InstrumentMetadataStoreProtocol) -> dict[str, Any] | None:
        """Function that accepts any store implementing the protocol."""
        return store.get_metadata("US10Y.BOND")

    # Both implementations should work
    dummy_store = DummyInstrumentMetadataStore()
    result1 = process_metadata(dummy_store)
    assert result1 is None or isinstance(result1, dict)


if __name__ == "__main__":
    # Run tests with: pytest ml/tests/unit/stores/test_instrument_metadata_basic.py -v
    pytest.main([__file__, "-v"])
