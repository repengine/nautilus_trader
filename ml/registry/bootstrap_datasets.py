#!/usr/bin/env python3
"""
Bootstrap script to pre-register standard dataset manifests.

This script creates manifests for all standard dataset types used in the ML pipeline,
ensuring consistent naming and avoiding orphaned events.

Usage:
    python -m ml.registry.bootstrap_datasets [--backend postgres|json] [--registry-path PATH]

"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

from ml.config.dataset_ids import EQUS_MINI_DATASET_ID
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.data_store import EARNINGS_ACTUALS_DATASET_ID
from ml.stores.data_store import EARNINGS_ESTIMATES_DATASET_ID


def create_standard_manifests() -> list[DatasetManifest]:
    """
    Create standard dataset manifests for all pipeline stages.
    """
    manifests = []

    # BARS/OHLCV dataset
    bars_manifest = DatasetManifest(
        dataset_id="bars",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.PARQUET,
        location="catalog/bars/",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=365,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ts_event", "instrument_id"],
        schema_hash="",
        constraints={
            "required_fields": ["ts_event", "ts_init", "instrument_id", "close"],
            "nullable_fields": ["volume"],
            "ranges": {
                "open": {"min": 0.0, "max": 1e9},
                "high": {"min": 0.0, "max": 1e9},
                "low": {"min": 0.0, "max": 1e9},
                "close": {"min": 0.0, "max": 1e9},
                "volume": {"min": 0, "max": 1e15},
            },
        },
        lineage=[],
        pipeline_signature="databento_scheduler_v1",
        version="1.0.0",
    )
    manifests.append(bars_manifest)

    # QUOTES dataset
    quotes_manifest = DatasetManifest(
        dataset_id="quotes",
        dataset_type=DatasetType.QUOTES,
        storage_kind=StorageKind.PARQUET,
        location="catalog/quotes/",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=365,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "bid": "float64",
            "ask": "float64",
            "bid_size": "float64",
            "ask_size": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ts_event", "instrument_id"],
        schema_hash="",
        constraints={
            "required_fields": ["ts_event", "ts_init", "instrument_id", "bid", "ask"],
            "nullable_fields": ["bid_size", "ask_size"],
            "ranges": {
                "bid": {"min": 0.0, "max": 1e12},
                "ask": {"min": 0.0, "max": 1e12},
                "bid_size": {"min": 0.0, "max": 1e12},
                "ask_size": {"min": 0.0, "max": 1e12},
            },
        },
        lineage=[],
        pipeline_signature="databento_scheduler_v1",
        version="1.0.0",
    )
    manifests.append(quotes_manifest)

    # TRADES dataset
    trades_manifest = DatasetManifest(
        dataset_id="trades",
        dataset_type=DatasetType.TRADES,
        storage_kind=StorageKind.PARQUET,
        location="catalog/trades/",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=365,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "price": "float64",
            "size": "float64",
            "aggressor_side": "str",
            "trade_id": "str",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ts_event", "instrument_id", "trade_id"],
        schema_hash="",
        constraints={
            "required_fields": ["ts_event", "ts_init", "instrument_id", "price", "size"],
            "nullable_fields": ["trade_id", "aggressor_side"],
            "ranges": {
                "price": {"min": 0.0, "max": 1e12},
                "size": {"min": 0.0, "max": 1e12},
            },
        },
        lineage=[],
        pipeline_signature="databento_scheduler_v1",
        version="1.0.0",
    )
    manifests.append(trades_manifest)

    # FEATURES dataset
    features_manifest = DatasetManifest(
        dataset_id="features",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.POSTGRES,
        location="ml_feature_values",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=180,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "feature_set_id": "str",
            "feature_values": "json",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ts_event", "instrument_id", "feature_set_id"],
        schema_hash="",
        constraints={
            "required_fields": ["ts_event", "ts_init", "instrument_id", "feature_set_id"],
            "nullable_fields": [],
        },
        lineage=["bars"],
        pipeline_signature="feature_store_v1",
        version="1.0.0",
    )
    manifests.append(features_manifest)

    # PREDICTIONS dataset
    predictions_manifest = DatasetManifest(
        dataset_id="predictions",
        dataset_type=DatasetType.PREDICTIONS,
        storage_kind=StorageKind.POSTGRES,
        location="ml_model_predictions",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=90,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "model_id": "str",
            "prediction": "float64",
            "confidence": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ts_event", "instrument_id", "model_id"],
        schema_hash="",
        constraints={
            "required_fields": ["ts_event", "ts_init", "instrument_id", "model_id", "prediction"],
            "nullable_fields": ["confidence"],
            "ranges": {
                "prediction": {"min": -1.0, "max": 1.0},
                "confidence": {"min": 0.0, "max": 1.0},
            },
        },
        lineage=["features"],
        pipeline_signature="model_store_v1",
        version="1.0.0",
    )
    manifests.append(predictions_manifest)

    # SIGNALS dataset
    signals_manifest = DatasetManifest(
        dataset_id="signals",
        dataset_type=DatasetType.SIGNALS,
        storage_kind=StorageKind.POSTGRES,
        location="ml_strategy_signals",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=90,
        schema={
            "ts_event": "int64",
            "ts_init": "int64",
            "instrument_id": "str",
            "strategy_id": "str",
            "signal": "int64",
            "strength": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ts_event", "instrument_id", "strategy_id"],
        schema_hash="",
        constraints={
            "required_fields": ["ts_event", "ts_init", "instrument_id", "strategy_id", "signal"],
            "nullable_fields": ["strength"],
            "ranges": {
                "signal": {"min": -1, "max": 1},
                "strength": {"min": 0.0, "max": 1.0},
            },
        },
        lineage=["predictions"],
        pipeline_signature="strategy_store_v1",
        version="1.0.0",
    )
    manifests.append(signals_manifest)

    # Canonical EQUS.MINI aggregation with lineage to ITCH fallback
    eq_us_mini_manifest = DatasetManifest(
        dataset_id=EQUS_MINI_DATASET_ID,
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.POSTGRES,
        location="market_data",
        partitioning={"by": "ts_event", "interval": "monthly"},
        retention_days=365,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={
            "required_fields": ["instrument_id", "ts_event", "close"],
            "nullable_fields": ["volume"],
            "ranges": {
                "open": {"min": 0.0},
                "high": {"min": 0.0},
                "low": {"min": 0.0},
                "close": {"min": 0.0},
                "volume": {"min": 0.0},
            },
        },
        lineage=["XNAS.ITCH"],
        pipeline_signature="databento_canonical_v1",
        version="1.0.0",
    )
    manifests.append(eq_us_mini_manifest)

    # EARNINGS ACTUALS dataset
    earnings_actuals_manifest = DatasetManifest(
        dataset_id=EARNINGS_ACTUALS_DATASET_ID,
        dataset_type=DatasetType.EARNINGS_ACTUALS,
        storage_kind=StorageKind.POSTGRES,
        location="ml.earnings_actuals",
        partitioning={"by": ["ticker"]},
        retention_days=3650,
        schema={
            "ticker": "str",
            "period_end": "date",
            "filing_date": "date",
            "ts_event": "int64",
            "ts_init": "int64",
            "eps_basic": "float64",
            "eps_diluted": "float64",
            "revenue": "float64",
            "net_income": "float64",
            "operating_income": "float64",
            "shares_outstanding": "int64",
            "filing_type": "str",
            "fiscal_year": "int64",
            "fiscal_quarter": "int64",
            "data_source": "str",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ticker", "period_end"],
        schema_hash="",
        constraints={
            "required_fields": [
                "ticker",
                "period_end",
                "filing_date",
                "ts_event",
                "ts_init",
            ],
            "nullable_fields": [
                "eps_basic",
                "eps_diluted",
                "revenue",
                "net_income",
                "operating_income",
                "shares_outstanding",
                "filing_type",
                "fiscal_year",
                "fiscal_quarter",
                "data_source",
            ],
        },
        lineage=["edgar_fetcher"],
        pipeline_signature="earnings_ingestion_v1",
        version="1.0.0",
    )
    manifests.append(earnings_actuals_manifest)

    # EARNINGS ESTIMATES dataset
    earnings_estimates_manifest = DatasetManifest(
        dataset_id=EARNINGS_ESTIMATES_DATASET_ID,
        dataset_type=DatasetType.EARNINGS_ESTIMATES,
        storage_kind=StorageKind.POSTGRES,
        location="ml.earnings_estimates",
        partitioning={"by": ["ticker"]},
        retention_days=1825,
        schema={
            "ticker": "str",
            "period_end": "date",
            "estimate_date": "date",
            "ts_event": "int64",
            "ts_init": "int64",
            "eps_consensus": "float64",
            "revenue_consensus": "float64",
            "num_analysts": "int64",
            "data_source": "str",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["ticker", "period_end", "estimate_date"],
        schema_hash="",
        constraints={
            "required_fields": [
                "ticker",
                "period_end",
                "estimate_date",
                "ts_event",
                "ts_init",
            ],
            "nullable_fields": [
                "eps_consensus",
                "revenue_consensus",
                "num_analysts",
                "data_source",
            ],
        },
        lineage=["yahoo_consensus"],
        pipeline_signature="earnings_ingestion_v1",
        version="1.0.0",
    )
    manifests.append(earnings_estimates_manifest)

    return manifests


def create_standard_contracts() -> dict[str, DataContract]:
    """
    Create standard data contracts for each dataset type.
    """
    contracts = {}

    # Helper to create validation rules
    def make_rule(rule_type: ValidationRuleType, field: str = "*", **params: Any) -> ValidationRule:
        return ValidationRule(
            rule_type=rule_type,
            field_name=field,
            parameters=params,
            severity=QualityFlag.FAIL,
            description=f"{rule_type.value} check for {field}",
        )

    # Bars contract - lenient mode for market data
    from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

    bars_contract = DataContract(
        contract_id="bars_contract_v1",
        dataset_id="bars",
        version="1.0.0",
        enforcement_mode="lenient",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.RANGE, "close", min=0.0),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.01,
            "duplicate_rate": 0.001,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:bars.created"),
        last_modified=_sanitize(int(time.time_ns()), context="registry.bootstrap:bars.modified"),
    )
    contracts["bars"] = bars_contract

    eq_us_mini_contract = DataContract(
        contract_id="equs_mini_contract_v1",
        dataset_id=EQUS_MINI_DATASET_ID,
        version="1.0.0",
        enforcement_mode="lenient",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.RANGE, "close", min=0.0),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.01,
            "duplicate_rate": 0.001,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:equs_mini.created"),
        last_modified=_sanitize(
            int(time.time_ns()),
            context="registry.bootstrap:equs_mini.modified",
        ),
    )
    contracts[EQUS_MINI_DATASET_ID] = eq_us_mini_contract

    # Quotes contract - lenient mode for market data
    quotes_contract = DataContract(
        contract_id="quotes_contract_v1",
        dataset_id="quotes",
        version="1.0.0",
        enforcement_mode="lenient",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.RANGE, "bid", min=0.0),
            make_rule(ValidationRuleType.RANGE, "ask", min=0.0),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.05,
            "duplicate_rate": 0.01,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:quotes.created"),
        last_modified=_sanitize(int(time.time_ns()), context="registry.bootstrap:quotes.modified"),
    )
    contracts["quotes"] = quotes_contract

    # Trades contract - lenient mode for market data
    trades_contract = DataContract(
        contract_id="trades_contract_v1",
        dataset_id="trades",
        version="1.0.0",
        enforcement_mode="lenient",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.RANGE, "price", min=0.0),
            make_rule(ValidationRuleType.RANGE, "size", min=0.0),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.05,
            "duplicate_rate": 0.01,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:trades.created"),
        last_modified=_sanitize(int(time.time_ns()), context="registry.bootstrap:trades.modified"),
    )
    contracts["trades"] = trades_contract

    # Features contract - strict mode for ML features
    features_contract = DataContract(
        contract_id="features_contract_v1",
        dataset_id="features",
        version="1.0.0",
        enforcement_mode="strict",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.UNIQUENESS),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.0,
            "duplicate_rate": 0.0,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:features.created"),
        last_modified=_sanitize(
            int(time.time_ns()),
            context="registry.bootstrap:features.modified",
        ),
    )
    contracts["features"] = features_contract

    # Predictions contract - strict mode for model outputs
    predictions_contract = DataContract(
        contract_id="predictions_contract_v1",
        dataset_id="predictions",
        version="1.0.0",
        enforcement_mode="strict",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.RANGE, "prediction", min=-1.0, max=1.0),
            make_rule(ValidationRuleType.UNIQUENESS),
        ],
        quality_thresholds={
            "null_rate": 0.0,
            "duplicate_rate": 0.0,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:predictions.created"),
        last_modified=_sanitize(
            int(time.time_ns()),
            context="registry.bootstrap:predictions.modified",
        ),
    )
    contracts["predictions"] = predictions_contract

    # Signals contract - monitor only for strategy signals
    signals_contract = DataContract(
        contract_id="signals_contract_v1",
        dataset_id="signals",
        version="1.0.0",
        enforcement_mode="monitor_only",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.RANGE, "signal", min=-1, max=1),
        ],
        quality_thresholds={
            "null_rate": 0.05,
            "duplicate_rate": 0.01,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:signals.created"),
        last_modified=_sanitize(int(time.time_ns()), context="registry.bootstrap:signals.modified"),
    )
    contracts["signals"] = signals_contract

    earnings_actuals_contract = DataContract(
        contract_id="earnings_actuals_contract_v1",
        dataset_id=EARNINGS_ACTUALS_DATASET_ID,
        version="1.0.0",
        enforcement_mode="monitor_only",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.05,
            "duplicate_rate": 0.01,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:earnings_actuals.created"),
        last_modified=_sanitize(
            int(time.time_ns()),
            context="registry.bootstrap:earnings_actuals.modified",
        ),
    )
    contracts[EARNINGS_ACTUALS_DATASET_ID] = earnings_actuals_contract

    earnings_estimates_contract = DataContract(
        contract_id="earnings_estimates_contract_v1",
        dataset_id=EARNINGS_ESTIMATES_DATASET_ID,
        version="1.0.0",
        enforcement_mode="monitor_only",
        validation_rules=[
            make_rule(ValidationRuleType.TYPE_CHECK),
            make_rule(ValidationRuleType.NULLABILITY),
            make_rule(ValidationRuleType.MONOTONICITY, "ts_event", direction="increasing"),
        ],
        quality_thresholds={
            "null_rate": 0.1,
            "duplicate_rate": 0.02,
        },
        created_at=_sanitize(int(time.time_ns()), context="registry.bootstrap:earnings_estimates.created"),
        last_modified=_sanitize(
            int(time.time_ns()),
            context="registry.bootstrap:earnings_estimates.modified",
        ),
    )
    contracts[EARNINGS_ESTIMATES_DATASET_ID] = earnings_estimates_contract

    return contracts


def bootstrap_datasets(
    backend: BackendType = BackendType.JSON,
    registry_path: Path | None = None,
) -> None:
    """
    Bootstrap the data registry with standard dataset manifests.
    """
    # Setup persistence configuration
    if backend == BackendType.POSTGRES:
        db_url = os.getenv("NAUTILUS_REGISTRY_DB_URL")
        if not db_url:
            raise ValueError(
                "NAUTILUS_REGISTRY_DB_URL environment variable must be set for PostgreSQL backend",
            )
        persistence_config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=db_url,
        )
    else:
        effective_registry_path: Path = (
            registry_path
            if registry_path is not None
            else Path.home() / ".nautilus" / "ml" / "registry"
        )
        persistence_config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=effective_registry_path,
        )

    # Setup registry
    registry = DataRegistry(
        registry_path=effective_registry_path if backend == BackendType.JSON else Path("."),
        persistence_config=persistence_config,
    )

    # Create and register manifests
    manifests = create_standard_manifests()
    contracts = create_standard_contracts()

    print(f"Bootstrapping {len(manifests)} dataset manifests...")

    for manifest in manifests:
        try:
            # Check if already exists
            existing = registry.get_manifest(manifest.dataset_id)
            if existing:
                print(f"  ✓ {manifest.dataset_id} already exists (v{existing.version})")
                continue
        except Exception:
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Manifest lookup failed (expected if new); proceeding to register",
                exc_info=True,
            )

        # Register new manifest
        dataset_id = registry.register_dataset(manifest)
        print(f"  ✓ Registered {dataset_id} ({manifest.dataset_type.value})")

        # Register contract if available
        if manifest.dataset_id in contracts and backend == BackendType.JSON:
            # Store contract in JSON registry and persist
            contract = contracts[manifest.dataset_id]
            registry._contracts[manifest.dataset_id] = contract
            registry._save_registry(immediate=True)
            print(f"    → Added contract (mode: {contract.enforcement_mode})")

    print(f"\n✅ Bootstrap complete! Registered {len(manifests)} datasets.")
    print(f"   Backend: {backend.value}")
    if backend == BackendType.JSON:
        print(f"   Registry path: {registry_path}")


def main() -> None:
    """
    Main entry point for bootstrap script.
    """
    parser = argparse.ArgumentParser(
        description="Bootstrap standard dataset manifests for ML pipeline",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["json", "postgres"],
        default="json",
        help="Backend type to use (default: json)",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        help="Path to registry directory (for JSON backend)",
    )

    args = parser.parse_args()

    backend = BackendType.JSON if args.backend == "json" else BackendType.POSTGRES

    try:
        bootstrap_datasets(backend=backend, registry_path=args.registry_path)
    except Exception as e:
        print(f"❌ Bootstrap failed: {e}")
        raise


if __name__ == "__main__":
    main()
