"""
Feature store for ML pipeline integration with Nautilus Trader.

This module provides a unified interface for computing, storing, and retrieving
ML features from the same PostgreSQL instance used by Nautilus Trader.

Key principles:
- Single PostgreSQL container (Nautilus's existing one)
- FeatureEngineer provides all computation logic (training/inference parity)
- Features stored alongside Nautilus market data for unified access
- Efficient batch computation for historical data
- Real-time computation for live trading

"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec


if TYPE_CHECKING:
    from ml._imports import pl
    from nautilus_trader.model.data import Bar


class FeatureStore:
    """
    Unified feature computation and storage for ML pipeline.

    This class ensures training/inference parity by using the same FeatureEngineer for
    both batch (historical) and online (live) computation.

    Features are stored in the same PostgreSQL instance as Nautilus data, enabling
    efficient joins for training and avoiding data duplication.

    """

    def __init__(
        self,
        connection_string: str,
        feature_config: FeatureConfig | None = None,
        pipeline_spec: PipelineSpec | None = None,
    ):
        """
        Initialize the feature store.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection string (same as Nautilus uses).
            Example: "postgresql://postgres:postgres@localhost:5432/nautilus"
        feature_config : FeatureConfig, optional
            Configuration for feature engineering.
        pipeline_spec : PipelineSpec, optional
            Pipeline specification for feature computation.

        """
        self.connection_string = connection_string
        self.feature_config = feature_config or FeatureConfig()
        self.pipeline_spec = pipeline_spec

        # Create engine and setup tables (reflect partitioned table created by migrations)
        self.engine: Engine = create_engine(connection_string)
        self.metadata = MetaData()
        self._setup_tables()

        # Feature engineer for computation (ensures parity)
        self.feature_engineer = FeatureEngineer(self.feature_config)
        # Internal indicator managers (fallback for online computation when actor does not pass one)
        self._indicator_managers: dict[str, IndicatorManager] = {}

        # Pipeline runner for declarative features
        if self.pipeline_spec:
            from ml.registry.base import DataRequirements

            self.pipeline_runner = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_L2,  # Adjust based on available data
            )
            self.pipeline_hash = self.pipeline_runner.compute_signature()
        else:
            self.pipeline_runner = None
            self.pipeline_hash = self._compute_config_hash()

    def _setup_tables(self) -> None:
        """
        Reflect (preferred) or create a compatible ml_feature_values table.

        The canonical schema is created by migrations (partitioned by ts_event):
        - feature_set_id VARCHAR(255)
        - instrument_id VARCHAR(100)
        - ts_event BIGINT
        - ts_init BIGINT
        - values JSONB
        - is_live BOOLEAN
        - source VARCHAR(50)
        - created_at TIMESTAMPTZ
        Primary key (id, ts_event) where id is BIGSERIAL.
        """
        try:
            # Prefer reflecting the migrated table
            self.feature_values_table = Table(
                "ml_feature_values",
                self.metadata,
                autoload_with=self.engine,
            )
        except Exception:
            # Fallback: create a non-partitioned compatible table for tests/dev
            self.feature_values_table = Table(
                "ml_feature_values",
                self.metadata,
                Column("id", BIGINT, primary_key=True, autoincrement=True),
                Column("feature_set_id", String(255), nullable=False),
                Column("instrument_id", String(100), nullable=False),
                Column("ts_event", BIGINT, nullable=False),
                Column("ts_init", BIGINT, nullable=False),
                Column("values", JSON, nullable=False),
                Column("is_live", BOOLEAN, default=False),
                Column("source", String(50)),
                Column("created_at", BIGINT),
                Index(
                    "idx_ml_feature_values_lookup",
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                ),
                Index("idx_ml_feature_values_live", "is_live"),
            )
            self.metadata.create_all(self.engine)

    def _compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.
        """
        config_str = json.dumps(self.feature_config.__dict__, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store features for historical data.

        This method:
        1. Queries bars from Nautilus PostgreSQL tables
        2. Computes features using FeatureEngineer (same logic as live)
        3. Stores features in ml_feature_values table

        Parameters
        ----------
        instrument_id : str
            Instrument to compute features for.
        start : datetime
            Start time for historical computation.
        end : datetime
            End time for historical computation.
        force_recompute : bool, default False
            If True, recompute even if features exist.

        Returns
        -------
        int
            Number of feature rows computed and stored.

        """
        # Check if features already exist
        if not force_recompute and self._features_exist(instrument_id, start, end):
            return 0

        # Load bars from Nautilus tables
        bars_df = self._load_bars_from_nautilus(instrument_id, start, end)
        if bars_df.is_empty():
            return 0

        # Compute features (batch) ensuring parity with online
        features_df, _ = self.feature_engineer.calculate_features_batch(bars_df)

        feature_names = self._get_feature_names()
        feature_set_id = self._get_feature_set_id()

        # Prepare rows with JSONB values mapping
        rows: list[dict[str, Any]] = []
        timestamps = bars_df["ts_event"].to_numpy()
        created_at_ns = int(datetime.utcnow().timestamp() * 1e9)

        # Convert feature rows to dicts
        # features_df is a DataFrame (polars or pandas). Use row-wise access safely.
        if hasattr(features_df, "iter_rows"):
            # Polars
            for (i, row_vals) in enumerate(features_df.iter_rows()):
                ts_event = int(timestamps[i])
                values_map = {name: float(row_vals[idx]) for idx, name in enumerate(feature_names)}
                rows.append(
                    {
                        "feature_set_id": feature_set_id,
                        "instrument_id": instrument_id,
                        "ts_event": ts_event,
                        "ts_init": ts_event,
                        "values": json.dumps(values_map),
                        "is_live": False,
                        "source": "historical",
                        "created_at": created_at_ns,
                    },
                )
        else:
            # Pandas path
            for i in range(len(features_df)):
                ts_event = int(timestamps[i])
                row = features_df.iloc[i]
                values_map = {name: float(row[name]) for name in feature_names}
                rows.append(
                    {
                        "feature_set_id": feature_set_id,
                        "instrument_id": instrument_id,
                        "ts_event": ts_event,
                        "ts_init": ts_event,
                        "values": json.dumps(values_map),
                        "is_live": False,
                        "source": "historical",
                        "created_at": created_at_ns,
                    },
                )

        # Bulk upsert into partitioned table
        with self.engine.begin() as conn:
            stmt = insert(self.feature_values_table)
            # Upsert on (feature_set_id, instrument_id, ts_event)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                ],
                set_={
                    "values": stmt.excluded.values,
                    "ts_init": stmt.excluded.ts_init,
                    "source": stmt.excluded.source,
                    "created_at": stmt.excluded.created_at,
                },
            )
            conn.execute(stmt, rows)

        return len(rows)

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
        indicator_manager: IndicatorManager | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        Uses the SAME FeatureEngineer as historical computation to ensure
        perfect parity between training and inference.

        Parameters
        ----------
        bar : Bar
            Current bar from Nautilus.
        store : bool, default True
            Whether to store computed features for future training.

        Returns
        -------
        npt.NDArray[np.float32]
            Computed feature vector.

        """
        # Prepare indicator manager (prefer provided from actor for shared state)
        instrument_key = str(getattr(bar, "instrument_id", getattr(bar, "bar_type", getattr(bar, "instrument_id", None))))
        instrument_key = (
            str(bar.bar_type.instrument_id)
            if hasattr(bar, "bar_type") and hasattr(bar.bar_type, "instrument_id")
            else str(getattr(bar, "instrument_id", "unknown"))
        )

        if indicator_manager is None:
            indicator_manager = self._indicator_managers.get(instrument_key)
            if indicator_manager is None:
                indicator_manager = IndicatorManager(self.feature_engineer.config)
                self._indicator_managers[instrument_key] = indicator_manager

        # Update indicators from bar and compute online features
        indicator_manager.update_from_bar(bar)
        if not indicator_manager.all_initialized():
            # Not enough history yet – return empty array to signal no prediction
            return np.zeros(0, dtype=np.float32)

        current_bar = {
            "close": float(bar.close),
            "volume": float(bar.volume),
            "high": float(bar.high),
            "low": float(bar.low),
        }

        features = self.feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_manager,
            scaler=None,
        )

        # Optionally store for future training
        if store and features.size > 0:
            feature_names = self._get_feature_names()
            values_map = {name: float(features[idx]) for idx, name in enumerate(feature_names) if idx < features.size}
            row = {
                "feature_set_id": self._get_feature_set_id(),
                "instrument_id": str(bar.bar_type.instrument_id if hasattr(bar, "bar_type") else getattr(bar, "instrument_id", "unknown")),
                "ts_event": int(bar.ts_event),
                "ts_init": int(bar.ts_init),
                "values": json.dumps(values_map),
                "is_live": True,
                "source": "live",
                "created_at": int(datetime.utcnow().timestamp() * 1e9),
            }

            with self.engine.begin() as conn:
                stmt = insert(self.feature_values_table).on_conflict_do_update(
                    index_elements=["feature_set_id", "instrument_id", "ts_event"],
                    set_={
                        "values": stmt.excluded.values,
                        "ts_init": stmt.excluded.ts_init,
                        "is_live": stmt.excluded.is_live,
                        "source": stmt.excluded.source,
                        "created_at": stmt.excluded.created_at,
                    },
                )
                conn.execute(stmt, row)

        return features

    def get_training_data(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        include_bars: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
        """
        Load features for training.

        Parameters
        ----------
        instrument_id : str
            Instrument to load features for.
        start : datetime
            Start time.
        end : datetime
            End time.
        include_bars : bool, default True
            Whether to join with bar data for labels.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]
            Features array, timestamps array, and feature names.

        """
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)

        # Query features for feature_set_id and time range
        feature_set_id = self._get_feature_set_id()
        query = (
            select(
                self.feature_values_table.c.ts_event,
                self.feature_values_table.c.values,
            )
            .where(
                (self.feature_values_table.c.feature_set_id == feature_set_id)
                & (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event >= start_ns)
                & (self.feature_values_table.c.ts_event <= end_ns),
            )
            .order_by(self.feature_values_table.c.ts_event)
        )

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return np.array([]), np.array([]), []

        # Extract data
        feature_names = self._get_feature_names()
        timestamps = np.array([row[0] for row in rows], dtype=np.int64)
        # Rows contain JSON values map; reconstruct arrays in feature_names order
        feature_arrays: list[list[float]] = []
        for _, values_json in rows:
            mapping = values_json
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except Exception:
                    mapping = {}
            feature_arrays.append([float(mapping.get(name, 0.0)) for name in feature_names])
        features = np.array(feature_arrays, dtype=np.float64)

        return features, timestamps, feature_names

    def _load_bars_from_nautilus(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Load bars from Nautilus PostgreSQL tables.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        start : datetime
            Start time.
        end : datetime
            End time.

        Returns
        -------
        pl.DataFrame
            Bars dataframe with Nautilus schema.

        """
        from ml._imports import HAS_POLARS
        from ml._imports import check_ml_dependencies
        from ml._imports import pl

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)

        # Query Nautilus bar table
        # NOTE: This query uses f-strings for simplicity in example form; values are
        # controlled inputs from the application. If taking user input, parameterize.
        query = f"""  # noqa: S608
        SELECT
            ts_event,
            open,
            high,
            low,
            close,
            volume
        FROM bar
        WHERE instrument_id = '{instrument_id}'
        AND ts_event >= {start_ns}
        AND ts_event <= {end_ns}
        ORDER BY ts_event
        """

        # Use Polars for efficient reading
        df = pl.read_database(query, self.connection_string)
        return df

    def _features_exist(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """
        Check if features already exist for the given range.
        """
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)

        feature_set_id = self._get_feature_set_id()
        query = (
            select(self.feature_values_table.c.ts_event)
            .where(
                (self.feature_values_table.c.feature_set_id == feature_set_id)
                & (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event >= start_ns)
                & (self.feature_values_table.c.ts_event <= end_ns),
            )
            .limit(1)
        )

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchone() is not None

    def _get_feature_names(self) -> list[str]:
        """
        Get feature names from pipeline or config.
        """
        if self.pipeline_runner:
            return self.pipeline_runner.compute_feature_names()
        else:
            # Get from FeatureEngineer
            return self.feature_engineer.get_feature_names()

    def _get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Prefer pipeline signature; otherwise use config hash prefix.
        """
        if self.pipeline_hash:
            return f"fs_{self.pipeline_hash[:12]}"
        return f"fs_{self._compute_config_hash()[:12]}"

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ):
        """
        Clear stored features.

        Parameters
        ----------
        instrument_id : str, optional
            Clear only for specific instrument.
        feature_version : str, optional
            Clear only specific version.

        """
        with self.engine.begin() as conn:
            delete_stmt = self.feature_values_table.delete()

            if instrument_id:
                delete_stmt = delete_stmt.where(
                    self.feature_values_table.c.instrument_id == instrument_id,
                )

            if feature_version:
                delete_stmt = delete_stmt.where(
                    self.feature_values_table.c.feature_version == feature_version,
                )

            conn.execute(delete_stmt)

    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int,
    ) -> None:
        """
        Write computed features to storage.

        This method stores features that have already been computed by an actor.
        It ensures data persistence for training/inference parity tracking.

        Parameters
        ----------
        feature_set_id : str
            Feature set identifier
        instrument_id : str
            Instrument identifier
        features : dict[str, float]
            Feature name to value mapping
        ts_event : int
            Event timestamp in nanoseconds
        ts_init : int
            Initialization timestamp in nanoseconds

        """
        # Convert features dict to JSON for storage
        features_json = json.dumps(features)

        # Use feature set ID as version identifier
        feature_version = self._get_feature_set_id() if hasattr(self, "pipeline_hash") else feature_set_id

        # Prepare data for insertion
        data = {
            "feature_set_id": feature_set_id,
            "feature_version": feature_version,
            "instrument_id": instrument_id,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "features": features_json,
            "computed_at": int(datetime.utcnow().timestamp() * 1e9),
        }

        # Insert with ON CONFLICT for idempotency
        with self.engine.begin() as conn:
            stmt = insert(self.feature_values_table).values(data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["feature_set_id", "instrument_id", "ts_event"],
                set_={
                    "features": stmt.excluded.features,
                    "ts_init": stmt.excluded.ts_init,
                    "computed_at": stmt.excluded.computed_at,
                },
            )
            conn.execute(stmt)

    def flush(self) -> None:
        """
        Flush any pending writes to storage.

        Note: FeatureStore currently writes synchronously, so this is a no-op.
        Future versions may implement write buffering for performance.
        """
        # Currently a no-op as writes are synchronous
        # Future: implement write buffering similar to ModelStore
        pass

    def is_healthy(self) -> bool:
        """
        Check if the feature store is healthy and accessible.

        Returns
        -------
        bool
            True if store is healthy, False otherwise

        """
        try:
            # Try a simple query to verify connection
            with self.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT 1"))
                return result is not None
        except Exception:
            return False
