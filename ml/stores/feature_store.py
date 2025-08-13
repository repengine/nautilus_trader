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
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sqlalchemy import ARRAY
from sqlalchemy import BIGINT
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Float
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

        # Create engine and setup tables
        self.engine: Engine = create_engine(connection_string)
        self.metadata = MetaData()
        self._setup_tables()

        # Feature engineer for computation (ensures parity)
        self.feature_engineer = FeatureEngineer(self.feature_config)

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

    def _setup_tables(self):
        """
        Create feature_values table if it doesn't exist.
        """
        # Define feature_values table
        self.feature_values_table = Table(
            "ml_feature_values",
            self.metadata,
            Column("instrument_id", String(100), primary_key=True),
            Column("ts_event", BIGINT, primary_key=True),  # Nautilus convention: nanoseconds
            Column("feature_version", String(64), primary_key=True),  # Pipeline/config hash
            Column("features", ARRAY(Float)),  # Feature array
            Column("feature_names", JSON),  # Feature name mapping
            Column("created_at", BIGINT),  # When computed (nanoseconds)
            Index("idx_ml_features_instrument_time", "instrument_id", "ts_event"),
            Index("idx_ml_features_version", "feature_version"),
        )

        # Create tables
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

        # Compute features using FeatureEngineer (ensures parity with live)
        features_array, scaler = self.feature_engineer.calculate_features_batch(bars_df)

        # Get feature names
        feature_names = self._get_feature_names()

        # Prepare data for insertion
        rows = []
        timestamps = bars_df["ts_event"].to_numpy()

        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "ts_event": int(ts),
                    "feature_version": self.pipeline_hash,
                    "features": features_array[i].tolist(),
                    "feature_names": feature_names,
                    "created_at": int(datetime.utcnow().timestamp() * 1e9),
                }
            )

        # Bulk insert with upsert
        with self.engine.begin() as conn:
            stmt = insert(self.feature_values_table)
            stmt = stmt.on_conflict_do_update(
                index_elements=["instrument_id", "ts_event", "feature_version"],
                set_={"features": stmt.excluded.features, "created_at": stmt.excluded.created_at},
            )
            conn.execute(stmt, rows)

        return len(rows)

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        Uses the SAME FeatureEngineer as historical computation,
        ensuring perfect parity between training and inference.

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
        # Compute features using same engine as historical
        features = self.feature_engineer.calculate_features_online(
            close_price=float(bar.close),
            high_price=float(bar.high),
            low_price=float(bar.low),
            volume=float(bar.volume),
        )

        # Optionally store for future training
        if store:
            feature_names = self._get_feature_names()
            row = {
                "instrument_id": str(bar.instrument_id),
                "ts_event": bar.ts_event,
                "feature_version": self.pipeline_hash,
                "features": features.tolist(),
                "feature_names": feature_names,
                "created_at": int(datetime.utcnow().timestamp() * 1e9),
            }

            with self.engine.begin() as conn:
                stmt = insert(self.feature_values_table)
                stmt = stmt.on_conflict_do_nothing()
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

        # Query features
        query = (
            select(
                self.feature_values_table.c.ts_event,
                self.feature_values_table.c.features,
                self.feature_values_table.c.feature_names,
            )
            .where(
                (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event >= start_ns)
                & (self.feature_values_table.c.ts_event <= end_ns)
                & (self.feature_values_table.c.feature_version == self.pipeline_hash),
            )
            .order_by(self.feature_values_table.c.ts_event)
        )

        with self.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return np.array([]), np.array([]), []

        # Extract data
        timestamps = np.array([row[0] for row in rows], dtype=np.int64)
        features = np.array([row[1] for row in rows], dtype=np.float64)
        feature_names = rows[0][2] if rows else []

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
        query = f"""
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

        query = (
            select(self.feature_values_table.c.ts_event)
            .where(
                (self.feature_values_table.c.instrument_id == instrument_id)
                & (self.feature_values_table.c.ts_event >= start_ns)
                & (self.feature_values_table.c.ts_event <= end_ns)
                & (self.feature_values_table.c.feature_version == self.pipeline_hash),
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
