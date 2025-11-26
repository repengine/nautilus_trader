"""
FeatureStoreAccessor component - encapsulates feature store read/write/validation logic.

Extracted from FeatureEngineer god class (Phase 2.1.1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from ml.stores.protocols import FeatureStoreStrictProtocol

    DataFrameLike = pd.DataFrame | pl.DataFrame
else:
    DataFrameLike = Any


logger = logging.getLogger(__name__)


class FeatureStoreAccessor:
    """
    Encapsulates feature store access operations.

    Provides a clean interface between feature engineering logic and the FeatureStore,
    handling read, write, and schema validation operations with graceful degradation
    when the store is unavailable.

    Parameters
    ----------
    feature_store : FeatureStoreStrictProtocol | None
        The feature store instance to use for persistence. If None, operations
        will gracefully degrade (returning None for reads, False for writes).

    Examples
    --------
    >>> from ml.stores.feature_store import FeatureStore
    >>> store = FeatureStore(connection_string="postgresql://...")
    >>> accessor = FeatureStoreAccessor(feature_store=store)
    >>>
    >>> # Read features
    >>> features_df = accessor.read_features_from_store(
    ...     instrument_id="SPY",
    ...     ts_start=1609459200000000000,
    ...     ts_end=1609545600000000000,
    ... )
    >>>
    >>> # Write features
    >>> import pandas as pd
    >>> features = pd.DataFrame({"sma_20": [100.5], "rsi_14": [55.3]})
    >>> success = accessor.write_features_to_store(
    ...     instrument_id="SPY",
    ...     features_df=features,
    ...     ts_event=1609459200000000000,
    ...     ts_init=1609459200000000100,
    ... )
    >>>
    >>> # Validate schema
    >>> is_valid, errors = accessor.validate_feature_schema(
    ...     features_df=features,
    ...     expected_columns=["sma_20", "rsi_14"],
    ... )
    """

    def __init__(
        self,
        feature_store: FeatureStoreStrictProtocol | None = None,
    ) -> None:
        """
        Initialize the FeatureStoreAccessor.

        Parameters
        ----------
        feature_store : FeatureStoreStrictProtocol | None
            The feature store instance. If None, operations will degrade gracefully.
        """
        self._feature_store = feature_store

    def read_features_from_store(
        self,
        instrument_id: str,
        ts_start: int,
        ts_end: int,
        *,
        feature_names: list[str] | None = None,
    ) -> DataFrameLike | None:
        """
        Read features from the FeatureStore for a given instrument and time range.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier (e.g., "SPY", "AAPL.NASDAQ").
        ts_start : int
            Start timestamp in nanoseconds since epoch.
        ts_end : int
            End timestamp in nanoseconds since epoch.
        feature_names : list[str] | None, optional
            Optional list of specific feature columns to retrieve. If None, returns
            all available features.

        Returns
        -------
        pd.DataFrame | pl.DataFrame | None
            DataFrame containing features with columns: instrument_id, ts_event,
            ts_init, and feature columns. Returns None if:
            - FeatureStore is not available
            - No features found for the specified time range
            - An error occurred during read operation

        Examples
        --------
        >>> accessor = FeatureStoreAccessor(feature_store=store)
        >>> df = accessor.read_features_from_store(
        ...     "SPY",
        ...     1609459200000000000,  # 2021-01-01 00:00:00 UTC
        ...     1609545600000000000,  # 2021-01-02 00:00:00 UTC
        ... )
        >>> assert "ts_event" in df.columns
        >>> assert "instrument_id" in df.columns
        """
        # Graceful degradation when store unavailable
        if self._feature_store is None:
            logger.warning(
                "FeatureStore not available, cannot read features",
                extra={
                    "instrument_id": instrument_id,
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                },
            )
            return None

        try:
            # Read from store using read_range method
            # Note: FeatureStore.read_range returns pd.DataFrame
            df = self._feature_store.read_range(  # type: ignore[attr-defined]
                start_ns=ts_start,
                end_ns=ts_end,
                instrument_id=instrument_id,
            )

            # Check if any data was returned
            if df is None or len(df) == 0:
                logger.debug(
                    "No features found for instrument in specified time range",
                    extra={
                        "instrument_id": instrument_id,
                        "ts_start": ts_start,
                        "ts_end": ts_end,
                    },
                )
                return None

            # Filter to specific features if requested
            if feature_names is not None:
                # Keep mandatory columns plus requested features
                mandatory_cols = ["instrument_id", "ts_event", "ts_init"]
                available_features = [col for col in feature_names if col in df.columns]

                if not available_features:
                    logger.warning(
                        "None of the requested features found in store",
                        extra={
                            "instrument_id": instrument_id,
                            "requested_features": feature_names,
                            "available_columns": list(df.columns),
                        },
                    )
                    return None

                # Select only requested columns (plus mandatory)
                cols_to_select = mandatory_cols + available_features
                df = df[cols_to_select]

            return df  # type: ignore[no-any-return]

        except Exception:
            logger.error(
                "Error reading features from store",
                extra={
                    "instrument_id": instrument_id,
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                },
                exc_info=True,
            )
            return None

    def write_features_to_store(
        self,
        instrument_id: str,
        features_df: DataFrameLike,
        *,
        ts_event: int,
        ts_init: int,
    ) -> bool:
        """
        Write computed feature values to the FeatureStore.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier (e.g., "SPY", "AAPL.NASDAQ").
        features_df : pd.DataFrame | pl.DataFrame
            DataFrame containing feature values. Each row represents a feature vector.
        ts_event : int
            Event timestamp in nanoseconds since epoch (when the features were generated).
        ts_init : int
            Initialization timestamp in nanoseconds since epoch (when the write occurred).
            Must be >= ts_event.

        Returns
        -------
        bool
            True if write was successful, False if:
            - FeatureStore is not available
            - Schema validation failed
            - Timestamp validation failed (ts_init < ts_event)
            - An error occurred during write operation

        Examples
        --------
        >>> accessor = FeatureStoreAccessor(feature_store=store)
        >>> features = pd.DataFrame({
        ...     "sma_20": [100.5, 101.2],
        ...     "rsi_14": [55.3, 58.7],
        ... })
        >>> success = accessor.write_features_to_store(
        ...     instrument_id="SPY",
        ...     features_df=features,
        ...     ts_event=1609459200000000000,
        ...     ts_init=1609459200000000100,
        ... )
        >>> assert success is True
        """
        # Graceful degradation when store unavailable
        if self._feature_store is None:
            logger.warning(
                "FeatureStore not available, cannot write features",
                extra={
                    "instrument_id": instrument_id,
                    "ts_event": ts_event,
                    "ts_init": ts_init,
                    "feature_count": len(features_df.columns),
                },
            )
            return False

        # Validate timestamps
        if ts_init < ts_event:
            logger.error(
                "Invalid timestamps: ts_init must be >= ts_event",
                extra={
                    "instrument_id": instrument_id,
                    "ts_event": ts_event,
                    "ts_init": ts_init,
                    "delta_ns": ts_init - ts_event,
                },
            )
            return False

        # Validate schema before writing
        is_valid, errors = self.validate_feature_schema(
            features_df=features_df,
            strict=False,  # Allow extra columns (they won't be written)
        )

        if not is_valid:
            logger.error(
                "Schema validation failed, cannot write features",
                extra={
                    "instrument_id": instrument_id,
                    "validation_errors": errors,
                    "columns": list(features_df.columns),
                },
            )
            return False

        try:
            # Convert features to dict format for FeatureStore.write_features()
            # Handle both pandas and polars DataFrames
            if hasattr(features_df, "to_dict"):
                # Pandas DataFrame
                # Get first row as dict (if multiple rows, write each separately)
                import pandas as pd_runtime
                if isinstance(features_df, pd_runtime.DataFrame):
                    for idx in range(len(features_df)):
                        row = features_df.iloc[idx].to_dict()
                        self._feature_store.write_features(
                            feature_set_id="default",  # Default feature set ID
                            instrument_id=instrument_id,
                            features=row,
                            ts_event=ts_event,
                            ts_init=ts_init,
                        )
            else:
                # Polars DataFrame
                for idx in range(len(features_df)):
                    row_dict = features_df.row(idx, named=True)  # type: ignore[operator]
                    self._feature_store.write_features(
                        feature_set_id="default",
                        instrument_id=instrument_id,
                        features=row_dict,
                        ts_event=ts_event,
                        ts_init=ts_init,
                    )

            logger.debug(
                "Successfully wrote features to store",
                extra={
                    "instrument_id": instrument_id,
                    "ts_event": ts_event,
                    "ts_init": ts_init,
                    "row_count": len(features_df),
                    "feature_count": len(features_df.columns),
                },
            )

            return True

        except Exception:
            logger.error(
                "Error writing features to store",
                extra={
                    "instrument_id": instrument_id,
                    "ts_event": ts_event,
                    "ts_init": ts_init,
                },
                exc_info=True,
            )
            return False

    def validate_feature_schema(
        self,
        features_df: DataFrameLike,
        *,
        expected_columns: list[str] | None = None,
        strict: bool = True,
    ) -> tuple[bool, list[str]]:
        """
        Validate that a feature DataFrame conforms to the expected schema.

        Parameters
        ----------
        features_df : pd.DataFrame | pl.DataFrame
            The DataFrame to validate.
        expected_columns : list[str] | None, optional
            Expected column names. If None, validation only checks that the DataFrame
            is not empty and has valid structure.
        strict : bool, default True
            If True, reject DataFrames with extra columns not in expected_columns.
            If False, allow extra columns (they will be ignored during processing).

        Returns
        -------
        tuple[bool, list[str]]
            A tuple of (is_valid, list_of_errors).
            - is_valid: True if validation passed, False otherwise.
            - list_of_errors: List of validation error messages. Empty if valid.

        Examples
        --------
        >>> accessor = FeatureStoreAccessor()
        >>> features = pd.DataFrame({"sma_20": [100.5], "rsi_14": [55.3]})
        >>> is_valid, errors = accessor.validate_feature_schema(
        ...     features_df=features,
        ...     expected_columns=["sma_20", "rsi_14"],
        ... )
        >>> assert is_valid is True
        >>> assert errors == []
        """
        errors: list[str] = []

        # Check that DataFrame is not empty
        if len(features_df) == 0:
            errors.append("DataFrame is empty (0 rows)")

        # Get actual columns (handle both pandas and polars)
        if hasattr(features_df, "columns"):
            actual_columns = set(features_df.columns)
        else:
            # Polars uses .columns as well
            actual_columns = set(features_df.columns)

        # If no expected columns specified, just check non-empty
        if expected_columns is None:
            if len(actual_columns) == 0:
                errors.append("DataFrame has no columns")
            return (len(errors) == 0, errors)

        expected_set = set(expected_columns)

        # Check for missing columns
        missing = expected_set - actual_columns
        if missing:
            errors.append(f"Missing columns: {', '.join(sorted(missing))}")

        # Check for extra columns (if strict mode)
        if strict:
            extra = actual_columns - expected_set
            if extra:
                errors.append(f"Unexpected columns: {', '.join(sorted(extra))}")

        return (len(errors) == 0, errors)
