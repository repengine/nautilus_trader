"""
Data preparation component for BaseMLTrainer decomposition.

This module provides the DataPreparationComponent which encapsulates data
preparation logic from BaseMLTrainer (lines 284-357 and 1496-1517), including:
- FeatureStore integration for training data preparation
- Label generation from features
- Train/validation data splitting

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from ml.config.base import MLTrainingConfig
    from ml.stores.feature_store import FeatureStore


logger = logging.getLogger(__name__)


class DataPreparationTrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with data preparation component.

    Defines the interface that any trainer must implement to work with
    the DataPreparationComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration.
    _feature_store : FeatureStore | None
        Feature store for data retrieval.
    _feature_names : list[str]
        List of feature names.

    """

    _config: MLTrainingConfig
    _feature_store: FeatureStore | None
    _feature_names: list[str]

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log warning message."""
        ...


class DataPreparationComponent:
    """
    Component responsible for data preparation operations.

    This component encapsulates the data preparation logic from BaseMLTrainer
    (lines 284-357 and 1496-1517), handling:
    - FeatureStore integration for training/inference parity
    - Label generation from features
    - Train/validation data splitting

    The component delegates logging operations to the trainer instance through
    the DataPreparationTrainerProtocol interface, following Protocol-First design.

    Parameters
    ----------
    trainer : DataPreparationTrainerProtocol
        The trainer instance that implements the DataPreparationTrainerProtocol.

    Example
    -------
    >>> from ml.training.common import DataPreparationComponent
    >>> # trainer is an instance implementing DataPreparationTrainerProtocol
    >>> data_prep = DataPreparationComponent(trainer)
    >>> X, y, feature_names = data_prep.prepare_data_with_feature_store(
    ...     instrument_id="BTC-USD",
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 12, 31),
    ... )

    """

    def __init__(self, trainer: DataPreparationTrainerProtocol) -> None:
        """
        Initialize the data preparation component with a trainer reference.

        Parameters
        ----------
        trainer : DataPreparationTrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def prepare_data_with_feature_store(
        self,
        instrument_id: str,
        start: Any,  # datetime
        end: Any,  # datetime
        compute_if_missing: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
        """
        Prepare training data using FeatureStore for guaranteed parity.

        This method ensures training/inference parity by using the same
        FeatureStore that is used during live inference. It:
        1. Optionally computes and stores features if missing
        2. Retrieves features from the store
        3. Generates labels from the features
        4. Updates the trainer's feature names

        Parameters
        ----------
        instrument_id : str
            Instrument to train on.
        start : Any
            Training period start (datetime).
        end : Any
            Training period end (datetime).
        compute_if_missing : bool, default True
            Whether to compute features if not already stored.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]
            X (features), y (labels), feature_names

        Raises
        ------
        ValueError
            If FeatureStore is not configured (db_connection not provided).
        ValueError
            If no features found for the specified instrument and period.

        Example
        -------
        >>> X, y, feature_names = data_prep.prepare_data_with_feature_store(
        ...     instrument_id="BTC-USD",
        ...     start=datetime(2023, 1, 1),
        ...     end=datetime(2023, 12, 31),
        ...     compute_if_missing=True,
        ... )
        >>> assert X.shape[0] == y.shape[0]

        """
        if self._trainer._feature_store is None:
            raise ValueError("FeatureStore not configured. Provide db_connection in config.")

        # Compute and store features if needed
        if compute_if_missing:
            rows_computed = self._trainer._feature_store.compute_and_store_historical(
                instrument_id=instrument_id,
                start=start,
                end=end,
                force_recompute=False,
            )
            if rows_computed > 0:
                self._trainer._log_info(f"Computed {rows_computed} feature rows")

        # Load features from store
        features, timestamps, feature_names = self._trainer._feature_store.get_training_data(
            instrument_id=instrument_id,
            start=start,
            end=end,
            include_bars=True,
        )

        if len(features) == 0:
            raise ValueError(f"No features found for {instrument_id} in specified period")

        # Generate labels (simplified - override in subclass for specific logic)
        labels = self._generate_labels(features, timestamps)

        self._trainer._feature_names = feature_names
        self._trainer._log_info(f"Loaded {len(features)} samples with {len(feature_names)} features")

        return features, labels, feature_names

    def _generate_labels(
        self,
        features: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """
        Generate labels for training.

        This is a default implementation that generates binary labels based
        on the sign of the next return. Override in subclass for specific
        labeling logic.

        Parameters
        ----------
        features : npt.NDArray[np.float64]
            Feature array of shape (n_samples, n_features).
        timestamps : npt.NDArray[np.int64]
            Timestamps for each sample (nanoseconds since epoch).

        Returns
        -------
        npt.NDArray[np.float64]
            Labels array of shape (n_samples,).

        Example
        -------
        >>> labels = data_prep._generate_labels(features, timestamps)
        >>> assert labels.shape[0] == features.shape[0]
        >>> assert set(labels).issubset({0.0, 1.0})

        """
        # Simple example: 1 if next return > 0, else 0
        returns = np.diff(features[:, 0]) if features.shape[1] > 0 else np.array([])
        labels = (returns > 0).astype(np.float64)
        # Pad to match features length
        labels = np.append(labels, 0) if len(labels) < len(features) else labels[: len(features)]
        return labels

    def _split_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
    ) -> tuple[Any, Any]:
        """
        Split data into training and validation sets.

        Performs a time-based split (no shuffling) to preserve temporal
        ordering, which is critical for financial time series data.

        Parameters
        ----------
        data : Any
            The data to split (pl.DataFrame when polars available, or any
            object supporting __len__ and slicing).

        Returns
        -------
        tuple[Any, Any]
            Training and validation datasets as (train_data, val_data).

        Example
        -------
        >>> train_data, val_data = data_prep._split_data(df)
        >>> assert len(train_data) + len(val_data) == len(df)

        """
        n_samples = len(data)
        split_idx = int(n_samples * self._trainer._config.train_test_split)

        return data[:split_idx], data[split_idx:]


__all__ = [
    "DataPreparationComponent",
    "DataPreparationTrainerProtocol",
]
