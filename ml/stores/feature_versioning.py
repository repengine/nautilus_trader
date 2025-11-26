"""
Feature versioning and identification for FeatureStore.

This module handles feature set identification, configuration hashing, and feature name
management for hot/cold paths.

"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast


if TYPE_CHECKING:
    from ml.features import FeatureConfig as EngineeringFeatureConfig
    from ml.features.config import FeatureConfig as FacadeFeatureConfig
    from ml.features.pipeline import PipelineRunner

    FeatureConfigLike = FacadeFeatureConfig | EngineeringFeatureConfig
else:
    # Runtime placeholder - actual type validation is structural
    FeatureConfigLike: type = object  # type: ignore[misc]


logger = logging.getLogger(__name__)


class FeatureVersioningProtocol(Protocol):
    """
    Protocol for feature versioning operations.
    """

    def compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration.
        """
        ...

    def get_feature_set_id(self) -> str:
        """
        Get stable feature set identifier.
        """
        ...

    def get_feature_names(self, online: bool = False) -> list[str]:
        """
        Get feature names for given mode.
        """
        ...


class FeatureVersioning:
    """
    Handles feature set identification and versioning.

    Responsibilities:
    - Compute stable hashes for feature configurations
    - Derive feature set IDs
    - Manage feature name lists for hot/cold paths

    """

    def __init__(
        self,
        feature_config: "FeatureConfigLike",  # type: ignore[valid-type]  # noqa: UP037
        pipeline_runner_offline: "PipelineRunner | None" = None,  # noqa: UP037
        pipeline_runner_online: "PipelineRunner | None" = None,  # noqa: UP037
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize feature versioning.

        Parameters
        ----------
        feature_config : FeatureConfigLike
            Feature engineering configuration
        pipeline_runner_offline : PipelineRunner | None
            Offline (batch) pipeline runner for feature names
        pipeline_runner_online : PipelineRunner | None
            Online (realtime) pipeline runner for feature names
        logger : logging.Logger | None
            Logger for operations

        """
        self._feature_config = feature_config
        self._pipeline_runner_offline = pipeline_runner_offline
        self._pipeline_runner_online = pipeline_runner_online
        self._logger = logger or logging.getLogger(__name__)

        # Compute pipeline hash if available
        self._pipeline_hash: str = ""
        if self._pipeline_runner_offline:
            self._pipeline_hash = self._pipeline_runner_offline.compute_signature()

        # Cache config hash
        self._config_hash = self._compute_config_hash()

    def compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.

        Returns
        -------
        str
            16-character hex hash of configuration

        """
        return self._config_hash

    def _compute_config_hash(self) -> str:
        """
        Internal method to compute config hash.

        Handles both dict-like and dataclass objects.

        Returns
        -------
        str
            16-character hex hash

        """
        # Handle both dict-like and dataclass objects
        if hasattr(self._feature_config, "__dict__"):
            config_dict = self._feature_config.__dict__  # type: ignore[attr-defined]
        else:
            # For frozen dataclasses, convert to dict
            try:
                import msgspec

                config_dict = msgspec.to_builtins(self._feature_config)
            except Exception:
                self._logger.warning(
                    "Failed to convert feature_config to dict, using empty dict",
                )
                config_dict = {}

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Prefer pipeline signature; otherwise use config hash prefix.

        Returns
        -------
        str
            Feature set identifier (e.g., "fs_abc123def456")

        """
        if self._pipeline_hash:
            return f"fs_{self._pipeline_hash[:12]}"
        return f"fs_{self._config_hash[:12]}"

    def get_feature_names(self, online: bool = False) -> list[str]:
        """
        Get feature names for specified mode.

        Parameters
        ----------
        online : bool, default False
            If True, return online (hot-path) feature names with L1_ONLY gating.
            If False, return offline (cold-path) feature names with full L1_L2.

        Returns
        -------
        list[str]
            List of feature names

        """
        if online:
            return self._get_feature_names_online()
        return self._get_feature_names_offline()

    def _get_feature_names_offline(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.

        Returns
        -------
        list[str]
            Offline feature names

        """
        if self._pipeline_runner_offline:
            return self._pipeline_runner_offline.compute_feature_names()

        # Get from FeatureEngineer (if no pipeline specified)
        # Import locally to avoid circular dependencies
        from ml.features import FeatureEngineer

        engineer = FeatureEngineer(cast(Any, self._feature_config))  # type: ignore[operator]
        return list(engineer.get_feature_names())

    def _get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names from pipeline or config with L1_ONLY gating.

        Returns
        -------
        list[str]
            Online feature names

        """
        if self._pipeline_runner_online:
            return self._pipeline_runner_online.compute_feature_names()

        # Derive from current FeatureEngineer configuration if no pipeline_spec provided
        from ml.features import FeatureEngineer
        from ml.features.pipeline import PipelineRunner
        from ml.registry.base import DataRequirements

        engineer = FeatureEngineer(cast(Any, self._feature_config))  # type: ignore[operator]
        spec = engineer.build_pipeline_spec_from_config()
        return PipelineRunner(spec, allowable=DataRequirements.L1_ONLY).compute_feature_names()
