#!/usr/bin/env python3

"""
Data contract management component.

This module provides functionality for creating and retrieving data contracts from dataset
manifests with support for both JSON and PostgreSQL backends.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType


if TYPE_CHECKING:
    from ml.registry.dataclasses import DatasetManifest
    from ml.registry.manifest_manager import ManifestManagerProtocol
    from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


class ContractManagerProtocol(Protocol):
    """
    Protocol for contract management operations.

    This protocol defines the interface for data contract creation and retrieval from
    dataset manifests.

    """

    def create_contract_from_manifest(
        self,
        manifest: DatasetManifest,
    ) -> DataContract:
        """
        Create a data contract from a dataset manifest.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to create contract from

        Returns
        -------
        DataContract
            Created data contract

        """
        ...

    def get_contract(
        self,
        dataset_id: str,
        manifest_manager: ManifestManagerProtocol,
        persistence: PersistenceManager,
    ) -> DataContract:
        """
        Get the data contract for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to get contract for
        manifest_manager : ManifestManagerProtocol
            Manifest manager for retrieving manifests
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        DataContract
            The data contract

        Raises
        ------
        ValueError
            If dataset or contract doesn't exist

        """
        ...


class ContractManager:
    """
    Manages data contract lifecycle operations.

    This component handles creation and retrieval of data contracts from dataset manifests.
    Contracts are created from manifest constraints and cached for performance.

    Thread-safe operations are expected to be coordinated by the parent registry.

    Examples
    --------
    >>> manager = ContractManager()
    >>> contract = manager.create_contract_from_manifest(manifest)
    >>> print(contract.enforcement_mode)
    strict

    """

    def __init__(self) -> None:
        """Initialize contract manager with empty cache."""
        self._contracts: dict[str, DataContract] = {}

    def create_contract_from_manifest(
        self,
        manifest: DatasetManifest,
    ) -> DataContract:
        """
        Create a data contract from a dataset manifest.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to create contract from

        Returns
        -------
        DataContract
            Created data contract

        Examples
        --------
        >>> contract = manager.create_contract_from_manifest(manifest)
        >>> print(f"Contract has {len(contract.validation_rules)} rules")

        """
        rules: list[ValidationRule] = []
        constraints = manifest.constraints or {}

        # Convert constraints to validation rules
        if "ranges" in constraints:
            for field, range_spec in constraints["ranges"].items():
                if "min" in range_spec or "max" in range_spec:
                    rules.append(
                        ValidationRule(
                            rule_type=ValidationRuleType.RANGE,
                            field_name=field,
                            parameters=range_spec,
                            severity=QualityFlag.FAIL,
                            description=f"Range validation for {field}",
                        ),
                    )

        if "nullability" in constraints:
            for field, nullable in constraints["nullability"].items():
                if not nullable:
                    rules.append(
                        ValidationRule(
                            rule_type=ValidationRuleType.NULLABILITY,
                            field_name=field,
                            parameters={"nullable": False},
                            severity=QualityFlag.FAIL,
                            description=f"{field} cannot be null",
                        ),
                    )

        # Regex constraints (e.g., IDs)
        if "regex" in constraints:
            for field, pattern in constraints["regex"].items():
                if isinstance(pattern, str) and pattern:
                    rules.append(
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            field_name=str(field),
                            parameters={"pattern": pattern},
                            severity=QualityFlag.FAIL,
                            description=f"Regex validation for {field}",
                        ),
                    )

        # Per-dataset null-rate threshold
        quality_thresholds: dict[str, float] = {}
        try:
            thr = constraints.get("null_rate_threshold")
            if isinstance(thr, (int, float)) and 0.0 <= float(thr) <= 1.0:
                quality_thresholds["null_rate"] = float(thr)
        except Exception:
            quality_thresholds = {}

        # Create default rule if no rules defined
        if not rules:
            rules.append(
                ValidationRule(
                    rule_type=ValidationRuleType.TYPE_CHECK,
                    field_name="*",
                    parameters={},
                    severity=QualityFlag.WARN,
                    description="Type validation for all fields",
                ),
            )

        contract = DataContract(
            contract_id=f"{manifest.dataset_id}_contract",
            dataset_id=manifest.dataset_id,
            version="1.0.0",
            validation_rules=rules,
            quality_thresholds=quality_thresholds,
            enforcement_mode="strict",
            created_at=manifest.created_at,
            last_modified=manifest.last_modified,
        )

        # Cache the contract
        self._contracts[manifest.dataset_id] = contract

        logger.debug("Created contract for dataset '%s' with %d rules", manifest.dataset_id, len(rules))
        return contract

    def get_contract(
        self,
        dataset_id: str,
        manifest_manager: ManifestManagerProtocol,
        persistence: PersistenceManager,
    ) -> DataContract:
        """
        Get the data contract for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to get contract for
        manifest_manager : ManifestManagerProtocol
            Manifest manager for retrieving manifests
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        DataContract
            The data contract

        Raises
        ------
        ValueError
            If dataset or contract doesn't exist

        Examples
        --------
        >>> contract = manager.get_contract("bars_eurusd_1m", manifest_mgr, persistence)
        >>> print(contract.enforcement_mode)
        strict

        """
        # Check cache first
        if dataset_id in self._contracts:
            return self._contracts[dataset_id]

        # Create contract from manifest if not cached
        manifest = manifest_manager.get_manifest(dataset_id, persistence)
        contract = self.create_contract_from_manifest(manifest)

        return contract

    def _contract_to_dict(self, contract: DataContract) -> dict[str, Any]:
        """
        Convert DataContract to dictionary.

        Parameters
        ----------
        contract : DataContract
            Data contract to convert

        Returns
        -------
        dict[str, Any]
            Dictionary representation of contract

        """
        rules = []
        for rule in contract.validation_rules:
            rules.append(
                {
                    "rule_type": rule.rule_type.value,
                    "field_name": rule.field_name,
                    "parameters": rule.parameters,
                    "severity": rule.severity.value,
                    "description": rule.description,
                },
            )

        return {
            "contract_id": contract.contract_id,
            "dataset_id": contract.dataset_id,
            "version": contract.version,
            "validation_rules": rules,
            "quality_thresholds": contract.quality_thresholds,
            "enforcement_mode": contract.enforcement_mode,
            "created_at": contract.created_at,
            "last_modified": contract.last_modified,
            "metadata": contract.metadata,
        }

    def _dict_to_contract(self, data: dict[str, Any]) -> DataContract:
        """
        Convert dictionary to DataContract.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation of contract

        Returns
        -------
        DataContract
            Data contract object

        """
        # Convert validation rules
        rules = []
        for rule_data in data.get("validation_rules", []):
            rule_data["rule_type"] = ValidationRuleType(rule_data["rule_type"])
            rule_data["severity"] = QualityFlag(rule_data["severity"])
            rules.append(ValidationRule(**rule_data))

        data["validation_rules"] = rules
        return DataContract(**data)


__all__ = [
    "ContractManager",
    "ContractManagerProtocol",
]
