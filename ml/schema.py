"""
Centralized schema registry for schema→dataset/dataclass/template lookups.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick

from ml.registry.dataclasses import DatasetType


__all__ = [
    "DATASET_TYPE_IDENTIFIER_DEFAULTS",
    "DEFAULT_BAR_IDENTIFIER_TEMPLATE",
    "PREDICTION_SURFACE_V1",
    "PredictionSurfaceSpec",
    "SchemaSpec",
    "dataset_type_to_dataclass",
    "default_identifier_template_for_dataset_type",
    "map_schema_to_dataset_type",
    "schema_spec_for",
    "schema_to_dataclass",
    "schema_to_identifier_template",
    "validate_dataset_type_templates",
    "validate_identifier_template",
    "validate_schema_identifier_templates",
]


@dataclass(frozen=True)
class SchemaSpec:
    """
    Registered schema metadata.
    """

    dataset_type: DatasetType
    data_class: type[Any]
    identifier_template: str

    def __post_init__(self) -> None:
        """
        Validate spec configuration.
        """
        validate_identifier_template(self.identifier_template, label="schema identifier template")


@dataclass(frozen=True)
class PredictionSurfaceSpec:
    """
    Canonical prediction surface specification.

    Attributes
    ----------
    surface : str
        Surface type (e.g., "probability").
    range_min : float
        Minimum value for the surface.
    range_max : float
        Maximum value for the surface.
    threshold : float
        Decision threshold center (default 0.5 for probability).
    neutral_band_supported : bool
        Whether a neutral band is supported around the threshold.
    confidence_semantics : str
        Description of how confidence is derived.
    output_is_logits : bool
        True if raw outputs are logits prior to normalization.
    calibration : Mapping[str, Any] | None
        Optional calibration metadata (method, version, params).
    version : str
        Schema version identifier.

    """

    surface: str
    range_min: float
    range_max: float
    threshold: float = 0.5
    neutral_band_supported: bool = True
    confidence_semantics: str = "max_probability"
    output_is_logits: bool = False
    calibration: Mapping[str, Any] | None = None
    version: str = "v1"

    def to_metadata(self) -> dict[str, Any]:
        """
        Render the surface spec as a JSON-serializable metadata dict.
        """
        metadata: dict[str, Any] = {
            "version": self.version,
            "surface": self.surface,
            "range": {"min": self.range_min, "max": self.range_max},
            "threshold": self.threshold,
            "neutral_band_supported": self.neutral_band_supported,
            "confidence_semantics": self.confidence_semantics,
            "output_is_logits": self.output_is_logits,
        }
        if self.calibration is not None:
            metadata["calibration"] = dict(self.calibration)
        return metadata


PREDICTION_SURFACE_V1 = PredictionSurfaceSpec(
    surface="probability",
    range_min=0.0,
    range_max=1.0,
    threshold=0.5,
    neutral_band_supported=True,
    confidence_semantics="max_probability",
    output_is_logits=False,
    calibration=None,
    version="v1",
)


def validate_identifier_template(template: str, *, label: str) -> str:
    """
    Ensure identifier templates include instrument context.

    Args:
        template: Template string used to build catalog identifiers.
        label: Human-friendly label for error messages.

    Returns:
        The validated template.

    Raises:
        ValueError: If the template is empty or missing ``{instrument_id}``.

    """
    if not template or "{instrument_id}" not in template:
        msg = f"{label} must include '{{instrument_id}}'"
        raise ValueError(msg)
    return template


def _normalize_schema(schema: str) -> str:
    normalized = schema.strip().lower()
    if not normalized:
        msg = "schema cannot be empty"
        raise ValueError(msg)
    return normalized


def _register(
    registry: dict[str, SchemaSpec],
    keys: tuple[str, ...],
    spec: SchemaSpec,
) -> None:
    for key in keys:
        registry[_normalize_schema(key)] = spec


_SCHEMA_REGISTRY: dict[str, SchemaSpec] = {}

DEFAULT_BAR_IDENTIFIER_TEMPLATE = "{instrument_id}-1-MINUTE-LAST-EXTERNAL"
_QUOTE_IDENTIFIER_TEMPLATE = "{instrument_id}"

_register(
    _SCHEMA_REGISTRY,
    ("bars", "bar", "bar_1_minute", "ohlcv", "ohlcv-1m"),
    SchemaSpec(
        dataset_type=DatasetType.BARS,
        data_class=Bar,
        identifier_template=DEFAULT_BAR_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("ohlcv-1h",),
    SchemaSpec(
        dataset_type=DatasetType.BARS,
        data_class=Bar,
        identifier_template="{instrument_id}-1-HOUR-LAST-EXTERNAL",
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("ohlcv-1d",),
    SchemaSpec(
        dataset_type=DatasetType.BARS,
        data_class=Bar,
        identifier_template="{instrument_id}-1-DAY-LAST-EXTERNAL",
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("tbbo", "bbo", "bbo-1s", "bbo-1m", "tcbbo"),
    SchemaSpec(
        dataset_type=DatasetType.TBBO,
        data_class=QuoteTick,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("quote", "quotes", "quote_tick", "quote-tick"),
    SchemaSpec(
        dataset_type=DatasetType.QUOTES,
        data_class=QuoteTick,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("trades", "trade"),
    SchemaSpec(
        dataset_type=DatasetType.TRADES,
        data_class=TradeTick,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("mbp-1", "mbp-10", "mbp", "mbo"),
    SchemaSpec(
        dataset_type=DatasetType.MBP1,
        data_class=QuoteTick,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)

# ------------------------------------------------------------------
# Tier-1 and feature-family schemas (non-Nautilus dataclasses)
# ------------------------------------------------------------------

_register(
    _SCHEMA_REGISTRY,
    ("earnings", "earnings_actuals", "earnings-actuals"),
    SchemaSpec(
        dataset_type=DatasetType.EARNINGS_ACTUALS,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("earnings_estimates", "earnings-estimates"),
    SchemaSpec(
        dataset_type=DatasetType.EARNINGS_ESTIMATES,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("macro_release_calendar", "macro_releases", "macro-release-calendar"),
    SchemaSpec(
        dataset_type=DatasetType.MACRO_RELEASES,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("macro_observations", "macro-observations"),
    SchemaSpec(
        dataset_type=DatasetType.MACRO_OBSERVATIONS,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("events_calendar", "events-calendar"),
    SchemaSpec(
        dataset_type=DatasetType.EVENTS_CALENDAR,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("microstructure_minute", "microstructure-minute"),
    SchemaSpec(
        dataset_type=DatasetType.MICRO_MINUTE_FEATURES,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("l2_minute", "l2-minute"),
    SchemaSpec(
        dataset_type=DatasetType.L2_MINUTE_FEATURES,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("feature_values", "feature-values", "features"),
    SchemaSpec(
        dataset_type=DatasetType.FEATURES,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("predictions", "prediction", "model_predictions", "model-predictions"),
    SchemaSpec(
        dataset_type=DatasetType.PREDICTIONS,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)
_register(
    _SCHEMA_REGISTRY,
    ("signals", "signal", "strategy_signals", "strategy-signals"),
    SchemaSpec(
        dataset_type=DatasetType.SIGNALS,
        data_class=dict,
        identifier_template=_QUOTE_IDENTIFIER_TEMPLATE,
    ),
)


def schema_spec_for(schema: str) -> SchemaSpec:
    """
    Return the registered schema spec.

    Args:
        schema: Schema token to resolve (case-insensitive).

    Returns:
        SchemaSpec containing dataset type, data class, and identifier template.

    Raises:
        ValueError: If the schema is not registered.

    """
    normalized = _normalize_schema(schema)
    try:
        return _SCHEMA_REGISTRY[normalized]
    except KeyError as exc:
        msg = f"Unknown schema '{schema}'"
        raise ValueError(msg) from exc


def map_schema_to_dataset_type(schema: str) -> DatasetType:
    """
    Map a schema string to the corresponding DatasetType.

    Raises
    ------
    ValueError
        If the schema is not registered.

    """
    return schema_spec_for(schema).dataset_type


def schema_to_dataclass(schema: str) -> type[Any]:
    """
    Return the Nautilus data class inferred from a schema string.

    Raises
    ------
    ValueError
        If the schema is not registered.

    """
    return schema_spec_for(schema).data_class


def schema_to_identifier_template(schema: str) -> str:
    """
    Return the default identifier template for a schema.

    Raises
    ------
    ValueError
        If the schema is not registered.

    """
    return schema_spec_for(schema).identifier_template


def _build_dataset_defaults(registry: Mapping[str, SchemaSpec]) -> dict[DatasetType, str]:
    defaults: dict[DatasetType, str] = {}
    for spec in registry.values():
        defaults.setdefault(spec.dataset_type, spec.identifier_template)
    return defaults


DATASET_TYPE_IDENTIFIER_DEFAULTS = _build_dataset_defaults(_SCHEMA_REGISTRY)
DATASET_TYPE_IDENTIFIER_DEFAULTS.setdefault(DatasetType.QUOTES, _QUOTE_IDENTIFIER_TEMPLATE)


def default_identifier_template_for_dataset_type(dataset_type: DatasetType) -> str:
    """
    Return the default identifier template for a dataset type.

    Args:
        dataset_type: Dataset type to resolve.

    Returns:
        Template string for identifier resolution.

    Raises:
        ValueError: If no default exists for the dataset type.

    """
    try:
        return DATASET_TYPE_IDENTIFIER_DEFAULTS[dataset_type]
    except KeyError as exc:
        msg = f"No identifier template registered for dataset type {dataset_type!r}"
        raise ValueError(msg) from exc


_DATA_CLASS_BY_DATASET: dict[DatasetType, type[Any]] = {
    DatasetType.QUOTES: QuoteTick,
}
for spec in _SCHEMA_REGISTRY.values():
    _DATA_CLASS_BY_DATASET.setdefault(spec.dataset_type, spec.data_class)


def dataset_type_to_dataclass(dataset_type: DatasetType) -> type[Any]:
    """
    Return the Nautilus data class for a DatasetType.
    """
    try:
        return _DATA_CLASS_BY_DATASET[dataset_type]
    except KeyError as exc:
        msg = f"No dataclass registered for dataset type {dataset_type!r}"
        raise ValueError(msg) from exc


def validate_schema_identifier_templates(
    templates: Mapping[str, str] | None,
) -> dict[str, str]:
    """
    Validate and normalize schema→identifier template overrides.

    Args:
        templates: Optional mapping of schema tokens to identifier templates.

    Returns:
        Normalized mapping keyed by lowercased schema.

    Raises:
        ValueError: If a schema is unknown or a template is invalid.

    """
    if templates is None:
        return {}
    normalized: dict[str, str] = {}
    for key, template in templates.items():
        schema_spec_for(key)
        normalized_key = _normalize_schema(key)
        normalized[normalized_key] = validate_identifier_template(
            template,
            label=f"schema template for {normalized_key}",
        )
    return normalized


def validate_dataset_type_templates(
    templates: Mapping[DatasetType, str] | None,
) -> dict[DatasetType, str]:
    """
    Validate dataset-type→identifier template overrides.

    Args:
        templates: Optional mapping keyed by DatasetType.

    Returns:
        Copy of validated templates.

    Raises:
        ValueError: If keys are not DatasetType or templates are invalid.

    """
    if templates is None:
        return {}
    normalized: dict[DatasetType, str] = {}
    for dataset_type, template in templates.items():
        if not isinstance(dataset_type, DatasetType):
            msg = (
                f"dataset_type_identifier_templates keys must be DatasetType (got {dataset_type!r})"
            )
            raise ValueError(msg)
        normalized[dataset_type] = validate_identifier_template(
            template,
            label=f"dataset type template for {dataset_type.value}",
        )
    return normalized
