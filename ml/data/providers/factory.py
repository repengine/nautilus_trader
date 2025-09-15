"""
Factory pattern for creating and managing data providers.

This module provides a factory for creating data providers and an adapter for connecting
feature transforms to their corresponding providers.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, cast

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.ml_types import PolarsDF, PolarsSeries
from ml.data.providers.base import BaseStaticProvider
from ml.data.providers.base import BaseTimeSeriesProvider
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider
from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import PandasCalendarSource
from ml.data.sources.events import MockEventSource
from ml.data.sources.metadata import MockMetadataSource


if TYPE_CHECKING:
    from ml.data.sources.calendar import CalendarSource
    from ml.data.sources.events import EventSource
    from ml.data.sources.metadata import MetadataSource
    from ml.features.pipeline import TransformSpec


logger = logging.getLogger(__name__)

DataProvider: TypeAlias = BaseStaticProvider | BaseTimeSeriesProvider


class ProviderFactory:
    """
    Factory for creating and managing data providers.

    This factory implements the singleton pattern for providers,
    ensuring that the same provider instance is reused throughout
    the application lifecycle.

    Attributes
    ----------
    _providers : dict[str, DataProvider]
        Cache of created provider instances
    _metadata_source : MetadataSource
        Source for instrument metadata
    _calendar_source : CalendarSource
        Source for calendar data
    _event_source : EventSource
        Source for event data

    """

    def __init__(
        self,
        metadata_source: MetadataSource | None = None,
        calendar_source: CalendarSource | None = None,
        event_source: EventSource | None = None,
    ) -> None:
        """
        Initialize provider factory.

        Parameters
        ----------
        metadata_source : MetadataSource, optional
            Source for instrument metadata. Uses MockMetadataSource if None.
        calendar_source : CalendarSource, optional
            Source for calendar data. Uses MockCalendarSource if None.
        event_source : EventSource, optional
            Source for event data. Uses MockEventSource if None.

        """
        # Initialize sources (use real implementations when available, mocks as fallback)
        self._metadata_source = metadata_source or MockMetadataSource()

        # Try to use PandasCalendarSource by default, fall back to Mock if unavailable
        if calendar_source is None:
            try:
                self._calendar_source: CalendarSource = PandasCalendarSource()
                logger.info("Using PandasCalendarSource for real market calendar data")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize PandasCalendarSource: {e}, using MockCalendarSource",
                )
                self._calendar_source = MockCalendarSource()
        else:
            self._calendar_source = calendar_source

        self._event_source = event_source or MockEventSource()

        # Creator registry (Open/Closed): name -> zero-arg factory
        self._creators: dict[str, Callable[[], DataProvider]] = {
            "metadata": lambda: InstrumentMetadataProvider(self._metadata_source),
            "calendar": lambda: MarketCalendarProvider(self._calendar_source),
            "events": lambda: EventScheduleProvider(self._event_source),
        }

        # Provider cache (singleton pattern)
        self._providers: dict[str, DataProvider] = {}

        logger.info("Initialized ProviderFactory with sources")

    def get_metadata_provider(self) -> InstrumentMetadataProvider:
        """
        Get or create metadata provider.

        Returns
        -------
        InstrumentMetadataProvider
            Provider for instrument metadata

        """
        if "metadata" not in self._providers:
            self._providers["metadata"] = self._creators["metadata"]()
            logger.debug("Created InstrumentMetadataProvider")

        provider = self._providers["metadata"]
        assert isinstance(provider, InstrumentMetadataProvider)
        return provider

    def get_calendar_provider(self) -> MarketCalendarProvider:
        """
        Get or create calendar provider.

        Returns
        -------
        MarketCalendarProvider
            Provider for calendar features

        """
        if "calendar" not in self._providers:
            self._providers["calendar"] = self._creators["calendar"]()
            logger.debug("Created MarketCalendarProvider")

        provider = self._providers["calendar"]
        assert isinstance(provider, MarketCalendarProvider)
        return provider

    def get_event_provider(self) -> EventScheduleProvider:
        """
        Get or create event provider.

        Returns
        -------
        EventScheduleProvider
            Provider for event features

        """
        if "events" not in self._providers:
            self._providers["events"] = self._creators["events"]()
            logger.debug("Created EventScheduleProvider")

        provider = self._providers["events"]
        assert isinstance(provider, EventScheduleProvider)
        return provider

    def register_provider_creator(self, name: str, creator: Callable[[], DataProvider]) -> None:
        """
        Register a zero-argument creator function for a provider name.

        The creator may close over factory-managed sources or any external
        dependencies to construct the provider lazily when requested.

        Examples
        --------
        >>> factory = ProviderFactory()
        >>> def make_macro_provider() -> DataProvider:
        ...     from ml.data.providers.macro import MacroProvider
        ...     return MacroProvider()
        >>> factory.register_provider_creator("macro", make_macro_provider)
        >>> provider = factory.get_provider("macro")
        >>> type(provider).__name__
        'MacroProvider'

        """
        self._creators[name] = creator
        logger.info(f"Registered provider creator: {name}")

    def get_provider(self, name: str) -> DataProvider:
        """
        Get provider by name.

        Parameters
        ----------
        name : str
            Provider name ("metadata", "calendar", "events")

        Returns
        -------
        DataProvider
            The requested provider

        Raises
        ------
        ValueError
            If provider name is unknown

        """
        # Return cached instance if available
        if name in self._providers:
            return self._providers[name]

        # Create via registered creator if present
        creator = self._creators.get(name)
        if creator is not None:
            provider = creator()
            self._providers[name] = provider
            logger.debug(f"Created provider via registry: {name}")
            return provider

        # For compatibility, fall back to known names
        if name == "metadata":
            return self.get_metadata_provider()
        if name == "calendar":
            return self.get_calendar_provider()
        if name == "events":
            return self.get_event_provider()

        msg = f"Unknown provider: {name}"
        raise ValueError(msg)

    def register_provider(
        self,
        name: str,
        provider: DataProvider,
    ) -> None:
        """
        Register a custom provider.

        Parameters
        ----------
        name : str
            Name to register provider under
        provider : DataProvider
            Provider instance to register

        """
        self._providers[name] = provider
        logger.info(f"Registered custom provider: {name}")


class TransformProviderAdapter:
    """
    Adapter between feature transforms and data providers.

    This adapter maps transform specifications to their corresponding
    data providers and handles data loading for transforms.

    Attributes
    ----------
    _factory : ProviderFactory
        Factory for creating providers
    _transform_mappings : dict[str, str]
        Mapping from transform names to provider names
    _provider_cache : dict[str, DataProvider]
        Cache of providers by transform name

    """

    def __init__(self, factory: ProviderFactory) -> None:
        """
        Initialize transform provider adapter.

        Parameters
        ----------
        factory : ProviderFactory
            Factory for creating providers

        """
        self._factory = factory

        # Default mappings from transforms to providers
        self._transform_mappings = {
            "calendar": "calendar",
            "static_covariates": "metadata",
            "event_schedule": "events",
        }

        # Cache for provider lookups
        self._provider_cache: dict[str, DataProvider] = {}

        logger.info("Initialized TransformProviderAdapter")

    def register_transform_mapping(self, transform_name: str, provider_name: str) -> None:
        """
        Register a mapping from transform to provider.

        Parameters
        ----------
        transform_name : str
            Name of the transform
        provider_name : str
            Name of the provider to use

        """
        self._transform_mappings[transform_name] = provider_name
        logger.debug(f"Registered mapping: {transform_name} -> {provider_name}")

    def get_provider_for_transform(
        self,
        transform: TransformSpec,
    ) -> BaseStaticProvider | BaseTimeSeriesProvider | None:
        """
        Get the provider for a transform.

        Parameters
        ----------
        transform : TransformSpec
            Transform specification

        Returns
        -------
        DataProvider or None
            Provider for the transform, or None if not mapped

        """
        # Check cache first
        if transform.name in self._provider_cache:
            return self._provider_cache[transform.name]

        # Look up provider name
        provider_name = self._transform_mappings.get(transform.name)
        if provider_name is None:
            logger.debug(f"No provider mapping for transform: {transform.name}")
            return None

        try:
            # Get provider from factory
            provider = self._factory.get_provider(provider_name)
            self._provider_cache[transform.name] = provider
            return provider
        except ValueError as e:
            logger.warning(f"Failed to get provider for {transform.name}: {e}")
            return None

    def load_transform_data(
        self,
        transform: TransformSpec,
        timestamps: PolarsSeries | None,
        instruments: list[str],
    ) -> PolarsDF:
        """
        Load data for a transform.

        Parameters
        ----------
        transform : TransformSpec
            Transform specification
        timestamps : pl.Series, optional
            Timestamps for time series data
        instruments : list[str]
            List of instruments

        Returns
        -------
        pl.DataFrame
            Data for the transform

        """
        if pl is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used
        assert pl is not None

        # Get provider for transform
        provider = self.get_provider_for_transform(transform)

        if provider is None:
            # Return empty DataFrame for unknown transforms
            logger.debug(f"No provider for transform {transform.name}, returning empty DataFrame")
            from typing import cast as _cast
            return _cast(PolarsDF, pl.DataFrame())

        # Load data based on provider type
        if isinstance(provider, BaseStaticProvider):
            return self._load_static_data(provider, instruments)
        elif isinstance(provider, BaseTimeSeriesProvider):
            return self._load_timeseries_data(provider, transform, timestamps, instruments)
        else:
            return self._load_custom_provider_data(provider, timestamps, instruments)

    def _load_static_data(
        self,
        provider: BaseStaticProvider,
        instruments: list[str],
    ) -> PolarsDF:
        """
        Load data from static provider.
        """
        if hasattr(provider, "load_metadata"):
            from typing import cast as _cast
            return _cast(PolarsDF, provider.load_metadata(instruments))
        else:
            logger.warning(f"Static provider {type(provider).__name__} doesn't have load method")
            from typing import cast as _cast
            _pl = pl
            assert _pl is not None
            return _cast(PolarsDF, _pl.DataFrame())

    def _load_timeseries_data(
        self,
        provider: BaseTimeSeriesProvider,
        transform: TransformSpec,
        timestamps: PolarsSeries | None,
        instruments: list[str],
    ) -> PolarsDF:
        """
        Load data from time series provider.
        """
        if timestamps is None:
            logger.warning(f"No timestamps provided for time series transform {transform.name}")
            from typing import cast as _cast
            _pl = pl
            assert _pl is not None
            return _cast(PolarsDF, _pl.DataFrame())
        if timestamps.is_empty():
            logger.warning(f"Empty timestamps for time series transform {transform.name}")
            from typing import cast as _cast
            _pl = pl
            assert _pl is not None
            return _cast(PolarsDF, _pl.DataFrame())

        if not hasattr(provider, "compute_features"):
            logger.warning(
                f"Time series provider {type(provider).__name__} doesn't have compute method",
            )
            from typing import cast as _cast
            _pl = pl
            assert _pl is not None
            return _cast(PolarsDF, _pl.DataFrame())

        # Different providers have different methods
        if transform.name == "calendar":
            return cast(PolarsDF, provider.compute_features(timestamps))
        elif transform.name == "event_schedule":
            return cast(PolarsDF, provider.compute_features(timestamps, instruments=instruments))
        else:
            return cast(PolarsDF, provider.load_timeseries(instruments, timestamps))

    def _load_custom_provider_data(
        self,
        provider: BaseStaticProvider | BaseTimeSeriesProvider,
        timestamps: PolarsSeries | None,
        instruments: list[str],
    ) -> PolarsDF:
        """
        Load data from custom/mock providers.
        """
        if hasattr(provider, "compute_features"):
            return cast(PolarsDF, provider.compute_features(timestamps, instruments=instruments))
        elif hasattr(provider, "load_timeseries"):
            if timestamps is None:
                logger.warning(
                    "No timestamps provided for time series provider; returning empty DataFrame",
                )
                from typing import cast as _cast
                _pl = pl
                assert _pl is not None
                return _cast(PolarsDF, _pl.DataFrame())
            return cast(PolarsDF, provider.load_timeseries(instruments, timestamps))
        elif hasattr(provider, "load_metadata"):
            return cast(PolarsDF, provider.load_metadata(instruments))
        else:
            logger.warning(f"Unknown provider type: {type(provider).__name__}")
            from typing import cast as _cast
            _pl = pl
            assert _pl is not None
            return _cast(PolarsDF, _pl.DataFrame())
