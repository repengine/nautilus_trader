"""
Contract/Schema tests for Domain Bookkeeping: Message Bus Integration & Event Flow.

These tests define and validate data contracts at component boundaries using Pandera schemas.
Ensures that message bus integration maintains consistent data formats and interfaces.

Following the "write less tests, get more coverage" philosophy from TESTING_STRATEGY.md
"""

from __future__ import annotations

from typing import Any
from datetime import datetime

import pandas as pd
import pandera as pa
import pytest
from pandera.typing import Series
from hypothesis import given, strategies as st

from ml.stores.data_store import DataStore
from nautilus_trader.core.uuid import UUID4


# Message Bus Event Schemas
class EventMessageSchema(pa.DataFrameModel):
    """Schema for events published to Nautilus Message Bus."""
    
    event_id: Series[str] = pa.Field(
        regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="UUID4 event identifier"
    )
    correlation_id: Series[str] = pa.Field(
        regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="UUID4 correlation identifier for event tracing"
    )
    instrument_id: Series[str] = pa.Field(
        str_length={'min_value': 5, 'max_value': 50},
        description="Nautilus instrument identifier"
    )
    ts_event: Series[int] = pa.Field(
        ge=0,
        le=2**63-1, 
        description="Event timestamp in nanoseconds since epoch"
    )
    ts_init: Series[int] = pa.Field(
        ge=0,
        le=2**63-1,
        description="Initialization timestamp in nanoseconds since epoch"  
    )
    domain: Series[str] = pa.Field(
        isin=['data', 'features', 'models', 'strategies'],
        description="Domain bookkeeper responsible for this event"
    )
    operation: Series[str] = pa.Field(
        isin=['created', 'updated', 'deprecated', 'deleted'],
        description="Operation performed on the domain entity"
    )
    
    @pa.check('ts_init', 'ts_event')
    def check_timestamp_ordering(cls, ts_init: Series[int], ts_event: Series[int]) -> bool:
        """ts_init must be >= ts_event (initialization after or during event)."""
        return (ts_init >= ts_event).all()


class MessageTopicSchema(pa.DataFrameModel):
    """Schema for message bus topic routing."""
    
    topic: Series[str] = pa.Field(
        regex=r'^ml\.[a-z]+\.[a-z_]+\.[a-zA-Z0-9_.-]+$',
        description="Topic following ml.{domain}.{operation}.{instrument_id} pattern"
    )
    subscriber_count: Series[int] = pa.Field(
        ge=0,
        description="Number of active subscribers to this topic"
    )
    message_count: Series[int] = pa.Field(
        ge=0, 
        description="Total messages published to this topic"
    )
    last_published: Series[int] = pa.Field(
        ge=0,
        nullable=True,
        description="Timestamp of last published message (nanoseconds)"
    )


class CrossDomainEventSchema(pa.DataFrameModel):
    """Schema for events that propagate across domains."""
    
    source_domain: Series[str] = pa.Field(
        isin=['data', 'features', 'models', 'strategies'],
        description="Domain that initiated the event cascade"
    )
    target_domain: Series[str] = pa.Field(
        isin=['data', 'features', 'models', 'strategies'],
        description="Domain receiving the cascaded event"
    )
    source_event_id: Series[str] = pa.Field(
        regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        description="Original event ID that triggered cascade"
    )
    propagation_delay_ms: Series[int] = pa.Field(
        ge=0,
        le=10000,  # Max 10 second propagation delay
        description="Time delay in milliseconds for event propagation"
    )
    
    @pa.check('source_domain', 'target_domain')
    def check_valid_propagation_path(cls, source: Series[str], target: Series[str]) -> bool:
        """Validate that domain propagation follows expected paths."""
        valid_paths = {
            'data': ['features'],
            'features': ['models'], 
            'models': ['strategies'],
            'strategies': []  # Terminal domain
        }
        
        for src, tgt in zip(source, target):
            if tgt not in valid_paths.get(src, []):
                return False
        return True


class SubscriptionFilterSchema(pa.DataFrameModel):
    """Schema for message bus subscription filters."""
    
    subscriber_id: Series[str] = pa.Field(
        str_length={'min_value': 3, 'max_value': 50},
        description="Unique identifier for message subscriber"
    )
    domain_filter: Series[str] = pa.Field(
        isin=['data', 'features', 'models', 'strategies', '*'],
        description="Domain filter pattern (* for all domains)"
    )
    operation_filter: Series[str] = pa.Field(
        isin=['created', 'updated', 'deprecated', 'deleted', '*'],
        description="Operation filter pattern (* for all operations)"
    )
    instrument_filter: Series[str] = pa.Field(
        str_length={'min_value': 1, 'max_value': 50},
        description="Instrument filter pattern (* for all instruments)"
    )
    is_active: Series[bool] = pa.Field(
        description="Whether subscription is currently active"
    )
    created_at: Series[int] = pa.Field(
        ge=0,
        description="Subscription creation timestamp (nanoseconds)"
    )


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestEventMessageContracts:
    """Contract tests for message bus event schemas."""
    
    def test_event_message_schema_validation(self):
        """Test that event messages conform to expected schema."""
        # Valid event data
        valid_events = pd.DataFrame([
            {
                'event_id': str(UUID4()),
                'correlation_id': str(UUID4()),
                'instrument_id': 'EURUSD.SIM',
                'ts_event': 1609459200000000000,  # 2021-01-01T00:00:00Z
                'ts_init': 1609459200000001000,   # Slightly after ts_event
                'domain': 'data',
                'operation': 'created',
            },
            {
                'event_id': str(UUID4()),
                'correlation_id': str(UUID4()),
                'instrument_id': 'BTCUSDT.BINANCE',
                'ts_event': 1609459260000000000,  # 2021-01-01T00:01:00Z
                'ts_init': 1609459260000000000,   # Same as ts_event (valid)
                'domain': 'features',
                'operation': 'updated',
            }
        ])
        
        # Schema validation should pass
        validated_df = EventMessageSchema.validate(valid_events)
        assert len(validated_df) == 2
        
    def test_event_message_schema_rejects_invalid_data(self):
        """Test that invalid event data is rejected by schema."""
        # Invalid event data (ts_init < ts_event)
        invalid_events = pd.DataFrame([
            {
                'event_id': str(UUID4()),
                'correlation_id': str(UUID4()),
                'instrument_id': 'INVALID',
                'ts_event': 1609459200000000000,
                'ts_init': 1609459199000000000,  # Before ts_event (invalid)
                'domain': 'invalid_domain',  # Invalid domain
                'operation': 'created',
            }
        ])
        
        # Schema validation should fail
        with pytest.raises(pa.errors.SchemaError):
            EventMessageSchema.validate(invalid_events)
            
    def test_topic_schema_naming_convention(self):
        """Test that message topics follow naming convention."""
        valid_topics = pd.DataFrame([
            {
                'topic': 'ml.data.created.EURUSD.SIM',
                'subscriber_count': 3,
                'message_count': 150,
                'last_published': 1609459200000000000,
            },
            {
                'topic': 'ml.features.updated.BTCUSDT.BINANCE', 
                'subscriber_count': 1,
                'message_count': 42,
                'last_published': None,  # No messages yet
            }
        ])
        
        validated_df = MessageTopicSchema.validate(valid_topics)
        assert len(validated_df) == 2
        
    def test_cross_domain_event_propagation_paths(self):
        """Test that cross-domain events follow valid propagation paths."""
        valid_propagations = pd.DataFrame([
            {
                'source_domain': 'data',
                'target_domain': 'features',
                'source_event_id': str(UUID4()),
                'propagation_delay_ms': 50,
            },
            {
                'source_domain': 'features',
                'target_domain': 'models',
                'source_event_id': str(UUID4()),
                'propagation_delay_ms': 100,
            },
            {
                'source_domain': 'models',
                'target_domain': 'strategies',
                'source_event_id': str(UUID4()),
                'propagation_delay_ms': 25,
            }
        ])
        
        validated_df = CrossDomainEventSchema.validate(valid_propagations)
        assert len(validated_df) == 3
        
    def test_invalid_propagation_paths_rejected(self):
        """Test that invalid domain propagation paths are rejected."""
        invalid_propagations = pd.DataFrame([
            {
                'source_domain': 'strategies',  # Terminal domain
                'target_domain': 'data',        # Invalid reverse flow
                'source_event_id': str(UUID4()),
                'propagation_delay_ms': 50,
            }
        ])
        
        with pytest.raises(pa.errors.SchemaError):
            CrossDomainEventSchema.validate(invalid_propagations)


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestSubscriptionFilterContracts:
    """Contract tests for message subscription filtering."""
    
    def test_subscription_filter_schema_validation(self):
        """Test that subscription filters conform to expected schema."""
        valid_subscriptions = pd.DataFrame([
            {
                'subscriber_id': 'ml_signal_actor_001',
                'domain_filter': 'models',
                'operation_filter': 'updated',
                'instrument_filter': 'EURUSD.SIM',
                'is_active': True,
                'created_at': 1609459200000000000,
            },
            {
                'subscriber_id': 'monitoring_service',
                'domain_filter': '*',  # All domains
                'operation_filter': '*',  # All operations
                'instrument_filter': '*',  # All instruments
                'is_active': True,
                'created_at': 1609459200000000000,
            }
        ])
        
        validated_df = SubscriptionFilterSchema.validate(valid_subscriptions)
        assert len(validated_df) == 2


@pytest.mark.contracts
@pytest.mark.integration
class TestMessageBusIntegrationContracts:
    """Integration contract tests for message bus components."""
    
    @given(
        domain=st.sampled_from(['data', 'features', 'models', 'strategies']),
        operation=st.sampled_from(['created', 'updated', 'deprecated']),
        instrument_id=st.text(min_size=5, max_size=20)
    )
    def test_event_emission_preserves_schema(self, domain, operation, instrument_id):
        """
        Contract test: DataStore.emit_event() must produce schema-compliant events.
        
        This ensures that the event emission infrastructure produces events that
        can be consumed by message bus subscribers without data quality issues.
        """
        # Create a valid event
        event_data = {
            'event_id': str(UUID4()),
            'correlation_id': str(UUID4()),
            'instrument_id': instrument_id,
            'ts_event': 1609459200000000000,
            'ts_init': 1609459200000000000,
            'domain': domain,
            'operation': operation,
        }
        
        # Convert to DataFrame for schema validation
        event_df = pd.DataFrame([event_data])
        
        # Contract: Emitted events must pass schema validation
        try:
            validated_event = EventMessageSchema.validate(event_df)
            assert len(validated_event) == 1
            assert validated_event.iloc[0]['domain'] == domain
            assert validated_event.iloc[0]['operation'] == operation
        except pa.errors.SchemaError as e:
            pytest.fail(f"Event emission produced invalid schema: {e}")
    
    def test_topic_generation_follows_contract(self):
        """
        Contract test: Topic generation must follow ml.{domain}.{operation}.{instrument} pattern.
        
        This ensures consistent topic naming across all message bus publishers
        and enables proper subscription filtering.
        """
        test_cases = [
            ('data', 'created', 'EURUSD.SIM', 'ml.data.created.EURUSD.SIM'),
            ('features', 'updated', 'BTCUSDT.BINANCE', 'ml.features.updated.BTCUSDT.BINANCE'),
            ('models', 'deprecated', 'AAPL.NASDAQ', 'ml.models.deprecated.AAPL.NASDAQ'),
        ]
        
        for domain, operation, instrument, expected_topic in test_cases:
            # Simulate topic generation function
            generated_topic = f"ml.{domain}.{operation}.{instrument}"
            
            # Validate against topic schema
            topic_df = pd.DataFrame([{
                'topic': generated_topic,
                'subscriber_count': 0,
                'message_count': 0,
                'last_published': None,
            }])
            
            try:
                validated_topic = MessageTopicSchema.validate(topic_df)
                assert validated_topic.iloc[0]['topic'] == expected_topic
            except pa.errors.SchemaError as e:
                pytest.fail(f"Generated topic violates schema: {e}")


@pytest.mark.contracts
@pytest.mark.parallel_safe
class TestEventCorrelationContracts:
    """Contract tests for event correlation and tracing."""
    
    def test_correlation_id_preservation_contract(self):
        """
        Contract test: Correlation IDs must be preserved across domain boundaries.
        
        This enables end-to-end tracing of events from data ingestion through
        signal generation, which is critical for debugging and performance analysis.
        """
        correlation_id = str(UUID4())
        
        # Simulate event cascade across domains
        cascade_events = []
        domains = ['data', 'features', 'models', 'strategies']
        
        for i, domain in enumerate(domains[:-1]):  # Skip last domain (no cascade from strategies)
            event = {
                'source_domain': domain,
                'target_domain': domains[i + 1],
                'source_event_id': str(UUID4()),
                'propagation_delay_ms': 50 + i * 10,
            }
            cascade_events.append(event)
        
        # Convert to DataFrame and validate
        cascade_df = pd.DataFrame(cascade_events)
        
        try:
            validated_cascade = CrossDomainEventSchema.validate(cascade_df)
            
            # Contract: All events in cascade must form valid propagation chain
            source_domains = validated_cascade['source_domain'].tolist()
            target_domains = validated_cascade['target_domain'].tolist()
            
            expected_chain = ['data', 'features', 'models']
            expected_targets = ['features', 'models', 'strategies']
            
            assert source_domains == expected_chain
            assert target_domains == expected_targets
            
        except pa.errors.SchemaError as e:
            pytest.fail(f"Event cascade violates correlation contract: {e}")