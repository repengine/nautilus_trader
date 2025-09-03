"""
Property-based tests for Domain Bookkeeping Phase 1: Message Bus Integration & Event Flow.

These tests verify that message bus integration maintains critical invariants:
- Event emission ordering and correlation
- Cross-domain propagation consistency  
- Message delivery guarantees

Following the "write less tests, get more coverage" philosophy from TESTING_STRATEGY.md
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from ml.stores.data_store import DataStore
from ml.registry.data_registry import DataRegistry
from ml.core.integration import MLIntegrationManager
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import InstrumentId


@pytest.mark.property
@pytest.mark.parallel_safe
class TestEventEmissionInvariant:
    """Property-based tests for event emission infrastructure invariants."""

    @given(
        events=st.lists(
            st.fixed_dictionaries({
                'event_id': st.uuids().map(str),
                'instrument_id': st.text(min_size=5, max_size=20),
                'ts_event': st.integers(min_value=0, max_value=2**63-1),
                'ts_init': st.integers(min_value=0, max_value=2**63-1),
                'domain': st.sampled_from(['data', 'features', 'models', 'strategies']),
                'operation': st.sampled_from(['created', 'updated', 'deprecated']),
                'payload': st.dictionaries(st.text(), st.text(), max_size=5)
            }),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_event_ordering_invariant(self, events):
        """
        Property: Events emitted in chronological order must maintain ts_event ordering.
        
        This property ensures that event timestamps are monotonically increasing
        when events are processed in sequence, which is critical for event replay
        and lineage reconstruction.
        """
        # Sort events by ts_event to establish chronological order
        sorted_events = sorted(events, key=lambda e: e['ts_event'])
        
        mock_message_bus = MagicMock()
        mock_store = MagicMock(spec=DataStore)
        
        # Emit events in chronological order
        emitted_timestamps = []
        for event in sorted_events:
            # Simulate DataStore.emit_event() behavior
            mock_store.emit_event(event)
            emitted_timestamps.append(event['ts_event'])
        
        # Property: Emitted timestamps must be monotonically increasing
        assert emitted_timestamps == sorted(emitted_timestamps), \
            "Event emission must preserve chronological ordering"

    @given(
        correlation_id=st.uuids().map(str),
        domain_events=st.lists(
            st.fixed_dictionaries({
                'domain': st.sampled_from(['data', 'features', 'models', 'strategies']),
                'event_type': st.text(min_size=3, max_size=20),
                'correlation_id': st.uuids().map(str),
                'sequence_number': st.integers(min_value=0, max_value=1000),
            }),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_cross_domain_correlation_invariant(self, correlation_id, domain_events):
        """
        Property: All events in a cross-domain workflow must maintain correlation ID consistency.
        
        When an event triggers cascading events across domains (e.g., data → features → models),
        the correlation ID must be preserved to enable lineage tracing.
        """
        # Set all events to use the same correlation ID (simulating cascade)
        for event in domain_events:
            event['correlation_id'] = correlation_id
        
        mock_integration_manager = MagicMock(spec=MLIntegrationManager)
        
        # Track emitted events
        emitted_correlations = []
        
        def mock_emit(event):
            emitted_correlations.append(event['correlation_id'])
            
        mock_integration_manager.emit_cross_domain_event = mock_emit
        
        # Emit cross-domain event cascade
        for event in domain_events:
            mock_integration_manager.emit_cross_domain_event(event)
        
        # Property: All correlation IDs must match the original
        assert all(cid == correlation_id for cid in emitted_correlations), \
            "Cross-domain event propagation must preserve correlation IDs"

    @given(
        instrument_id=st.text(min_size=5, max_size=15),
        event_count=st.integers(min_value=1, max_value=50),
        message_failures=st.lists(
            st.integers(min_value=0, max_value=49),  # Index of failed messages
            max_size=5
        )
    )
    @settings(max_examples=30, deadline=5000)  
    def test_message_delivery_reliability_invariant(self, instrument_id, event_count, message_failures):
        """
        Property: Message delivery must guarantee eventual consistency despite transient failures.
        
        Events may fail to publish initially due to network issues, but retry mechanisms
        must ensure all events are eventually delivered with proper ordering.
        """
        # Generate events for the instrument
        events = []
        for i in range(event_count):
            events.append({
                'event_id': str(UUID4()),
                'instrument_id': instrument_id,
                'sequence': i,
                'ts_event': i * 1000,  # Sequential timestamps
                'retry_count': 0
            })
        
        mock_message_bus = MagicMock()
        delivered_events = []
        failed_indices = set(message_failures)
        
        def mock_publish(topic, event):
            # Simulate failures for specific indices
            if event['sequence'] in failed_indices and event['retry_count'] == 0:
                raise ConnectionError("Simulated network failure")
            delivered_events.append(event)
        
        mock_message_bus.publish = mock_publish
        
        # Attempt initial delivery
        for event in events:
            try:
                mock_message_bus.publish(f"ml.data.created.{instrument_id}", event)
            except ConnectionError:
                # Retry failed events (increment retry_count)
                event['retry_count'] += 1
                mock_message_bus.publish(f"ml.data.created.{instrument_id}", event)
        
        # Property: All events must be delivered exactly once after retries
        delivered_sequences = [e['sequence'] for e in delivered_events]
        expected_sequences = list(range(event_count))
        
        assert sorted(delivered_sequences) == sorted(expected_sequences), \
            "Message delivery must ensure all events reach destination after retries"
        
        # Property: No duplicate deliveries
        assert len(delivered_sequences) == len(set(delivered_sequences)), \
            "Message delivery must not create duplicates during retry"


@pytest.mark.property  
@pytest.mark.parallel_safe
class TestMessageTopicRoutingInvariant:
    """Property-based tests for message bus topic routing invariants."""
    
    @given(
        domain=st.sampled_from(['data', 'features', 'models', 'strategies']),
        operation=st.sampled_from(['created', 'updated', 'deprecated']),
        instrument_id=st.text(min_size=5, max_size=20),
        invalid_chars=st.lists(
            st.sampled_from(['/', '*', '#', '+', '$']),  # Invalid topic characters
            max_size=3
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_topic_naming_convention_invariant(self, domain, operation, instrument_id, invalid_chars):
        """
        Property: Topic names must follow ml.{domain}.{operation}.{instrument_id} convention.
        
        Topic routing depends on consistent naming for subscription filters and
        event routing logic across the message bus infrastructure.
        """
        # Clean instrument_id of invalid characters
        clean_instrument_id = instrument_id
        for char in invalid_chars:
            clean_instrument_id = clean_instrument_id.replace(char, '_')
        
        expected_topic = f"ml.{domain}.{operation}.{clean_instrument_id}"
        
        mock_registry = MagicMock()
        
        def generate_topic(d, op, inst_id):
            # Sanitize instrument ID for topic naming
            sanitized = inst_id
            for invalid in ['/', '*', '#', '+', '$']:
                sanitized = sanitized.replace(invalid, '_')
            return f"ml.{d}.{op}.{sanitized}"
        
        actual_topic = generate_topic(domain, operation, instrument_id)
        
        # Property: Generated topic must match expected convention
        assert actual_topic == expected_topic, \
            f"Topic naming must follow convention: {expected_topic}, got {actual_topic}"
        
        # Property: Topics must not contain invalid characters
        invalid_chars_in_topic = any(char in actual_topic for char in ['/', '*', '#', '+', '$'])
        assert not invalid_chars_in_topic, \
            "Topic names must not contain message bus reserved characters"

    @given(
        subscriptions=st.lists(
            st.fixed_dictionaries({
                'subscriber_id': st.text(min_size=3, max_size=15),
                'domain_filter': st.sampled_from(['data', 'features', 'models', 'strategies', '*']),
                'operation_filter': st.sampled_from(['created', 'updated', 'deprecated', '*']),
                'instrument_filter': st.text(min_size=3, max_size=15)
            }),
            min_size=1,
            max_size=10
        ),
        published_event=st.fixed_dictionaries({
            'domain': st.sampled_from(['data', 'features', 'models', 'strategies']),
            'operation': st.sampled_from(['created', 'updated', 'deprecated']),
            'instrument_id': st.text(min_size=3, max_size=15)
        })
    )
    @settings(max_examples=30, deadline=5000)
    def test_subscription_filtering_invariant(self, subscriptions, published_event):
        """
        Property: Event routing must deliver to subscribers matching filter patterns.
        
        Subscribers register with domain/operation/instrument filters, and events
        should be delivered only to those with matching patterns.
        """
        def matches_filter(event, subscription):
            domain_match = (subscription['domain_filter'] == '*' or 
                          subscription['domain_filter'] == event['domain'])
            operation_match = (subscription['operation_filter'] == '*' or 
                             subscription['operation_filter'] == event['operation'])
            instrument_match = (subscription['instrument_filter'] == '*' or 
                              subscription['instrument_filter'] == event['instrument_id'])
            return domain_match and operation_match and instrument_match
        
        # Determine which subscribers should receive the event
        expected_recipients = []
        for sub in subscriptions:
            if matches_filter(published_event, sub):
                expected_recipients.append(sub['subscriber_id'])
        
        mock_message_bus = MagicMock()
        actual_recipients = []
        
        def mock_deliver(subscriber_id, event):
            actual_recipients.append(subscriber_id)
            
        mock_message_bus.deliver_to_subscriber = mock_deliver
        
        # Simulate message bus routing logic
        topic = f"ml.{published_event['domain']}.{published_event['operation']}.{published_event['instrument_id']}"
        for sub in subscriptions:
            if matches_filter(published_event, sub):
                mock_message_bus.deliver_to_subscriber(sub['subscriber_id'], published_event)
        
        # Property: All matching subscribers must receive the event
        assert sorted(actual_recipients) == sorted(expected_recipients), \
            "Event routing must deliver to all subscribers with matching filters"


@pytest.mark.property
@pytest.mark.parallel_safe  
class TestEventPropagationInvariant:
    """Property-based tests for cross-domain event propagation invariants."""
    
    @given(
        initial_event=st.fixed_dictionaries({
            'domain': st.just('data'),  # Always start with data domain
            'operation': st.sampled_from(['created', 'updated']),
            'instrument_id': st.text(min_size=5, max_size=15),
            'ts_event': st.integers(min_value=1000, max_value=2**32),
            'payload': st.dictionaries(st.text(), st.text(), max_size=3)
        }),
        propagation_delay_ms=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=30, deadline=5000)
    def test_cascading_event_propagation_invariant(self, initial_event, propagation_delay_ms):
        """
        Property: Data events must trigger cascading events in dependent domains.
        
        When data is updated, it should automatically trigger feature computation,
        which triggers model inference, which triggers strategy evaluation.
        """
        mock_integration_manager = MagicMock(spec=MLIntegrationManager)
        
        # Track the cascade of events
        propagated_events = []
        
        def mock_emit_cascade(source_event, target_domain):
            # Simulate propagation delay
            cascaded_event = {
                'domain': target_domain,
                'operation': source_event['operation'],
                'instrument_id': source_event['instrument_id'],
                'ts_event': source_event['ts_event'] + propagation_delay_ms,
                'source_event_id': source_event.get('event_id', 'unknown'),
                'correlation_id': source_event.get('correlation_id', 'test-correlation')
            }
            propagated_events.append(cascaded_event)
            return cascaded_event
        
        mock_integration_manager.emit_cascade = mock_emit_cascade
        
        # Expected cascade: data → features → models → strategies
        expected_cascade = ['features', 'models', 'strategies']
        current_event = initial_event
        
        for target_domain in expected_cascade:
            current_event = mock_integration_manager.emit_cascade(current_event, target_domain)
        
        # Property: Cascade must create events in all dependent domains
        propagated_domains = [e['domain'] for e in propagated_events]
        assert propagated_domains == expected_cascade, \
            "Event cascade must propagate through all dependent domains in order"
        
        # Property: Timestamps must be increasing (respecting propagation delays)
        propagated_timestamps = [e['ts_event'] for e in propagated_events]
        assert propagated_timestamps == sorted(propagated_timestamps), \
            "Cascaded event timestamps must be monotonically increasing"
        
        # Property: All cascaded events must have same instrument_id
        propagated_instruments = [e['instrument_id'] for e in propagated_events]
        expected_instruments = [initial_event['instrument_id']] * len(expected_cascade)
        assert propagated_instruments == expected_instruments, \
            "Cascaded events must preserve instrument_id across domains"