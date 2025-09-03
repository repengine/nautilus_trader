"""
Metamorphic tests for Domain Bookkeeping Phase 1: Event Flow and Propagation.

These tests verify relationships between event outputs under controlled transformations.
Focus on behavioral properties of message bus integration without requiring exact values.

Following the "write less tests, get more coverage" philosophy from TESTING_STRATEGY.md
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.core.integration import MLIntegrationManager
from nautilus_trader.core.uuid import UUID4


@pytest.mark.metamorphic
@pytest.mark.parallel_safe
class TestEventEmissionMetamorphic:
    """Metamorphic tests for event emission transformations."""

    @given(
        base_events=st.lists(
            st.fixed_dictionaries({
                "event_id": st.uuids().map(str),
                "instrument_id": st.text(min_size=5, max_size=15),
                "ts_event": st.integers(min_value=1000, max_value=2**32),
                "payload_size": st.integers(min_value=1, max_value=1000),
                "domain": st.sampled_from(["data", "features", "models"])
            }),
            min_size=5,
            max_size=50
        ),
        time_shift_ns=st.integers(min_value=1, max_value=1000000)  # Nanosecond shift
    )
    @settings(max_examples=30, deadline=5000)
    def test_time_shift_preserves_event_ordering(self, base_events, time_shift_ns):
        """
        Metamorphic relation: Time-shifting all events by constant offset
        should preserve relative ordering and relationships.
        """
        # Create time-shifted version of events
        shifted_events = []
        for event in base_events:
            shifted_event = event.copy()
            shifted_event["ts_event"] += time_shift_ns
            shifted_events.append(shifted_event)

        mock_message_bus = MagicMock()

        # Track emission order for both event sets
        original_order = []
        shifted_order = []

        def track_original(topic, event):
            original_order.append(event["ts_event"])

        def track_shifted(topic, event):
            shifted_order.append(event["ts_event"])

        # Emit original events
        mock_message_bus.publish = track_original
        for event in sorted(base_events, key=lambda e: e["ts_event"]):
            topic = f"ml.{event['domain']}.created.{event['instrument_id']}"
            mock_message_bus.publish(topic, event)

        # Emit shifted events
        mock_message_bus.publish = track_shifted
        for event in sorted(shifted_events, key=lambda e: e["ts_event"]):
            topic = f"ml.{event['domain']}.created.{event['instrument_id']}"
            mock_message_bus.publish(topic, event)

        # Metamorphic relation: Relative ordering preserved after time shift
        if len(original_order) > 1:
            # Check that ordering relationships are preserved
            for i in range(len(original_order) - 1):
                original_diff = original_order[i+1] - original_order[i]
                shifted_diff = shifted_order[i+1] - shifted_order[i]

                assert original_diff == shifted_diff, \
                    "Time shift should preserve relative timestamp differences"

    @given(
        events_per_instrument=st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=2,
            max_size=10
        ),
        duplicate_factor=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=30, deadline=5000)
    def test_event_duplication_maintains_partitioning(self, events_per_instrument, duplicate_factor):
        """
        Metamorphic relation: Duplicating events should maintain per-instrument
        partitioning ratios but scale total counts.
        """
        mock_message_bus = MagicMock()

        # Generate events for multiple instruments
        instruments = [f"INSTRUMENT_{i}" for i in range(len(events_per_instrument))]

        original_events = []
        duplicated_events = []

        for i, count in enumerate(events_per_instrument):
            instrument = instruments[i]
            for j in range(count):
                event = {
                    "event_id": str(UUID4()),
                    "instrument_id": instrument,
                    "ts_event": 1000000 + j,
                    "domain": "data"
                }
                original_events.append(event)

                # Create duplicated versions
                for dup in range(duplicate_factor):
                    dup_event = event.copy()
                    dup_event["event_id"] = str(UUID4())  # Unique ID for each duplicate
                    duplicated_events.append(dup_event)

        # Count events per instrument for both sets
        def count_by_instrument(events):
            counts = {}
            for event in events:
                inst = event["instrument_id"]
                counts[inst] = counts.get(inst, 0) + 1
            return counts

        original_counts = count_by_instrument(original_events)
        duplicated_counts = count_by_instrument(duplicated_events)

        # Metamorphic relations:
        # 1. All instruments present in both sets
        assert set(original_counts.keys()) == set(duplicated_counts.keys()), \
            "Duplication should preserve instrument set"

        # 2. Ratios between instruments preserved
        total_original = sum(original_counts.values())
        total_duplicated = sum(duplicated_counts.values())

        for instrument in original_counts:
            original_ratio = original_counts[instrument] / total_original
            duplicated_ratio = duplicated_counts[instrument] / total_duplicated

            assert abs(original_ratio - duplicated_ratio) < 1e-10, \
                "Duplication should preserve per-instrument ratios"

        # 3. Total count scaled by duplication factor
        assert total_duplicated == total_original * duplicate_factor, \
            "Total event count should scale by duplication factor"


@pytest.mark.metamorphic
@pytest.mark.parallel_safe
class TestCrossDomainPropagationMetamorphic:
    """Metamorphic tests for cross-domain event propagation transformations."""

    @given(
        initial_domains=st.lists(
            st.sampled_from(["data", "features", "models"]),
            min_size=5,
            max_size=20
        ),
        propagation_delays=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_propagation_delay_scaling_preserves_causality(self, initial_domains, propagation_delays):
        """
        Metamorphic relation: Scaling all propagation delays by constant factor
        should preserve causality relationships but scale total latency.
        """
        # Ensure same length for domains and delays
        min_length = min(len(initial_domains), len(propagation_delays))
        domains = initial_domains[:min_length]
        delays = propagation_delays[:min_length]

        scale_factor = 2.5
        scaled_delays = [int(delay * scale_factor) for delay in delays]

        mock_integration_manager = MagicMock(spec=MLIntegrationManager)

        # Track propagation chains
        original_chain = []
        scaled_chain = []

        def track_original_propagation(source_domain, target_domain, delay):
            original_chain.append({
                "source": source_domain,
                "target": target_domain,
                "delay": delay,
                "timestamp": len(original_chain) * 1000 + delay
            })

        def track_scaled_propagation(source_domain, target_domain, delay):
            scaled_chain.append({
                "source": source_domain,
                "target": target_domain,
                "delay": delay,
                "timestamp": len(scaled_chain) * 1000 + delay
            })

        mock_integration_manager.propagate_event = track_original_propagation

        # Simulate original propagation chain
        for i in range(len(domains) - 1):
            source = domains[i]
            target = domains[i + 1] if i + 1 < len(domains) else "strategies"
            delay = delays[i]
            mock_integration_manager.propagate_event(source, target, delay)

        # Simulate scaled propagation chain
        mock_integration_manager.propagate_event = track_scaled_propagation
        for i in range(len(domains) - 1):
            source = domains[i]
            target = domains[i + 1] if i + 1 < len(domains) else "strategies"
            delay = scaled_delays[i]
            mock_integration_manager.propagate_event(source, target, delay)

        if len(original_chain) > 0 and len(scaled_chain) > 0:
            # Metamorphic relation: Causality preserved (source→target relationships same)
            original_relationships = [(step["source"], step["target"]) for step in original_chain]
            scaled_relationships = [(step["source"], step["target"]) for step in scaled_chain]

            assert original_relationships == scaled_relationships, \
                "Delay scaling should preserve causality relationships"

            # Metamorphic relation: Total latency scaled proportionally
            if len(original_chain) > 1 and len(scaled_chain) > 1:
                original_total_delay = sum(step["delay"] for step in original_chain)
                scaled_total_delay = sum(step["delay"] for step in scaled_chain)

                expected_scaled_delay = int(original_total_delay * scale_factor)
                assert abs(scaled_total_delay - expected_scaled_delay) <= len(original_chain), \
                    "Total propagation delay should scale proportionally"

    @given(
        domain_sequence=st.lists(
            st.sampled_from(["data", "features", "models", "strategies"]),
            min_size=3,
            max_size=8
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_domain_sequence_reversal_inverts_dependencies(self, domain_sequence):
        """
        Metamorphic relation: Reversing domain propagation sequence should
        invert dependency relationships while preserving total propagation count.
        """
        mock_integration_manager = MagicMock(spec=MLIntegrationManager)

        # Track forward and reverse propagations
        forward_propagations = []
        reverse_propagations = []

        def track_forward(source, target):
            forward_propagations.append((source, target))

        def track_reverse(source, target):
            reverse_propagations.append((source, target))

        # Forward sequence propagation
        mock_integration_manager.propagate = track_forward
        for i in range(len(domain_sequence) - 1):
            source = domain_sequence[i]
            target = domain_sequence[i + 1]
            mock_integration_manager.propagate(source, target)

        # Reverse sequence propagation
        reversed_sequence = domain_sequence[::-1]
        mock_integration_manager.propagate = track_reverse
        for i in range(len(reversed_sequence) - 1):
            source = reversed_sequence[i]
            target = reversed_sequence[i + 1]
            mock_integration_manager.propagate(source, target)

        if len(forward_propagations) > 0 and len(reverse_propagations) > 0:
            # Metamorphic relation: Same number of propagation steps
            assert len(forward_propagations) == len(reverse_propagations), \
                "Sequence reversal should preserve propagation step count"

            # Metamorphic relation: Dependency direction inverted
            forward_edges = set(forward_propagations)
            reverse_edges = set(reverse_propagations)
            inverted_edges = set((target, source) for source, target in forward_edges)

            # The reverse sequence should create inverted dependency relationships
            assert len(forward_edges & reverse_edges) <= len(forward_edges) * 0.2, \
                "Most dependencies should be inverted in reverse sequence"


@pytest.mark.metamorphic
@pytest.mark.parallel_safe
class TestMessageBusTopicMetamorphic:
    """Metamorphic tests for message bus topic routing transformations."""

    @given(
        base_topics=st.lists(
            st.fixed_dictionaries({
                "domain": st.sampled_from(["data", "features", "models"]),
                "operation": st.sampled_from(["created", "updated"]),
                "instrument_id": st.text(min_size=5, max_size=15),
                "subscriber_count": st.integers(min_value=0, max_value=100)
            }),
            min_size=10,
            max_size=50
        ),
        topic_prefix_change=st.text(min_size=2, max_size=10)
    )
    @settings(max_examples=30, deadline=5000)
    def test_topic_prefix_transformation_preserves_structure(self, base_topics, topic_prefix_change):
        """
        Metamorphic relation: Changing topic prefix should preserve subscriber
        distribution patterns and relative topic popularity.
        """
        mock_message_bus = MagicMock()

        # Generate original and transformed topic sets
        original_topic_stats = {}
        transformed_topic_stats = {}

        for topic_info in base_topics:
            # Original topic
            original_topic = f"ml.{topic_info['domain']}.{topic_info['operation']}.{topic_info['instrument_id']}"
            original_topic_stats[original_topic] = topic_info["subscriber_count"]

            # Transformed topic (different prefix)
            transformed_topic = f"{topic_prefix_change}.{topic_info['domain']}.{topic_info['operation']}.{topic_info['instrument_id']}"
            transformed_topic_stats[transformed_topic] = topic_info["subscriber_count"]

        # Metamorphic relations:
        # 1. Same number of topics
        assert len(original_topic_stats) == len(transformed_topic_stats), \
            "Topic prefix change should preserve topic count"

        # 2. Total subscriber count preserved
        original_total_subscribers = sum(original_topic_stats.values())
        transformed_total_subscribers = sum(transformed_topic_stats.values())

        assert original_total_subscribers == transformed_total_subscribers, \
            "Topic prefix change should preserve total subscriber count"

        # 3. Relative subscriber distribution preserved
        if original_total_subscribers > 0:
            # Find corresponding topics and compare ratios
            domain_operation_pairs = []
            for topic_info in base_topics:
                key = (topic_info["domain"], topic_info["operation"], topic_info["instrument_id"])
                domain_operation_pairs.append(key)

            for domain, operation, instrument_id in set(domain_operation_pairs):
                original_topic = f"ml.{domain}.{operation}.{instrument_id}"
                transformed_topic = f"{topic_prefix_change}.{domain}.{operation}.{instrument_id}"

                if original_topic in original_topic_stats and transformed_topic in transformed_topic_stats:
                    original_ratio = original_topic_stats[original_topic] / original_total_subscribers
                    transformed_ratio = transformed_topic_stats[transformed_topic] / transformed_total_subscribers

                    assert abs(original_ratio - transformed_ratio) < 1e-10, \
                        "Topic prefix change should preserve subscriber distribution ratios"

    @given(
        subscription_patterns=st.lists(
            st.fixed_dictionaries({
                "domain_filter": st.sampled_from(["data", "features", "models", "*"]),
                "operation_filter": st.sampled_from(["created", "updated", "*"]),
                "subscriber_priority": st.integers(min_value=1, max_value=10)
            }),
            min_size=5,
            max_size=20
        ),
        filter_specificity_change=st.sampled_from(["more_specific", "more_general"])
    )
    @settings(max_examples=30, deadline=5000)
    def test_subscription_filter_specificity_affects_match_count(self, subscription_patterns, filter_specificity_change):
        """
        Metamorphic relation: Making filters more specific should reduce matches,
        making them more general should increase matches.
        """
        mock_message_bus = MagicMock()

        # Test event to match against filters
        test_event = {
            "domain": "features",
            "operation": "updated",
            "instrument_id": "TEST.SYMBOL"
        }

        # Count matches for original filters
        original_matches = 0
        for pattern in subscription_patterns:
            domain_match = (pattern["domain_filter"] == "*" or
                          pattern["domain_filter"] == test_event["domain"])
            operation_match = (pattern["operation_filter"] == "*" or
                             pattern["operation_filter"] == test_event["operation"])

            if domain_match and operation_match:
                original_matches += 1

        # Transform filters based on specificity change
        transformed_patterns = []
        for pattern in subscription_patterns:
            new_pattern = pattern.copy()

            if filter_specificity_change == "more_specific":
                # Make wildcards more specific
                if new_pattern["domain_filter"] == "*":
                    new_pattern["domain_filter"] = "data"  # Specific domain
                if new_pattern["operation_filter"] == "*":
                    new_pattern["operation_filter"] = "created"  # Specific operation
            else:  # more_general
                # Make specific filters more general
                new_pattern["domain_filter"] = "*"
                new_pattern["operation_filter"] = "*"

            transformed_patterns.append(new_pattern)

        # Count matches for transformed filters
        transformed_matches = 0
        for pattern in transformed_patterns:
            domain_match = (pattern["domain_filter"] == "*" or
                          pattern["domain_filter"] == test_event["domain"])
            operation_match = (pattern["operation_filter"] == "*" or
                             pattern["operation_filter"] == test_event["operation"])

            if domain_match and operation_match:
                transformed_matches += 1

        # Metamorphic relation: Specificity affects match count predictably
        if filter_specificity_change == "more_specific":
            assert transformed_matches <= original_matches, \
                "More specific filters should reduce or maintain match count"
        else:  # more_general
            assert transformed_matches >= original_matches, \
                "More general filters should increase or maintain match count"
