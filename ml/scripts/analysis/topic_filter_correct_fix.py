#!/usr/bin/env python3
"""
Correct fix for the topic filtering algorithm.

The issue: When a pattern ends with '*', it should match exactly one token,
but the current implementation fails if there are additional tokens.

Pattern: "ml.features.computed.*"
Topic:   "ml.features.computed.EURUSD.SIM"

The '*' should match "EURUSD", leaving "SIM" unmatched, which means NO MATCH.
But if the topic was "ml.features.computed.EURUSD", then '*' matches "EURUSD" exactly.

Actually, let me re-read the MQTT spec to understand this correctly...

Looking at the examples in the docstring:
>>> match_topic("ml.features.updated.*", "ml.features.updated.EURUSD.SIM")
True

This suggests that '*' should match "EURUSD.SIM" as a single token, which seems wrong.
Let me check what MQTT actually does...

In MQTT:
- '+' matches exactly one level
- '#' matches zero or more levels
- '+' does NOT match multiple levels

So "ml/features/updated/+" should match "ml/features/updated/EURUSD" but NOT "ml/features/updated/EURUSD/SIM"

But the docstring example suggests '*' SHOULD match multiple levels? Let me check the failing tests...

"""


def match_topic_correct(pattern: str, topic: str) -> bool:
    """
    Corrected version based on understanding the requirements from docstring examples.

    Looking at the docstring, it shows:
    >>> match_topic("ml.features.updated.*", "ml.features.updated.EURUSD.SIM")
    True

    This means '*' should match "EURUSD.SIM" (which is problematic if we're splitting on dots).

    But that contradicts the statement that '*' matches exactly one token.

    Let me implement what the examples actually expect...

    """
    p_parts: list[str] = pattern.split(".") if pattern else []
    t_parts: list[str] = topic.split(".") if topic else []

    i = j = 0
    while i < len(p_parts) and j < len(t_parts):
        token = p_parts[i]
        if token == "#":
            # '#' at end matches the rest
            if i == len(p_parts) - 1:
                return True
            # Try to find a position in t_parts where the remainder matches
            i += 1
            while j <= len(t_parts):
                if match_topic_correct(".".join(p_parts[i:]), ".".join(t_parts[j:])):
                    return True
                j += 1
            return False
        if token == "*":
            # '*' matches exactly one token
            i += 1
            j += 1
            continue
        if token != t_parts[j]:
            return False
        i += 1
        j += 1

    # CORRECTED exhaustion conditions
    # Case 1: Both pattern and topic fully consumed - perfect match
    if i == len(p_parts) and j == len(t_parts):
        return True

    # Case 2: Pattern consumed but topic has remaining tokens - no match
    # (unless the last pattern token was '#' which we handled above)
    if i == len(p_parts) and j < len(t_parts):
        return False

    # Case 3: Topic consumed but pattern has remaining tokens
    if j == len(t_parts) and i < len(p_parts):
        # Only valid if remaining pattern is a single trailing '#'
        if i == len(p_parts) - 1 and p_parts[i] == "#":
            return True
        return False

    # Case 4: Neither fully consumed (shouldn't happen if loop worked correctly)
    return False


if __name__ == "__main__":
    from ml.common.topic_filters import match_topic as original_match_topic

    examples = [
        ("ml.features.updated.*", "ml.features.updated.EURUSD.SIM", True),
        ("events.ml.FEATURE_COMPUTED.*", "events.ml.FEATURE_COMPUTED.EURUSD.SIM", True),
        ("ml.models.created.BTCUSDT.BINANCE", "ml.models.created.BTCUSDT.BINANCE", True),
        ("ml.models.created.*", "ml.models.created", False),
        ("events.ml.SIGNAL_EMITTED.#", "events.ml.SIGNAL_EMITTED", True),
    ]

    for pattern, topic, expected in examples:
        original = original_match_topic(pattern, topic)
        corrected = match_topic_correct(pattern, topic)
        print(pattern, topic, expected, original, corrected)
