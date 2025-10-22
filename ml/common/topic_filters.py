"""
Topic filter helpers for simple pub/sub matching.

Patterns follow a lightweight variant of MQTT semantics on dot-separated tokens:

- Literal tokens must match exactly (case-sensitive)
- `*` matches exactly one token
- `#` matches zero or more tokens (only valid as a whole token)

Examples
--------
>>> match_topic("ml.features.updated.*", "ml.features.updated.EURUSD.SIM")
True
>>> match_topic("events.ml.FEATURE_COMPUTED.*", "events.ml.FEATURE_COMPUTED.EURUSD.SIM")
True
>>> match_topic("ml.models.created.BTCUSDT.BINANCE", "ml.models.created.BTCUSDT.BINANCE")
True
>>> match_topic("ml.models.created.*", "ml.models.created")
False
>>> match_topic("events.ml.SIGNAL_EMITTED.#", "events.ml.SIGNAL_EMITTED")
True

"""

from __future__ import annotations

from typing import Final


SINGLE_TOKEN_WILDCARD: Final[str] = "*"  # nosec: B105 - wildcard token marker, not a credential
MULTI_TOKEN_WILDCARD: Final[str] = "#"  # nosec: B105 - wildcard token marker, not a credential


def match_topic(pattern: str, topic: str) -> bool:
    """
    Return True if ``topic`` matches the wildcard ``pattern``.

    Both ``pattern`` and ``topic`` are matched on dot-separated tokens. The
    wildcard tokens are:

    - ``*``: matches exactly one token
    - ``#``: matches zero or more tokens (may only appear as a full token)

    """
    p_parts: list[str] = pattern.split(".") if pattern else []
    t_parts: list[str] = topic.split(".") if topic else []

    i = j = 0
    while i < len(p_parts) and j < len(t_parts):
        token = p_parts[i]
        if token == MULTI_TOKEN_WILDCARD:
            # '#' at end matches the rest; if not last, attempt greedy match
            if i == len(p_parts) - 1:
                return True
            # Try to find a position in t_parts where the remainder matches
            i += 1
            while j <= len(t_parts):
                if match_topic(".".join(p_parts[i:]), ".".join(t_parts[j:])):
                    return True
                j += 1
            return False
        if token == SINGLE_TOKEN_WILDCARD:
            i += 1
            j += 1
            continue
        if token != t_parts[j]:
            return False
        i += 1
        j += 1

    # Exhaustion conditions
    # If remaining pattern is a single trailing '#', it matches empty remainder
    if i == len(p_parts) - 1 and p_parts[i] == MULTI_TOKEN_WILDCARD:
        return True
    return i == len(p_parts) and j == len(t_parts)


__all__: Final[list[str]] = ["match_topic"]
