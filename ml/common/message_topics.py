"""
Canonical message bus topic builder and normalization utilities.

This module centralizes construction of ML message bus topics to ensure
consistent routing and safe characters in topic segments. It adheres to the
contract tested by MessageTopicSchema:

    ml.{domain}.{operation}.{instrument_id}

- domain:    lowercase letters (a-z)
- operation: lowercase letters and underscore
- instrument_id: [A-Za-z0-9_.-] after normalization

Usage:
    >>> build_topic("data", "created", "EURUSD/SIM")
    'ml.data.created.EURUSD.SIM'

"""

from __future__ import annotations

import re
from typing import Final
from ml.config.events import Stage


_DOMAIN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z]+$")
_OP_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z_]+$")
_RESERVED_CHARS: Final[set[str]] = {"/", "*", "#", "+", "$"}
_ALLOWED_INSTRUMENT_CHARS_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]")


def _normalize_instrument_id(instrument_id: str) -> str:
    """
    Return a normalized instrument identifier safe for topics.

    - Replaces reserved wildcard and separator characters with '.'
    - Removes any character not in [A-Za-z0-9_.-]
    - Collapses consecutive '.' characters
    - Strips leading/trailing '.'
    """
    # Replace explicit reserved chars with '.'
    normalized = "".join("." if ch in _RESERVED_CHARS else ch for ch in instrument_id)
    # Remove any other disallowed characters by replacing with '.' and then filtering
    normalized = _ALLOWED_INSTRUMENT_CHARS_RE.sub(".", normalized)
    # Collapse multiple '.' into a single '.'
    normalized = re.sub(r"\.+", ".", normalized)
    # Trim leading/trailing '.'
    normalized = normalized.strip(".")
    return normalized or "UNKNOWN"


def build_topic(domain: str, operation: str, instrument_id: str) -> str:
    """
    Build a canonical ML topic string.

    Parameters
    ----------
    domain : str
        Topic domain segment (e.g., 'data', 'features', 'models', 'strategies').
    operation : str
        Operation segment (e.g., 'created', 'updated', 'deprecated').
    instrument_id : str
        Raw instrument identifier which will be normalized for topic safety.

    Returns
    -------
    str
        Topic string formatted as 'ml.{domain}.{operation}.{instrument}'.

    Raises
    ------
    ValueError
        If the domain or operation do not match the required patterns.
    """
    if not _DOMAIN_RE.match(domain):
        raise ValueError(f"Invalid domain '{domain}', must match [a-z]+")
    if not _OP_RE.match(operation):
        raise ValueError(f"Invalid operation '{operation}', must match [a-z_]+")

    instrument_norm = _normalize_instrument_id(instrument_id)
    return f"ml.{domain}.{operation}.{instrument_norm}"



def map_stage_to_topic_segments(stage: Stage) -> tuple[str, str]:
    """
    Map a pipeline ``Stage`` to (domain, operation) topic segments.

    The mapping aligns with tests and contracts expecting operations from
    {created, updated, deprecated, deleted}.
    """
    if stage is Stage.DATA_INGESTED:
        return ("data", "created")
    if stage is Stage.CATALOG_WRITTEN:
        return ("data", "updated")
    if stage is Stage.FEATURE_COMPUTED:
        return ("features", "updated")
    if stage is Stage.PREDICTION_EMITTED:
        return ("models", "created")
    if stage is Stage.SIGNAL_EMITTED:
        return ("strategies", "created")
    # Fallback conservative default
    return ("data", "updated")


__all__ = ["build_topic", "map_stage_to_topic_segments"]
