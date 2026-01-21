#!/usr/bin/env python3
"""
SEC identity configuration helpers.

The SEC data endpoints require a descriptive User-Agent that includes
contact information. This module centralizes environment lookups and
builds a compliant identity string without hard-coding values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ml.config._env_utils import ensure_env


@dataclass(frozen=True)
class SecIdentityConfig:
    """
    Configuration for SEC User-Agent identity strings.

    Attributes
    ----------
    identity:
        Fully formatted User-Agent identity string. When supplied, it takes
        precedence over the structured fields.
    name:
        Contact name for the User-Agent identity.
    email:
        Contact email for the User-Agent identity.
    phone:
        Contact phone for the User-Agent identity.
    """

    identity: str | None = None
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    def resolved_identity(self) -> str | None:
        """
        Return the resolved SEC User-Agent identity string.

        Returns
        -------
        str | None
            Fully formatted SEC identity string, if available.
        """
        identity = _normalize(self.identity)
        if identity:
            return identity
        name = _normalize(self.name)
        if not name:
            return None
        contact_parts: list[str] = []
        email = _normalize(self.email)
        phone = _normalize(self.phone)
        if email:
            contact_parts.append(email)
        if phone:
            contact_parts.append(phone)
        if contact_parts:
            contact = " ".join(contact_parts)
            return f"{name} <{contact}>"
        return name

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> SecIdentityConfig:
        """
        Build a config from environment variables.

        Environment variables
        ---------------------
        SEC_IDENTITY:
            Full SEC identity string (takes precedence).
        SEC_USER_AGENT_NAME:
            Contact name for the SEC User-Agent.
        SEC_USER_AGENT_EMAIL:
            Contact email for the SEC User-Agent.
        SEC_USER_AGENT_PHONE:
            Contact phone for the SEC User-Agent.

        Returns
        -------
        SecIdentityConfig
            Parsed SEC identity configuration.
        """
        source = ensure_env(env)
        return cls(
            identity=source.get("SEC_IDENTITY"),
            name=source.get("SEC_USER_AGENT_NAME"),
            email=source.get("SEC_USER_AGENT_EMAIL"),
            phone=source.get("SEC_USER_AGENT_PHONE"),
        )


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


__all__: tuple[str, ...] = ("SecIdentityConfig",)
